"""
Novelty Detector for Kosmos.

Prevents redundant tasks across research cycles using semantic similarity.

Key Innovation: Vector-based semantic matching with sentence transformers.

Design:
- Index past tasks as embeddings (384-dimensional vectors)
- Compute cosine similarity for new tasks
- Flag tasks above similarity threshold as redundant

Threshold: 75% similarity = redundant (configurable)

Performance: O(n) similarity check vs O(n²) pairwise comparison
"""

import logging
from typing import Dict, List, Optional, Set
import numpy as np

logger = logging.getLogger(__name__)


class NoveltyDetector:
    """
    Detects redundant tasks using semantic similarity.

    Uses sentence-transformers to compute embeddings and cosine similarity.

    Example:
        detector = NoveltyDetector(novelty_threshold=0.75)
        detector.index_past_tasks(previous_tasks)

        novelty_result = detector.check_task_novelty(new_task)
        if novelty_result['is_novel']:
            # Proceed with task
        else:
            # Task is redundant, similar to: novelty_result['similar_tasks']
    """

    def __init__(
        self,
        novelty_threshold: float = 0.75,
        model_name: str = "all-MiniLM-L6-v2",
        use_sentence_transformers: bool = True
    ):
        """
        Initialize Novelty Detector.

        Args:
            novelty_threshold: Similarity threshold (0-1) above which tasks are redundant
            model_name: Sentence transformer model name
            use_sentence_transformers: If False, uses simple token-based similarity
        """
        self.novelty_threshold = novelty_threshold
        self.model_name = model_name
        self.use_sentence_transformers = use_sentence_transformers

        # Task embeddings cache
        self.task_embeddings = []
        self.task_texts = []
        self.task_metadata = []

        # Initialize model
        self.model = None
        if use_sentence_transformers:
            self._initialize_model()

    def _initialize_model(self):
        """Initialize sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded sentence transformer: {self.model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Using fallback token-based similarity. "
                "Install with: pip install sentence-transformers"
            )
            self.use_sentence_transformers = False
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            self.use_sentence_transformers = False

    def index_past_tasks(self, tasks: List[Dict]):
        """
        Index past tasks for similarity search.

        Args:
            tasks: List of task dictionaries with 'type' and 'description'
        """
        if not tasks:
            return

        # Create text representations
        task_texts = []
        for task in tasks:
            task_type = task.get('type', 'unknown')
            description = task.get('description', '')
            task_text = f"{task_type}: {description}"
            task_texts.append(task_text)

        # Compute embeddings
        if self.use_sentence_transformers and self.model:
            try:
                embeddings = self.model.encode(task_texts)
                self.task_embeddings.extend(embeddings)
                self.task_texts.extend(task_texts)
                self.task_metadata.extend(tasks)

                logger.info(f"Indexed {len(tasks)} tasks, total: {len(self.task_texts)}")

            except Exception as e:
                logger.error(f"Failed to encode tasks: {e}")
                # Fallback to storing texts only
                self.task_texts.extend(task_texts)
                self.task_metadata.extend(tasks)
        else:
            # Store texts for token-based similarity
            self.task_texts.extend(task_texts)
            self.task_metadata.extend(tasks)

    def check_task_novelty(self, task: Dict) -> Dict:
        """
        Check if task is novel compared to indexed tasks.

        Args:
            task: Task dictionary with 'type' and 'description'

        Returns:
            Dictionary with:
            - is_novel: bool
            - novelty_score: float (1.0 = completely novel, 0.0 = identical)
            - max_similarity: float
            - similar_tasks: List of similar task dictionaries
        """
        # Create task text
        task_type = task.get('type', 'unknown')
        description = task.get('description', '')
        task_text = f"{task_type}: {description}"

        # If no past tasks indexed, it's novel
        if not self.task_texts:
            return {
                'is_novel': True,
                'novelty_score': 1.0,
                'max_similarity': 0.0,
                'similar_tasks': []
            }

        # Compute similarities
        if self.use_sentence_transformers and self.task_embeddings:
            similarities = self._compute_semantic_similarities(task_text)
        else:
            similarities = self._compute_token_similarities(task_text)

        # Find max similarity
        max_similarity = float(np.max(similarities)) if len(similarities) > 0 else 0.0
        novelty_score = 1.0 - max_similarity

        # Determine novelty
        is_novel = max_similarity < self.novelty_threshold

        # Find top 3 similar tasks
        similar_indices = np.argsort(similarities)[::-1][:3]
        similar_tasks = []

        for idx in similar_indices:
            if idx < len(self.task_metadata) and similarities[idx] > 0.6:
                similar_tasks.append({
                    'task': self.task_metadata[idx],
                    'similarity': float(similarities[idx])
                })

        return {
            'is_novel': is_novel,
            'novelty_score': novelty_score,
            'max_similarity': float(max_similarity),
            'similar_tasks': similar_tasks
        }

    def _compute_semantic_similarities(self, task_text: str) -> np.ndarray:
        """
        Compute semantic similarities using sentence embeddings.

        Args:
            task_text: Task text to compare

        Returns:
            Array of cosine similarities
        """
        # Encode new task
        task_embedding = self.model.encode([task_text])[0]

        # Compute cosine similarities
        task_embeddings_array = np.array(self.task_embeddings)

        # Normalize vectors
        task_embedding_norm = task_embedding / np.linalg.norm(task_embedding)
        embeddings_norm = task_embeddings_array / np.linalg.norm(
            task_embeddings_array, axis=1, keepdims=True
        )

        # Cosine similarity
        similarities = np.dot(embeddings_norm, task_embedding_norm)

        return similarities

    def _compute_token_similarities(self, task_text: str) -> np.ndarray:
        """
        Compute token-based similarities (fallback method).

        Uses Jaccard similarity on word tokens.

        Args:
            task_text: Task text to compare

        Returns:
            Array of Jaccard similarities
        """
        # Tokenize
        task_tokens = set(task_text.lower().split())

        similarities = []
        for past_text in self.task_texts:
            past_tokens = set(past_text.lower().split())

            # Jaccard similarity
            intersection = len(task_tokens & past_tokens)
            union = len(task_tokens | past_tokens)

            similarity = intersection / union if union > 0 else 0.0
            similarities.append(similarity)

        return np.array(similarities)

    def check_plan_novelty(self, plan: Dict) -> Dict:
        """
        Check novelty of entire plan.

        Args:
            plan: Plan dictionary with 'tasks' list

        Returns:
            Dictionary with:
            - plan_novelty_score: Average novelty across tasks
            - redundant_task_count: Number of redundant tasks
            - novel_task_count: Number of novel tasks
            - task_novelties: List of per-task novelty results
        """
        tasks = plan.get('tasks', [])
        if not tasks:
            return {
                'plan_novelty_score': 1.0,
                'redundant_task_count': 0,
                'novel_task_count': 0,
                'task_novelties': []
            }

        # Check each task
        task_novelties = []
        for task in tasks:
            novelty_result = self.check_task_novelty(task)
            task_novelties.append(novelty_result)

        # Aggregate
        novelty_scores = [t['novelty_score'] for t in task_novelties]
        plan_novelty_score = sum(novelty_scores) / len(novelty_scores)

        novel_count = sum(1 for t in task_novelties if t['is_novel'])
        redundant_count = len(task_novelties) - novel_count

        return {
            'plan_novelty_score': plan_novelty_score,
            'redundant_task_count': redundant_count,
            'novel_task_count': novel_count,
            'task_novelties': task_novelties
        }

    def clear_index(self):
        """Clear all indexed tasks."""
        self.task_embeddings = []
        self.task_texts = []
        self.task_metadata = []
        logger.info("Cleared novelty detector index")

    def get_statistics(self) -> Dict:
        """Get statistics about indexed tasks."""
        return {
            'total_indexed_tasks': len(self.task_texts),
            'has_embeddings': len(self.task_embeddings) > 0,
            'novelty_threshold': self.novelty_threshold,
            'using_semantic_similarity': self.use_sentence_transformers,
            'model': self.model_name if self.use_sentence_transformers else 'token-based'
        }

    def filter_redundant_tasks(
        self,
        tasks: List[Dict],
        keep_most_novel: bool = True
    ) -> List[Dict]:
        """
        Filter redundant tasks from a list.

        Args:
            tasks: List of task dictionaries
            keep_most_novel: If True, keeps most novel task among redundant ones

        Returns:
            Filtered list of novel tasks
        """
        if not tasks:
            return []

        novel_tasks = []
        seen_tasks = []

        for task in tasks:
            # Temporarily index seen tasks
            temp_embeddings = self.task_embeddings.copy()
            temp_texts = self.task_texts.copy()
            temp_metadata = self.task_metadata.copy()

            # Add seen tasks to index
            if seen_tasks:
                self.index_past_tasks(seen_tasks)

            # Check novelty
            novelty_result = self.check_task_novelty(task)

            if novelty_result['is_novel']:
                novel_tasks.append(task)
                seen_tasks.append(task)

            # Restore original index
            self.task_embeddings = temp_embeddings
            self.task_texts = temp_texts
            self.task_metadata = temp_metadata

        return novel_tasks
