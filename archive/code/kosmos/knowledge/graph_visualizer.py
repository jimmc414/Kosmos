"""
Knowledge graph visualization with static and interactive modes.

Supports:
- Static visualizations using NetworkX + Matplotlib
- Interactive visualizations using Plotly
- Multiple layout algorithms
- Filtering and highlighting capabilities
- Specialized views (citations, concepts, authors)
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from enum import Enum

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import numpy as np

from kosmos.knowledge.graph import get_knowledge_graph, KnowledgeGraph

logger = logging.getLogger(__name__)


class LayoutAlgorithm(Enum):
    """Available graph layout algorithms."""
    SPRING = "spring"
    HIERARCHICAL = "hierarchical"
    CIRCULAR = "circular"
    KAMADA_KAWAI = "kamada_kawai"
    SPECTRAL = "spectral"


class VisualizationMode(Enum):
    """Visualization modes."""
    STATIC = "static"
    INTERACTIVE = "interactive"


class GraphVisualizer:
    """
    Knowledge graph visualizer with dual modes (static + interactive).

    Provides publication-quality static graphs and explorable interactive
    visualizations of the knowledge graph.
    """

    def __init__(
        self,
        graph: Optional[KnowledgeGraph] = None,
        default_layout: LayoutAlgorithm = LayoutAlgorithm.SPRING
    ):
        """
        Initialize graph visualizer.

        Args:
            graph: KnowledgeGraph instance (default: singleton)
            default_layout: Default layout algorithm

        Example:
            ```python
            visualizer = GraphVisualizer()

            # Static visualization
            visualizer.visualize_static(
                output_file="graph.png",
                layout=LayoutAlgorithm.SPRING
            )

            # Interactive visualization
            visualizer.visualize_interactive(
                output_file="graph.html"
            )
            ```
        """
        self.graph = graph or get_knowledge_graph()
        self.default_layout = default_layout

        # Color schemes
        self.node_colors = {
            "Paper": "#3498db",      # Blue
            "Author": "#e74c3c",     # Red
            "Concept": "#2ecc71",    # Green
            "Method": "#f39c12"      # Orange
        }

        self.edge_colors = {
            "CITES": "#95a5a6",      # Gray
            "AUTHORED": "#e74c3c",   # Red
            "DISCUSSES": "#2ecc71",  # Green
            "USES_METHOD": "#f39c12", # Orange
            "RELATED_TO": "#9b59b6"  # Purple
        }

        logger.info("Initialized GraphVisualizer")

    def visualize_static(
        self,
        output_file: str = "knowledge_graph.png",
        layout: Optional[LayoutAlgorithm] = None,
        node_types: Optional[List[str]] = None,
        max_nodes: int = 100,
        figsize: Tuple[int, int] = (16, 12),
        dpi: int = 300,
        title: Optional[str] = None
    ):
        """
        Create static visualization using NetworkX and Matplotlib.

        Args:
            output_file: Output file path (PNG, SVG, or PDF)
            layout: Layout algorithm to use
            node_types: Node types to include (None = all)
            max_nodes: Maximum nodes to visualize
            figsize: Figure size in inches
            dpi: DPI for raster outputs
            title: Graph title

        Example:
            ```python
            visualizer.visualize_static(
                output_file="citations.png",
                layout=LayoutAlgorithm.HIERARCHICAL,
                node_types=["Paper"],
                title="Citation Network"
            )
            ```
        """
        logger.info(f"Creating static visualization: {output_file}")

        # Build NetworkX graph
        G = self._build_networkx_graph(node_types=node_types, max_nodes=max_nodes)

        if len(G.nodes()) == 0:
            logger.warning("No nodes to visualize")
            return

        # Compute layout
        layout_algo = layout or self.default_layout
        pos = self._compute_layout(G, layout_algo)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Draw nodes by type
        for node_type in self.node_colors.keys():
            nodes = [n for n, d in G.nodes(data=True) if d.get("type") == node_type]
            if nodes:
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=nodes,
                    node_color=self.node_colors[node_type],
                    node_size=500,
                    alpha=0.7,
                    label=node_type,
                    ax=ax
                )

        # Draw edges by type
        for edge_type, color in self.edge_colors.items():
            edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == edge_type]
            if edges:
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=edges,
                    edge_color=color,
                    alpha=0.5,
                    width=1.5,
                    arrows=True,
                    arrowsize=10,
                    ax=ax
                )

        # Draw labels for important nodes
        labels = {}
        for node, data in G.nodes(data=True):
            if data.get("importance", 0) > 0.7 or G.degree(node) > 5:
                labels[node] = data.get("label", node)[:30]  # Truncate

        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=8,
            font_weight="bold",
            ax=ax
        )

        # Add legend
        ax.legend(loc="upper left", fontsize=10)

        # Title
        if title:
            ax.set_title(title, fontsize=16, fontweight="bold")

        ax.axis("off")
        plt.tight_layout()

        # Save
        plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved static visualization to {output_file}")

        plt.close()

    def visualize_interactive(
        self,
        output_file: str = "knowledge_graph.html",
        node_types: Optional[List[str]] = None,
        max_nodes: int = 200,
        layout: Optional[LayoutAlgorithm] = None,
        title: Optional[str] = None
    ):
        """
        Create interactive visualization using Plotly.

        Args:
            output_file: Output HTML file path
            node_types: Node types to include
            max_nodes: Maximum nodes to visualize
            layout: Layout algorithm
            title: Graph title

        Example:
            ```python
            visualizer.visualize_interactive(
                output_file="interactive_graph.html",
                node_types=["Paper", "Concept"],
                title="Papers and Concepts"
            )
            ```
        """
        logger.info(f"Creating interactive visualization: {output_file}")

        # Build NetworkX graph
        G = self._build_networkx_graph(node_types=node_types, max_nodes=max_nodes)

        if len(G.nodes()) == 0:
            logger.warning("No nodes to visualize")
            return

        # Compute layout
        layout_algo = layout or self.default_layout
        pos = self._compute_layout(G, layout_algo)

        # Create edge traces
        edge_traces = []

        for edge_type, color in self.edge_colors.items():
            edge_x = []
            edge_y = []
            edge_text = []

            for u, v, data in G.edges(data=True):
                if data.get("type") == edge_type:
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

            if edge_x:
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=1, color=color),
                    hoverinfo='none',
                    mode='lines',
                    name=edge_type,
                    showlegend=True
                )
                edge_traces.append(edge_trace)

        # Create node traces by type
        node_traces = []

        for node_type, color in self.node_colors.items():
            node_x = []
            node_y = []
            node_text = []
            node_hover = []

            for node, data in G.nodes(data=True):
                if data.get("type") == node_type:
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)

                    label = data.get("label", node)
                    node_text.append(label[:30])

                    # Build hover text with metadata
                    hover_text = f"<b>{label}</b><br>"
                    hover_text += f"Type: {node_type}<br>"
                    hover_text += f"Degree: {G.degree(node)}<br>"

                    # Add type-specific info
                    if node_type == "Paper":
                        hover_text += f"Year: {data.get('year', 'N/A')}<br>"
                        hover_text += f"Citations: {data.get('citation_count', 0)}"
                    elif node_type == "Concept":
                        hover_text += f"Domain: {data.get('domain', 'N/A')}<br>"
                        hover_text += f"Frequency: {data.get('frequency', 0)}"
                    elif node_type == "Method":
                        hover_text += f"Category: {data.get('category', 'N/A')}"
                    elif node_type == "Author":
                        hover_text += f"Papers: {data.get('paper_count', 0)}"

                    node_hover.append(hover_text)

            if node_x:
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    hoverinfo='text',
                    hovertext=node_hover,
                    marker=dict(
                        color=color,
                        size=10,
                        line=dict(width=1, color='white')
                    ),
                    text=node_text,
                    name=node_type,
                    showlegend=True
                )
                node_traces.append(node_trace)

        # Create figure
        fig = go.Figure(
            data=edge_traces + node_traces,
            layout=go.Layout(
                title=title or "Knowledge Graph",
                titlefont_size=16,
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                height=800
            )
        )

        # Save
        fig.write_html(output_file)
        logger.info(f"Saved interactive visualization to {output_file}")

    def visualize_citation_network(
        self,
        paper_id: str,
        max_depth: int = 2,
        output_file: str = "citation_network.png",
        mode: VisualizationMode = VisualizationMode.STATIC
    ):
        """
        Visualize citation network around a paper.

        Args:
            paper_id: Center paper ID
            max_depth: Citation depth
            output_file: Output file
            mode: Visualization mode (static or interactive)

        Example:
            ```python
            visualizer.visualize_citation_network(
                paper_id="arxiv:1234.5678",
                max_depth=2,
                mode=VisualizationMode.INTERACTIVE
            )
            ```
        """
        logger.info(f"Visualizing citation network for {paper_id}")

        # Get citations from graph
        citations = self.graph.get_citations(paper_id, depth=max_depth)

        if not citations:
            logger.warning(f"No citations found for {paper_id}")
            return

        # Build subgraph
        G = nx.DiGraph()

        # Add center paper
        center_paper = self.graph.get_paper(paper_id)
        if center_paper:
            G.add_node(paper_id, type="Paper", label=center_paper["title"], importance=1.0)

        # Add cited papers
        for citation in citations:
            cited = citation["paper"]
            cited_id = cited.get("id")

            G.add_node(
                cited_id,
                type="Paper",
                label=cited.get("title", "Unknown"),
                importance=0.5
            )
            G.add_edge(paper_id, cited_id, type="CITES")

        # Visualize
        if mode == VisualizationMode.STATIC:
            self._visualize_networkx_static(
                G,
                output_file=output_file,
                title=f"Citation Network (depth={max_depth})"
            )
        else:
            self._visualize_networkx_interactive(
                G,
                output_file=output_file,
                title=f"Citation Network (depth={max_depth})"
            )

    def visualize_concept_network(
        self,
        concept_name: str,
        output_file: str = "concept_network.png",
        mode: VisualizationMode = VisualizationMode.STATIC,
        max_related: int = 20
    ):
        """
        Visualize concept and related concepts.

        Args:
            concept_name: Center concept
            output_file: Output file
            mode: Visualization mode
            max_related: Maximum related concepts

        Example:
            ```python
            visualizer.visualize_concept_network(
                concept_name="CRISPR",
                mode=VisualizationMode.INTERACTIVE
            )
            ```
        """
        logger.info(f"Visualizing concept network for {concept_name}")

        # Get related concepts
        related = self.graph.get_related_concepts(concept_name, limit=max_related)
        cooccurring = self.graph.get_concept_cooccurrence(concept_name)

        # Build graph
        G = nx.Graph()

        # Add center concept
        G.add_node(concept_name, type="Concept", label=concept_name, importance=1.0)

        # Add related concepts
        for rel in related:
            concept = rel["concept"]
            concept_id = concept["name"]

            G.add_node(concept_id, type="Concept", label=concept_id, importance=0.7)
            G.add_edge(concept_name, concept_id, type="RELATED_TO")

        # Add co-occurring concepts
        for coocc in cooccurring[:10]:
            concept = coocc["concept"]
            concept_id = concept["name"]

            if concept_id not in G:
                G.add_node(concept_id, type="Concept", label=concept_id, importance=0.5)
                G.add_edge(concept_name, concept_id, type="RELATED_TO")

        # Visualize
        if mode == VisualizationMode.STATIC:
            self._visualize_networkx_static(
                G,
                output_file=output_file,
                title=f"Concept Network: {concept_name}"
            )
        else:
            self._visualize_networkx_interactive(
                G,
                output_file=output_file,
                title=f"Concept Network: {concept_name}"
            )

    def visualize_author_network(
        self,
        author_name: str,
        output_file: str = "author_network.png",
        mode: VisualizationMode = VisualizationMode.STATIC,
        include_coauthors: bool = True
    ):
        """
        Visualize author's papers and co-authors.

        Args:
            author_name: Author name
            output_file: Output file
            mode: Visualization mode
            include_coauthors: Whether to include co-authors

        Example:
            ```python
            visualizer.visualize_author_network(
                author_name="John Doe",
                include_coauthors=True
            )
            ```
        """
        logger.info(f"Visualizing author network for {author_name}")

        # Get author's papers
        papers = self.graph.get_author_papers(author_name)

        # Build graph
        G = nx.Graph()

        # Add author
        G.add_node(author_name, type="Author", label=author_name, importance=1.0)

        # Add papers
        for paper_data in papers:
            paper = paper_data["paper"]
            paper_id = paper["id"]

            G.add_node(paper_id, type="Paper", label=paper["title"], importance=0.7)
            G.add_edge(author_name, paper_id, type="AUTHORED")

        # Visualize
        if mode == VisualizationMode.STATIC:
            self._visualize_networkx_static(
                G,
                output_file=output_file,
                title=f"Author Network: {author_name}"
            )
        else:
            self._visualize_networkx_interactive(
                G,
                output_file=output_file,
                title=f"Author Network: {author_name}"
            )

    def _build_networkx_graph(
        self,
        node_types: Optional[List[str]] = None,
        max_nodes: int = 100
    ) -> nx.MultiDiGraph:
        """
        Build NetworkX graph from Neo4j knowledge graph.

        Args:
            node_types: Node types to include
            max_nodes: Maximum nodes

        Returns:
            NetworkX MultiDiGraph
        """
        G = nx.MultiDiGraph()

        # Query Neo4j for nodes and edges
        node_types_str = "|".join(node_types) if node_types else "Paper|Author|Concept|Method"

        # Get nodes
        query = f"""
        MATCH (n)
        WHERE any(label IN labels(n) WHERE label IN split($node_types, '|'))
        RETURN n, labels(n)[0] as type
        LIMIT $max_nodes
        """

        try:
            results = self.graph.graph.run(
                query,
                node_types=node_types_str,
                max_nodes=max_nodes
            ).data()

            for result in results:
                node = result["n"]
                node_type = result["type"]
                node_id = node.get("id") or node.get("name")

                G.add_node(
                    node_id,
                    type=node_type,
                    label=node.get("title") or node.get("name", "Unknown"),
                    **dict(node)
                )

            # Get edges
            edge_query = """
            MATCH (a)-[r]->(b)
            WHERE id(a) IN $node_ids AND id(b) IN $node_ids
            RETURN a, b, type(r) as rel_type, r
            """

            # This is a simplified version - full implementation would
            # properly query relationships between loaded nodes

        except Exception as e:
            logger.error(f"Error building NetworkX graph: {e}")

        return G

    def _compute_layout(
        self,
        G: nx.Graph,
        layout: LayoutAlgorithm
    ) -> Dict[Any, Tuple[float, float]]:
        """
        Compute graph layout positions.

        Args:
            G: NetworkX graph
            layout: Layout algorithm

        Returns:
            Dictionary mapping nodes to (x, y) positions
        """
        if layout == LayoutAlgorithm.SPRING:
            return nx.spring_layout(G, k=0.5, iterations=50)
        elif layout == LayoutAlgorithm.HIERARCHICAL:
            return nx.kamada_kawai_layout(G)
        elif layout == LayoutAlgorithm.CIRCULAR:
            return nx.circular_layout(G)
        elif layout == LayoutAlgorithm.KAMADA_KAWAI:
            return nx.kamada_kawai_layout(G)
        elif layout == LayoutAlgorithm.SPECTRAL:
            return nx.spectral_layout(G)
        else:
            return nx.spring_layout(G)

    def _visualize_networkx_static(
        self,
        G: nx.Graph,
        output_file: str,
        title: str
    ):
        """Helper for static NetworkX visualization."""
        pos = self._compute_layout(G, self.default_layout)

        fig, ax = plt.subplots(figsize=(14, 10), dpi=150)

        # Draw nodes by type
        for node_type in self.node_colors.keys():
            nodes = [n for n, d in G.nodes(data=True) if d.get("type") == node_type]
            if nodes:
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=nodes,
                    node_color=self.node_colors[node_type],
                    node_size=300,
                    alpha=0.7,
                    label=node_type,
                    ax=ax
                )

        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, ax=ax)

        # Draw labels
        labels = {n: d.get("label", n)[:25] for n, d in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)

        ax.legend()
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved to {output_file}")

    def _visualize_networkx_interactive(
        self,
        G: nx.Graph,
        output_file: str,
        title: str
    ):
        """Helper for interactive Plotly visualization."""
        self.visualize_interactive(
            output_file=output_file,
            max_nodes=len(G.nodes()),
            title=title
        )


# Singleton instance
_graph_visualizer: Optional[GraphVisualizer] = None


def get_graph_visualizer(
    graph: Optional[KnowledgeGraph] = None,
    reset: bool = False
) -> GraphVisualizer:
    """
    Get or create the singleton graph visualizer instance.

    Args:
        graph: KnowledgeGraph instance
        reset: Whether to reset the singleton

    Returns:
        GraphVisualizer instance
    """
    global _graph_visualizer
    if _graph_visualizer is None or reset:
        _graph_visualizer = GraphVisualizer(graph=graph)
    return _graph_visualizer


def reset_graph_visualizer():
    """Reset the singleton graph visualizer (useful for testing)."""
    global _graph_visualizer
    _graph_visualizer = None
