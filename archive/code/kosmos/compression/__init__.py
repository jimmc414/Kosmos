"""
Context Compression Module for Kosmos.

Implements hierarchical compression pipeline achieving 20:1 reduction
(100K+ tokens → 5K tokens) to fit within LLM context windows.

Pattern source: kosmos-claude-skills-mcp (progressive disclosure)
Gap addressed: Gap 0 (Context Compression Architecture)

Key Components:
- ContextCompressor: Main orchestrator for multi-tier compression
- NotebookCompressor: Compresses notebooks → 2-line summary + stats (300:1)
- LiteratureCompressor: Compresses papers → structured summaries (25:1)

Compression Strategy:
1. Task-Level: Each notebook (42K lines) → 2-line summary + statistics
2. Cycle-Level: 10 task summaries → 1 cycle overview
3. Final Synthesis: 20 cycle overviews → Research narrative
4. Lazy Loading: Full content stored on disk, loaded only when needed

Performance Target: 20:1 overall compression ratio
"""

from .compressor import ContextCompressor, NotebookCompressor, LiteratureCompressor

__all__ = [
    "ContextCompressor",
    "NotebookCompressor",
    "LiteratureCompressor",
]
