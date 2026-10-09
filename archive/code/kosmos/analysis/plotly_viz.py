"""
Interactive visualization using Plotly.

Provides interactive versions of publication-quality plots for exploration.
"""

import numpy as np
from typing import Optional, List, Dict, Any
import logging

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly not installed. Install with: pip install plotly")

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


# Color scheme matching PublicationVisualizer
COLORS = {
    'red': '#d7191c',
    'blue_dark': '#2c7bb6',
    'blue': '#0072B2',
    'neutral': '#abd9e9',
    'gray': '#808080',
    'black': '#000000'
}


class PlotlyVisualizer:
    """
    Interactive visualization using Plotly.

    Provides hover tooltips, zoom/pan, and interactive exploration
    of experimental data.

    Example:
        ```python
        viz = PlotlyVisualizer()

        # Interactive scatter plot
        fig = viz.interactive_scatter(
            x=independent_var,
            y=dependent_var,
            x_label="Independent",
            y_label="Dependent",
            title="Interactive Correlation"
        )

        # Save to HTML
        fig.write_html("scatter.html")

        # Or show in browser
        fig.show()
        ```
    """

    def __init__(self):
        """Initialize Plotly visualizer."""
        if not HAS_PLOTLY:
            raise ImportError(
                "plotly is required for interactive visualizations. "
                "Install with: pip install plotly"
            )

        # Default template matching publication style
        self.template = "plotly_white"  # Clean white background

        logger.info("PlotlyVisualizer initialized")

    # ========================================================================
    # INTERACTIVE PLOT TYPES
    # ========================================================================

    def interactive_scatter(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_label: str,
        y_label: str,
        title: str,
        labels: Optional[List[str]] = None,
        show_regression: bool = True
    ) -> go.Figure:
        """
        Interactive scatter plot with optional regression line.

        Args:
            x: X values
            y: Y values
            x_label: X-axis label
            y_label: Y-axis label
            title: Plot title
            labels: Optional labels for hover tooltips
            show_regression: Whether to show regression line

        Returns:
            plotly Figure object
        """
        fig = go.Figure()

        # Hover text
        if labels is not None:
            hover_text = [
                f"{label}<br>{x_label}: {xi:.3f}<br>{y_label}: {yi:.3f}"
                for label, xi, yi in zip(labels, x, y)
            ]
        else:
            hover_text = [
                f"{x_label}: {xi:.3f}<br>{y_label}: {yi:.3f}"
                for xi, yi in zip(x, y)
            ]

        # Scatter plot
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=8,
                color=COLORS['neutral'],
                line=dict(width=1, color='black'),
                opacity=0.7
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            name='Data'
        ))

        # Add regression line if requested
        if show_regression and HAS_SCIPY:
            slope, intercept, r_value, p_value, _ = sp_stats.linregress(x, y)
            x_trend = np.linspace(x.min(), x.max(), 100)
            y_trend = slope * x_trend + intercept

            fig.add_trace(go.Scatter(
                x=x_trend,
                y=y_trend,
                mode='lines',
                line=dict(color=COLORS['red'], dash='dash', width=2),
                name=f'Linear fit (r={r_value:.3f}, p={p_value:.4f})',
                hovertemplate=f'r={r_value:.3f}, p={p_value:.4f}<extra></extra>'
            ))

        # Layout
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template=self.template,
            hovermode='closest',
            font=dict(family="Arial", size=12),
            showlegend=True
        )

        logger.info(f"Created interactive scatter plot: {title}")
        return fig

    def interactive_heatmap(
        self,
        data: np.ndarray,
        row_labels: List[str],
        col_labels: List[str],
        title: str = "Heatmap",
        colorscale: str = 'RdBu_r'
    ) -> go.Figure:
        """
        Interactive heatmap with hover information.

        Args:
            data: 2D array of values
            row_labels: Row labels
            col_labels: Column labels
            title: Plot title
            colorscale: Plotly colorscale name

        Returns:
            plotly Figure object
        """
        # Create hover text
        hover_text = []
        for i, row_label in enumerate(row_labels):
            row_hover = []
            for j, col_label in enumerate(col_labels):
                value = data[i, j]
                row_hover.append(
                    f"{row_label} vs {col_label}<br>Value: {value:.3f}"
                )
            hover_text.append(row_hover)

        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=col_labels,
            y=row_labels,
            colorscale=colorscale,
            hovertext=hover_text,
            hovertemplate='%{hovertext}<extra></extra>',
            colorbar=dict(title="Value")
        ))

        fig.update_layout(
            title=title,
            template=self.template,
            font=dict(family="Arial", size=12),
            xaxis=dict(side='bottom'),
            yaxis=dict(autorange='reversed')  # Top-to-bottom like matrix
        )

        logger.info(f"Created interactive heatmap: {title}")
        return fig

    def interactive_volcano(
        self,
        log2fc: np.ndarray,
        p_values: np.ndarray,
        labels: Optional[List[str]] = None,
        fc_threshold: float = 0.5,
        p_threshold: float = 0.05,
        title: str = "Volcano Plot"
    ) -> go.Figure:
        """
        Interactive volcano plot with click-to-explore.

        Args:
            log2fc: Log2 fold change values
            p_values: P-values
            labels: Optional labels for points
            fc_threshold: Fold change threshold
            p_threshold: P-value threshold
            title: Plot title

        Returns:
            plotly Figure object
        """
        fig = go.Figure()

        # Calculate -log10(p)
        log_p = -np.log10(p_values + 1e-300)

        # Determine significance
        significant = np.array([
            (abs(fc) > fc_threshold and p < p_threshold)
            for fc, p in zip(log2fc, p_values)
        ])

        # Create hover text
        if labels is not None:
            hover_text = [
                f"{label}<br>Log2FC: {fc:.3f}<br>P-value: {p:.4e}<br>"
                f"{'SIGNIFICANT' if sig else 'Not significant'}"
                for label, fc, p, sig in zip(labels, log2fc, p_values, significant)
            ]
        else:
            hover_text = [
                f"Log2FC: {fc:.3f}<br>P-value: {p:.4e}<br>"
                f"{'SIGNIFICANT' if sig else 'Not significant'}"
                for fc, p, sig in zip(log2fc, p_values, significant)
            ]

        # Plot significant points
        sig_indices = np.where(significant)[0]
        if len(sig_indices) > 0:
            fig.add_trace(go.Scatter(
                x=log2fc[sig_indices],
                y=log_p[sig_indices],
                mode='markers',
                marker=dict(
                    size=8,
                    color=COLORS['red'],
                    opacity=0.7
                ),
                text=[hover_text[i] for i in sig_indices],
                hovertemplate='%{text}<extra></extra>',
                name='Significant'
            ))

        # Plot non-significant points
        nonsig_indices = np.where(~significant)[0]
        if len(nonsig_indices) > 0:
            fig.add_trace(go.Scatter(
                x=log2fc[nonsig_indices],
                y=log_p[nonsig_indices],
                mode='markers',
                marker=dict(
                    size=8,
                    color=COLORS['gray'],
                    opacity=0.5
                ),
                text=[hover_text[i] for i in nonsig_indices],
                hovertemplate='%{text}<extra></extra>',
                name='Not significant'
            ))

        # Add threshold lines
        fig.add_hline(
            y=-np.log10(p_threshold),
            line_dash="dash",
            line_color="black",
            annotation_text=f"p={p_threshold}",
            annotation_position="right"
        )

        fig.add_vline(
            x=0,
            line_dash="solid",
            line_color="black",
            opacity=0.3
        )

        # Layout
        fig.update_layout(
            title=title,
            xaxis_title="Log2 Fold Change",
            yaxis_title="-log10(p-value)",
            template=self.template,
            hovermode='closest',
            font=dict(family="Arial", size=12),
            showlegend=True
        )

        logger.info(f"Created interactive volcano plot: {title}")
        return fig

    def interactive_box(
        self,
        data: Dict[str, np.ndarray],
        title: str = "Box Plot",
        y_label: str = "Value"
    ) -> go.Figure:
        """
        Interactive box plot with individual points on hover.

        Args:
            data: Dictionary of {label: values}
            title: Plot title
            y_label: Y-axis label

        Returns:
            plotly Figure object
        """
        fig = go.Figure()

        for label, values in data.items():
            # Create hover text with statistics
            mean_val = np.mean(values)
            median_val = np.median(values)
            std_val = np.std(values)

            hover_text = [
                f"{label}<br>Value: {v:.3f}<br>"
                f"Mean: {mean_val:.3f}<br>Median: {median_val:.3f}<br>SD: {std_val:.3f}"
                for v in values
            ]

            fig.add_trace(go.Box(
                y=values,
                name=label,
                boxpoints='all',  # Show all points
                jitter=0.3,  # Spread points horizontally
                pointpos=-1.8,  # Position points to the left
                marker=dict(
                    size=6,
                    color=COLORS['blue_dark'],
                    opacity=0.6
                ),
                line=dict(color=COLORS['black']),
                fillcolor=COLORS['neutral'],
                hovertext=hover_text,
                hovertemplate='%{hovertext}<extra></extra>'
            ))

        fig.update_layout(
            title=title,
            yaxis_title=y_label,
            template=self.template,
            font=dict(family="Arial", size=12),
            showlegend=True,
            hovermode='closest'
        )

        logger.info(f"Created interactive box plot: {title}")
        return fig

    # ========================================================================
    # MULTI-PANEL FIGURES
    # ========================================================================

    def create_multi_panel(
        self,
        figures: List[go.Figure],
        rows: int,
        cols: int,
        subplot_titles: Optional[List[str]] = None,
        main_title: Optional[str] = None
    ) -> go.Figure:
        """
        Create multi-panel figure with subplots.

        Args:
            figures: List of plotly Figures
            rows: Number of rows
            cols: Number of columns
            subplot_titles: Titles for each subplot
            main_title: Main title for entire figure

        Returns:
            plotly Figure with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles
        )

        # Add traces from each figure
        for idx, source_fig in enumerate(figures):
            row = idx // cols + 1
            col = idx % cols + 1

            for trace in source_fig.data:
                fig.add_trace(trace, row=row, col=col)

        # Update layout
        if main_title:
            fig.update_layout(title_text=main_title)

        fig.update_layout(
            template=self.template,
            font=dict(family="Arial", size=12),
            showlegend=True
        )

        logger.info(f"Created multi-panel figure with {len(figures)} panels")
        return fig

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def save_html(
        self,
        fig: go.Figure,
        output_path: str,
        include_plotlyjs: str = 'cdn'
    ):
        """
        Save Plotly figure to HTML file.

        Args:
            fig: Plotly Figure object
            output_path: Path to save HTML file
            include_plotlyjs: How to include plotly.js ('cdn', 'directory', True, False)
        """
        fig.write_html(output_path, include_plotlyjs=include_plotlyjs)
        logger.info(f"Saved interactive plot to {output_path}")

    def save_static_image(
        self,
        fig: go.Figure,
        output_path: str,
        width: int = 800,
        height: int = 600,
        scale: float = 2.0
    ):
        """
        Save Plotly figure as static image (requires kaleido).

        Args:
            fig: Plotly Figure object
            output_path: Path to save image (PNG, PDF, SVG, etc.)
            width: Image width in pixels
            height: Image height in pixels
            scale: Scale factor for resolution
        """
        try:
            fig.write_image(output_path, width=width, height=height, scale=scale)
            logger.info(f"Saved static image to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save static image: {e}")
            logger.info("Install kaleido for static image export: pip install kaleido")
            raise
