"""
Neuroscience Domain - Neurodegeneration Analysis

Analyzes neurodegenerative diseases (Alzheimer's, Parkinson's, aging):
- Differential gene expression analysis
- Temporal trajectory modeling
- Cross-species validation
- Pathway enrichment

Based on kosmos-figures Figures 7, 8 patterns.

Example usage:
    # Differential expression for AD
    analyzer = NeurodegenerationAnalyzer()

    # Run DESeq2 analysis
    results = analyzer.differential_expression_analysis(
        counts_matrix=rnaseq_counts,
        sample_metadata=metadata,
        condition_column='disease_status',
        case_label='AD',
        control_label='Control'
    )

    # Temporal ordering
    temporal_genes = analyzer.temporal_ordering(results)
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from enum import Enum


# Data models for neurodegeneration analysis

class TemporalStage(str, Enum):
    """Temporal stages of disease progression"""
    EARLY_DOWN = "early_downregulated"
    MILD_DOWN = "mildly_downregulated"
    STABLE = "stable"
    MILD_UP = "mildly_upregulated"
    LATE_UP = "late_upregulated"


@dataclass
class DifferentialExpressionResult:
    """Single gene differential expression result"""
    gene_id: str
    gene_name: Optional[str] = None
    log2_fold_change: float = 0.0
    p_value: float = 1.0
    adjusted_p_value: float = 1.0
    base_mean: float = 0.0

    # Significance flags
    significant_005: bool = False  # padj < 0.05
    significant_001: bool = False  # padj < 0.01

    # Direction
    direction: str = "unchanged"  # "up", "down", "unchanged"

    # Temporal stage (for temporal analysis)
    temporal_stage: Optional[TemporalStage] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived fields"""
        self.significant_005 = self.adjusted_p_value < 0.05
        self.significant_001 = self.adjusted_p_value < 0.01

        if self.significant_005:
            if self.log2_fold_change > 0:
                self.direction = "up"
            elif self.log2_fold_change < 0:
                self.direction = "down"


@dataclass
class NeurodegenerationResult:
    """Complete differential expression analysis results"""
    analysis_name: str
    case_label: str  # e.g., "AD"
    control_label: str  # e.g., "Control"

    # Results
    gene_results: List[DifferentialExpressionResult]

    # Summary statistics
    n_genes_tested: int = 0
    n_upregulated: int = 0  # padj < 0.05, log2FC > 0
    n_downregulated: int = 0  # padj < 0.05, log2FC < 0
    n_significant: int = 0  # padj < 0.05

    # Metadata
    sample_sizes: Dict[str, int] = field(default_factory=dict)
    analysis_notes: List[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame"""
        rows = []
        for result in self.gene_results:
            rows.append({
                'gene_id': result.gene_id,
                'gene_name': result.gene_name,
                'log2FoldChange': result.log2_fold_change,
                'pvalue': result.p_value,
                'padj': result.adjusted_p_value,
                'baseMean': result.base_mean,
                'significant_0.05': result.significant_005,
                'significant_0.01': result.significant_001,
                'direction': result.direction,
                'temporal_stage': result.temporal_stage.value if result.temporal_stage else None
            })
        return pd.DataFrame(rows)

    def get_significant_genes(
        self,
        p_threshold: float = 0.05,
        direction: Optional[str] = None
    ) -> List[DifferentialExpressionResult]:
        """Get significant genes, optionally filtered by direction"""
        significant = [
            r for r in self.gene_results
            if r.adjusted_p_value < p_threshold
        ]

        if direction:
            significant = [r for r in significant if r.direction == direction]

        return significant

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            'analysis': self.analysis_name,
            'comparison': f"{self.case_label} vs {self.control_label}",
            'n_genes_tested': self.n_genes_tested,
            'n_significant': self.n_significant,
            'n_upregulated': self.n_upregulated,
            'n_downregulated': self.n_downregulated,
            'percent_significant': f"{self.n_significant/self.n_genes_tested*100:.2f}%" if self.n_genes_tested > 0 else "N/A",
            'sample_sizes': self.sample_sizes
        }


@dataclass
class PathwayEnrichmentResult:
    """Pathway enrichment analysis result"""
    pathway_id: str
    pathway_name: str
    n_genes_in_pathway: int
    n_genes_significant: int
    enrichment_pvalue: float
    odds_ratio: float
    significant_genes: List[str] = field(default_factory=list)
    is_enriched: bool = False

    def __post_init__(self):
        """Calculate enrichment"""
        self.is_enriched = self.enrichment_pvalue < 0.05


@dataclass
class CrossSpeciesValidation:
    """Cross-species concordance result"""
    gene_id: str
    gene_name: Optional[str] = None

    # Mouse results
    mouse_log2fc: Optional[float] = None
    mouse_padj: Optional[float] = None
    mouse_significant: bool = False

    # Human results
    human_log2fc: Optional[float] = None
    human_padj: Optional[float] = None
    human_significant: bool = False

    # Concordance
    is_concordant: bool = False  # Same direction in both species
    concordance_strength: str = "none"  # "strong", "moderate", "weak", "none"

    def __post_init__(self):
        """Calculate concordance"""
        if (self.mouse_log2fc is not None and self.human_log2fc is not None and
            self.mouse_significant and self.human_significant):

            # Check if same direction
            same_direction = (
                (self.mouse_log2fc > 0 and self.human_log2fc > 0) or
                (self.mouse_log2fc < 0 and self.human_log2fc < 0)
            )

            if same_direction:
                self.is_concordant = True

                # Strength based on both being significant
                if self.mouse_padj < 0.01 and self.human_padj < 0.01:
                    self.concordance_strength = "strong"
                elif self.mouse_padj < 0.05 and self.human_padj < 0.05:
                    self.concordance_strength = "moderate"
                else:
                    self.concordance_strength = "weak"


# Analyzer class

class NeurodegenerationAnalyzer:
    """
    Analyzer for neurodegenerative disease data.

    Implements Figures 7, 8 patterns from kosmos-figures:
    - Differential expression analysis (DESeq2-like)
    - Temporal trajectory modeling
    - Pathway enrichment analysis
    - Cross-species validation
    - Volcano plot data generation

    Example:
        analyzer = NeurodegenerationAnalyzer()

        # Differential expression
        results = analyzer.differential_expression_analysis(
            counts_matrix=rnaseq_df,
            sample_metadata=metadata_df,
            condition_column='disease_status',
            case_label='AD',
            control_label='Control'
        )

        # Temporal ordering
        temporal = analyzer.temporal_ordering(results)

        # Pathway enrichment
        ecm_genes = ['COL1A1', 'COL1A2', 'FN1', 'LAMA1', ...]
        enrichment = analyzer.pathway_enrichment(
            deg_results=results,
            pathway_genes=ecm_genes,
            pathway_name='Extracellular Matrix'
        )
    """

    def __init__(self):
        """Initialize NeurodegenerationAnalyzer"""
        pass

    def differential_expression_analysis(
        self,
        counts_matrix: pd.DataFrame,
        sample_metadata: pd.DataFrame,
        condition_column: str,
        case_label: str,
        control_label: str,
        min_count: int = 10,
        use_pydeseq2: bool = True
    ) -> NeurodegenerationResult:
        """
        Perform differential expression analysis.

        Implements Figure 7 pattern using DESeq2-like methodology.

        Args:
            counts_matrix: Gene expression counts (genes x samples)
            sample_metadata: Sample information with condition labels
            condition_column: Column in metadata with case/control labels
            case_label: Label for case samples (e.g., "AD")
            control_label: Label for control samples (e.g., "Control")
            min_count: Minimum total count across samples to include gene
            use_pydeseq2: Use pydeseq2 if available, otherwise simple t-test

        Returns:
            NeurodegenerationResult with all genes

        Example:
            # Counts matrix: rows = genes, columns = samples
            counts = pd.DataFrame({
                'Sample1': [100, 200, 50],
                'Sample2': [120, 180, 55],
                'Sample3': [80, 220, 45]
            }, index=['GENE1', 'GENE2', 'GENE3'])

            # Metadata
            metadata = pd.DataFrame({
                'sample_id': ['Sample1', 'Sample2', 'Sample3'],
                'disease_status': ['Control', 'Control', 'AD']
            })

            results = analyzer.differential_expression_analysis(
                counts, metadata, 'disease_status', 'AD', 'Control'
            )
        """
        # Validate inputs
        if condition_column not in sample_metadata.columns:
            raise ValueError(f"Condition column '{condition_column}' not in metadata")

        # Filter samples
        case_samples = sample_metadata[sample_metadata[condition_column] == case_label]['sample_id'].tolist()
        control_samples = sample_metadata[sample_metadata[condition_column] == control_label]['sample_id'].tolist()

        if not case_samples or not control_samples:
            raise ValueError(f"No samples found for {case_label} or {control_label}")

        # Filter low-count genes
        total_counts = counts_matrix.sum(axis=1)
        genes_to_keep = total_counts >= min_count
        counts_filtered = counts_matrix[genes_to_keep]

        n_genes = len(counts_filtered)

        # Attempt pydeseq2 analysis
        if use_pydeseq2:
            try:
                results_df = self._run_pydeseq2(
                    counts_filtered, sample_metadata, condition_column,
                    case_label, control_label
                )
            except Exception as e:
                print(f"Warning: pydeseq2 failed ({e}), falling back to simple analysis")
                results_df = self._run_simple_differential_expression(
                    counts_filtered, case_samples, control_samples
                )
        else:
            results_df = self._run_simple_differential_expression(
                counts_filtered, case_samples, control_samples
            )

        # Convert to result objects
        gene_results = []
        n_up = 0
        n_down = 0
        n_sig = 0

        for gene_id, row in results_df.iterrows():
            result = DifferentialExpressionResult(
                gene_id=gene_id,
                log2_fold_change=row['log2FoldChange'],
                p_value=row['pvalue'],
                adjusted_p_value=row['padj'],
                base_mean=row['baseMean']
            )

            if result.significant_005:
                n_sig += 1
                if result.direction == "up":
                    n_up += 1
                elif result.direction == "down":
                    n_down += 1

            gene_results.append(result)

        # Create result
        neuro_result = NeurodegenerationResult(
            analysis_name=f"{case_label}_vs_{control_label}",
            case_label=case_label,
            control_label=control_label,
            gene_results=gene_results,
            n_genes_tested=n_genes,
            n_upregulated=n_up,
            n_downregulated=n_down,
            n_significant=n_sig,
            sample_sizes={
                case_label: len(case_samples),
                control_label: len(control_samples)
            }
        )

        neuro_result.analysis_notes.append(
            f"Tested {n_genes} genes (filtered from {len(counts_matrix)} total)"
        )

        return neuro_result

    def _run_pydeseq2(
        self,
        counts: pd.DataFrame,
        metadata: pd.DataFrame,
        condition_col: str,
        case_label: str,
        control_label: str
    ) -> pd.DataFrame:
        """
        Run pydeseq2 differential expression analysis.

        Args:
            counts: Filtered counts matrix
            metadata: Sample metadata
            condition_col: Condition column name
            case_label: Case label
            control_label: Control label

        Returns:
            DataFrame with DESeq2 results
        """
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats

        # Prepare metadata (ensure sample order matches counts)
        metadata_ordered = metadata.set_index('sample_id').loc[counts.columns]

        # Create DESeq2 dataset
        dds = DeseqDataSet(
            counts=counts,
            metadata=metadata_ordered,
            design_factors=condition_col,
            refit_cooks=True
        )

        # Run DESeq2
        dds.deseq2()

        # Get statistical results
        stat_res = DeseqStats(dds, contrast=[condition_col, case_label, control_label])
        stat_res.summary()

        # Extract results
        results_df = stat_res.results_df

        # Rename columns to match standard format
        results_df = results_df.rename(columns={
            'log2FoldChange': 'log2FoldChange',
            'pvalue': 'pvalue',
            'padj': 'padj',
            'baseMean': 'baseMean'
        })

        return results_df

    def _run_simple_differential_expression(
        self,
        counts: pd.DataFrame,
        case_samples: List[str],
        control_samples: List[str]
    ) -> pd.DataFrame:
        """
        Simple differential expression using log2 fold change and t-test.

        Fallback when pydeseq2 is not available.

        Args:
            counts: Counts matrix
            case_samples: Case sample IDs
            control_samples: Control sample IDs

        Returns:
            DataFrame with results
        """
        from scipy.stats import ttest_ind

        results = []

        for gene_id in counts.index:
            case_counts = counts.loc[gene_id, case_samples].values
            control_counts = counts.loc[gene_id, control_samples].values

            # Add pseudocount and log transform
            case_log = np.log2(case_counts + 1)
            control_log = np.log2(control_counts + 1)

            # Calculate statistics
            mean_case = np.mean(case_log)
            mean_control = np.mean(control_log)
            log2fc = mean_case - mean_control

            # T-test
            t_stat, p_val = ttest_ind(case_log, control_log)

            # Base mean (untransformed)
            base_mean = np.mean(np.concatenate([case_counts, control_counts]))

            results.append({
                'gene_id': gene_id,
                'log2FoldChange': log2fc,
                'pvalue': p_val,
                'baseMean': base_mean
            })

        results_df = pd.DataFrame(results).set_index('gene_id')

        # Benjamini-Hochberg FDR correction
        from statsmodels.stats.multitest import fdrcorrection
        _, results_df['padj'] = fdrcorrection(results_df['pvalue'].values)

        return results_df

    def temporal_ordering(
        self,
        deg_results: NeurodegenerationResult,
        n_stages: int = 5
    ) -> NeurodegenerationResult:
        """
        Order genes by temporal trajectory in disease progression.

        Implements Figure 7 pattern: assign genes to temporal stages based on
        effect size (log2 fold change).

        Args:
            deg_results: Differential expression results
            n_stages: Number of temporal stages (default: 5)

        Returns:
            Updated NeurodegenerationResult with temporal stages assigned
        """
        # Get significant genes
        significant = deg_results.get_significant_genes()

        if not significant:
            return deg_results

        # Extract log2 fold changes
        log2fcs = [r.log2_fold_change for r in significant]

        # Create bins for temporal stages
        if n_stages == 5:
            # Default: early down, mild down, stable, mild up, late up
            stages = [
                TemporalStage.EARLY_DOWN,
                TemporalStage.MILD_DOWN,
                TemporalStage.STABLE,
                TemporalStage.MILD_UP,
                TemporalStage.LATE_UP
            ]
        else:
            raise NotImplementedError(f"Only 5 stages supported, got {n_stages}")

        # Assign stages based on log2FC quantiles
        log2fc_array = np.array(log2fcs)
        percentiles = [0, 20, 40, 60, 80, 100]
        bins = np.percentile(log2fc_array, percentiles)

        for result in deg_results.gene_results:
            if result.significant_005:
                # Find bin
                for i in range(len(bins) - 1):
                    if bins[i] <= result.log2_fold_change < bins[i + 1]:
                        result.temporal_stage = stages[i]
                        break
                    elif i == len(bins) - 2:  # Last bin
                        result.temporal_stage = stages[i]

        deg_results.analysis_notes.append(
            f"Assigned temporal stages to {len(significant)} significant genes"
        )

        return deg_results

    def pathway_enrichment(
        self,
        deg_results: NeurodegenerationResult,
        pathway_genes: List[str],
        pathway_name: str = "Pathway",
        pathway_id: str = "custom",
        universe_size: Optional[int] = None
    ) -> PathwayEnrichmentResult:
        """
        Test pathway enrichment in differentially expressed genes.

        Uses Fisher's exact test to assess enrichment.

        Args:
            deg_results: Differential expression results
            pathway_genes: Genes in the pathway of interest
            pathway_name: Pathway name
            pathway_id: Pathway ID
            universe_size: Total genes tested (default: use from deg_results)

        Returns:
            PathwayEnrichmentResult

        Example:
            ecm_genes = ['COL1A1', 'COL1A2', 'FN1', 'LAMA1', 'LAMB1']
            enrichment = analyzer.pathway_enrichment(
                results, ecm_genes, "Extracellular Matrix", "GO:0031012"
            )
        """
        # Get significant genes
        sig_genes = deg_results.get_significant_genes()
        sig_gene_ids = set([r.gene_id for r in sig_genes])

        # All tested genes
        all_gene_ids = set([r.gene_id for r in deg_results.gene_results])

        # Universe size
        if universe_size is None:
            universe_size = len(all_gene_ids)

        # Pathway genes in universe
        pathway_genes_set = set(pathway_genes)
        pathway_in_universe = pathway_genes_set & all_gene_ids

        # Overlap: significant genes in pathway
        overlap = sig_gene_ids & pathway_in_universe

        # Fisher's exact test contingency table:
        #                In Pathway  | Not in Pathway
        # Significant    a           | b
        # Not Sig        c           | d

        a = len(overlap)  # Sig & in pathway
        b = len(sig_gene_ids) - a  # Sig & not in pathway
        c = len(pathway_in_universe) - a  # Not sig & in pathway
        d = universe_size - a - b - c  # Not sig & not in pathway

        # Fisher's exact test
        odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')

        result = PathwayEnrichmentResult(
            pathway_id=pathway_id,
            pathway_name=pathway_name,
            n_genes_in_pathway=len(pathway_in_universe),
            n_genes_significant=a,
            enrichment_pvalue=p_value,
            odds_ratio=odds_ratio,
            significant_genes=list(overlap)
        )

        return result

    def cross_species_validation(
        self,
        mouse_results: NeurodegenerationResult,
        human_results: NeurodegenerationResult,
        gene_mapping: Optional[Dict[str, str]] = None
    ) -> List[CrossSpeciesValidation]:
        """
        Validate findings across species (Figure 8 pattern).

        Args:
            mouse_results: Mouse differential expression results
            human_results: Human differential expression results
            gene_mapping: Dict mapping mouse gene IDs to human gene IDs
                         (if None, assumes same gene IDs)

        Returns:
            List of CrossSpeciesValidation results

        Example:
            mouse_res = analyzer.differential_expression_analysis(...)
            human_res = analyzer.differential_expression_analysis(...)
            validation = analyzer.cross_species_validation(mouse_res, human_res)

            concordant = [v for v in validation if v.is_concordant]
            print(f"Found {len(concordant)} concordant genes")
        """
        # Create lookup dicts
        mouse_dict = {r.gene_id: r for r in mouse_results.gene_results}
        human_dict = {r.gene_id: r for r in human_results.gene_results}

        # Find common genes
        if gene_mapping:
            common_genes = set(gene_mapping.keys()) & set(mouse_dict.keys())
        else:
            common_genes = set(mouse_dict.keys()) & set(human_dict.keys())

        # Compare results
        validation_results = []

        for mouse_gene_id in common_genes:
            human_gene_id = gene_mapping.get(mouse_gene_id, mouse_gene_id) if gene_mapping else mouse_gene_id

            if mouse_gene_id in mouse_dict and human_gene_id in human_dict:
                mouse_res = mouse_dict[mouse_gene_id]
                human_res = human_dict[human_gene_id]

                validation = CrossSpeciesValidation(
                    gene_id=human_gene_id,
                    mouse_log2fc=mouse_res.log2_fold_change,
                    mouse_padj=mouse_res.adjusted_p_value,
                    mouse_significant=mouse_res.significant_005,
                    human_log2fc=human_res.log2_fold_change,
                    human_padj=human_res.adjusted_p_value,
                    human_significant=human_res.significant_005
                )

                validation_results.append(validation)

        return validation_results

    def generate_volcano_plot_data(
        self,
        deg_results: NeurodegenerationResult,
        p_threshold: float = 0.05,
        fc_threshold: float = 1.0
    ) -> pd.DataFrame:
        """
        Generate data for volcano plot visualization.

        Args:
            deg_results: Differential expression results
            p_threshold: P-value threshold for significance line
            fc_threshold: Log2 fold change threshold for significance

        Returns:
            DataFrame with x=log2FC, y=-log10(padj), color=significance

        Example:
            plot_data = analyzer.generate_volcano_plot_data(results)

            # Then use with matplotlib/plotly:
            plt.scatter(plot_data['log2FC'], plot_data['-log10_padj'],
                       c=plot_data['color'], alpha=0.5)
        """
        rows = []

        for result in deg_results.gene_results:
            # -log10(p-value)
            neg_log10_p = -np.log10(result.adjusted_p_value) if result.adjusted_p_value > 0 else 100

            # Determine significance
            is_sig = (
                result.adjusted_p_value < p_threshold and
                abs(result.log2_fold_change) > fc_threshold
            )

            if is_sig:
                if result.log2_fold_change > 0:
                    color = "upregulated"
                else:
                    color = "downregulated"
            else:
                color = "not_significant"

            rows.append({
                'gene_id': result.gene_id,
                'log2FoldChange': result.log2_fold_change,
                '-log10_padj': neg_log10_p,
                'padj': result.adjusted_p_value,
                'significance': color,
                'is_significant': is_sig
            })

        return pd.DataFrame(rows)
