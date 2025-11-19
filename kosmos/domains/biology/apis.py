"""
Biology domain API clients.

Implements clients for biology-related databases and APIs:
- KEGG: Pathway mapping and metabolite classification
- GWAS Catalog: GWAS summary statistics
- GTEx: eQTL/pQTL data
- ENCODE: ATAC-seq, ChIP-seq data
- dbSNP: SNP annotations
- Ensembl: Variant effect predictions
- HMDB: Human metabolite database
- MetaboLights: Public metabolomics data
- UniProt: Protein information
- PDB: Protein structures
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

logger = logging.getLogger(__name__)


@dataclass
class KEGGPathway:
    """KEGG pathway information."""
    pathway_id: str
    name: str
    category: str
    compounds: List[str]
    genes: List[str]


@dataclass
class GWASVariant:
    """GWAS variant information."""
    snp_id: str
    chr: str
    position: int
    p_value: float
    beta: float
    trait: str
    sample_size: int


@dataclass
class eQTLData:
    """eQTL data from GTEx."""
    snp_id: str
    gene_id: str
    tissue: str
    beta: float
    p_value: float
    effect_size: float


class KEGGClient:
    """
    Client for KEGG (Kyoto Encyclopedia of Genes and Genomes) API.

    Provides access to pathway information and metabolite classification.
    """

    BASE_URL = "https://rest.kegg.jp"

    def __init__(self, timeout: int = 30):
        """Initialize KEGG client."""
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_compound(self, compound_id: str) -> Optional[Dict[str, Any]]:
        """
        Get compound information from KEGG.

        Args:
            compound_id: KEGG compound ID (e.g., 'C00002' for ATP)

        Returns:
            Compound information dict or None if not found
        """
        try:
            url = f"{self.BASE_URL}/get/{compound_id}"
            response = self.client.get(url)
            response.raise_for_status()

            # Parse KEGG text format
            data = self._parse_kegg_entry(response.text)
            return data

        except (httpx.HTTPError, RetryError, Exception) as e:
            logger.error(f"KEGG API error for {compound_id}: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_pathway(self, pathway_id: str) -> Optional[KEGGPathway]:
        """
        Get pathway information.

        Args:
            pathway_id: KEGG pathway ID (e.g., 'hsa00230' for purine metabolism)

        Returns:
            KEGGPathway object or None
        """
        try:
            url = f"{self.BASE_URL}/get/{pathway_id}"
            response = self.client.get(url)
            response.raise_for_status()

            data = self._parse_kegg_entry(response.text)

            return KEGGPathway(
                pathway_id=pathway_id,
                name=data.get('NAME', ['Unknown'])[0],
                category=data.get('CLASS', ['Unknown'])[0],
                compounds=data.get('COMPOUND', []),
                genes=data.get('GENE', []),
            )

        except Exception as e:
            logger.error(f"KEGG pathway error for {pathway_id}: {e}")
            return None

    def categorize_metabolite(self, compound_name: str) -> Dict[str, Any]:
        """
        Categorize metabolite by biochemical pathway.

        Args:
            compound_name: Name or ID of compound

        Returns:
            Dict with category, pathways, and metabolite type
        """
        # Search for compound
        try:
            url = f"{self.BASE_URL}/find/compound/{compound_name}"
            response = self.client.get(url)

            if response.status_code != 200:
                return {'category': 'unknown', 'pathways': []}

            # Parse search results
            lines = response.text.strip().split('\n')
            if not lines:
                return {'category': 'unknown', 'pathways': []}

            # Get first match
            first_match = lines[0].split('\t')[0]
            compound_id = first_match.split(':')[1] if ':' in first_match else first_match

            # Get compound details
            compound_data = self.get_compound(compound_id)

            if not compound_data:
                return {'category': 'unknown', 'pathways': []}

            # Extract pathways
            pathways = compound_data.get('PATHWAY', [])

            # Classify as purine/pyrimidine/other based on pathways
            category = 'other'
            metabolite_type = 'unknown'

            pathway_str = ' '.join(pathways).lower()

            if 'purine' in pathway_str:
                category = 'purine'
            elif 'pyrimidine' in pathway_str:
                category = 'pyrimidine'

            # Determine salvage vs synthesis based on compound name
            name_lower = compound_name.lower()
            if name_lower.endswith('osine') or name_lower in ['adenine', 'guanine', 'cytosine', 'uracil']:
                metabolite_type = 'salvage_precursor'
            elif 'monophosphate' in name_lower or 'diphosphate' in name_lower or 'triphosphate' in name_lower:
                metabolite_type = 'synthesis_product'
            else:
                metabolite_type = 'intermediate'

            return {
                'category': category,
                'metabolite_type': metabolite_type,
                'pathways': pathways,
                'compound_id': compound_id,
            }

        except Exception as e:
            logger.error(f"Metabolite categorization error: {e}")
            return {'category': 'unknown', 'pathways': [], 'error': str(e)}

    def _parse_kegg_entry(self, text: str) -> Dict[str, List[str]]:
        """Parse KEGG text format into structured dict."""
        data = {}
        current_key = None

        for line in text.split('\n'):
            if not line.strip():
                continue

            if line[0] != ' ':  # New section
                parts = line.split(maxsplit=1)
                current_key = parts[0]
                if len(parts) > 1:
                    data[current_key] = [parts[1].strip()]
                else:
                    data[current_key] = []
            elif current_key:  # Continuation
                data[current_key].append(line.strip())

        return data

    def get_atac_peaks(self, snp_id: str) -> Optional[Dict[str, Any]]:
        """
        Check for ATAC-seq peaks for a given SNP.

        This is a simplified mock implementation. A real implementation would query
        a database of ATAC-seq peaks (e.g., from ENCODE or a custom resource)
        to check for overlap with the SNP's genomic location.

        Args:
            snp_id: SNP ID

        Returns:
            Dict with 'has_peak' and 'significance' or None
        """
        # Mock implementation: for demonstration, assume some SNPs are in ATAC peaks
        # In a real scenario, this would involve a database query.
        mock_significant_snps = ['rs7903146', 'rs12255372']
        if snp_id in mock_significant_snps:
            return {'has_peak': True, 'significance': 1.5e-8}
        else:
            return {'has_peak': False, 'significance': None}

    def close(self):
        """Close HTTP client."""
        self.client.close()


class GWASCatalogClient:
    """Client for NHGRI-EBI GWAS Catalog."""

    BASE_URL = "https://www.ebi.ac.uk/gwas/rest/api"

    def __init__(self, timeout: int = 30):
        """Initialize GWAS Catalog client."""
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout, follow_redirects=True)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_variant(self, snp_id: str) -> Optional[GWASVariant]:
        """
        Get GWAS data for a variant.

        Args:
            snp_id: SNP ID (e.g., 'rs7903146')

        Returns:
            GWASVariant object or None
        """
        try:
            url = f"{self.BASE_URL}/singleNucleotidePolymorphisms/{snp_id}"
            response = self.client.get(
                url,
                headers={'Accept': 'application/json'}
            )
            response.raise_for_status()

            data = response.json()

            # Extract relevant fields
            if not data or '_embedded' not in data:
                return None

            snp_data = data['_embedded']['singleNucleotidePolymorphisms'][0]

            # Get associated studies
            associations_url = f"{url}/associations"
            assoc_response = self.client.get(
                associations_url,
                headers={'Accept': 'application/json'}
            )

            if assoc_response.status_code == 200:
                assoc_data = assoc_response.json()
                if '_embedded' in assoc_data and 'associations' in assoc_data['_embedded']:
                    associations = assoc_data['_embedded']['associations']
                    if associations:
                        first_assoc = associations[0]

                        return GWASVariant(
                            snp_id=snp_id,
                            chr=snp_data.get('chromosomeName', ''),
                            position=int(snp_data.get('chromosomePosition', 0)),
                            p_value=float(first_assoc.get('pvalue', 1.0)),
                            beta=float(first_assoc.get('betaNum', 0.0)),
                            trait=first_assoc.get('traitName', ''),
                            sample_size=int(first_assoc.get('sampleSize', 0)),
                        )

            return None

        except Exception as e:
            logger.error(f"GWAS Catalog error for {snp_id}: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def search_by_gene(self, gene_name: str) -> Optional[Dict[str, Any]]:
        """
        Search for GWAS variants associated with a gene.

        Args:
            gene_name: Gene symbol (e.g., 'TCF7L2')

        Returns:
            Dictionary with associations or None
        """
        try:
            url = f"{self.BASE_URL}/efoTraits/search/findByEfoTrait"
            # Alternative: search by gene name in associations
            url = f"{self.BASE_URL}/singleNucleotidePolymorphisms/search/findByGene"
            params = {'geneName': gene_name}

            response = self.client.get(
                url,
                params=params,
                headers={'Accept': 'application/json'}
            )

            if response.status_code == 200:
                return response.json()

            return None

        except Exception as e:
            logger.error(f"GWAS gene search error for {gene_name}: {e}")
            return None

    def close(self):
        """Close HTTP client."""
        self.client.close()


class GTExClient:
    """Client for GTEx Portal API (eQTL/pQTL data)."""

    BASE_URL = "https://gtexportal.org/api/v2"

    def __init__(self, timeout: int = 30):
        """Initialize GTEx client."""
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_eqtl(self, snp_id: str, gene_id: str, tissue: str = "Whole_Blood") -> Optional[eQTLData]:
        """
        Get eQTL data for SNP-gene pair.

        Args:
            snp_id: Variant ID
            gene_id: Gene ID (Ensembl format)
            tissue: GTEx tissue name

        Returns:
            eQTLData object or None
        """
        try:
            url = f"{self.BASE_URL}/association/dyneqtl"
            params = {
                'variantId': snp_id,
                'gencodeId': gene_id,
                'tissueSiteDetailId': tissue,
            }

            response = self.client.get(url, params=params)

            if response.status_code == 200:
                data = response.json()

                if data and 'data' in data and len(data['data']) > 0:
                    eqtl = data['data'][0]

                    return eQTLData(
                        snp_id=snp_id,
                        gene_id=gene_id,
                        tissue=tissue,
                        beta=float(eqtl.get('slope', 0.0)),
                        p_value=float(eqtl.get('pValue', 1.0)),
                        effect_size=float(eqtl.get('effectSize', 0.0)),
                    )

            return None

        except Exception as e:
            logger.error(f"GTEx eQTL error: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_pqtl(self, snp_id: str, gene_id: str, tissue: str = "Whole_Blood") -> Optional[eQTLData]:
        """
        Get pQTL data for SNP-gene pair.

        Args:
            snp_id: Variant ID
            gene_id: Gene ID (Ensembl format)
            tissue: GTEx tissue name

        Returns:
            eQTLData object or None (reusing eQTLData as the structure is similar)
        """
        try:
            url = f"{self.BASE_URL}/association/dynpqtl"
            params = {
                'variantId': snp_id,
                'gencodeId': gene_id,
                'tissueSiteDetailId': tissue,
            }

            response = self.client.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                if data and 'data' in data and len(data['data']) > 0:
                    pqtl = data['data'][0]
                    return eQTLData(
                        snp_id=snp_id,
                        gene_id=gene_id,
                        tissue=tissue,
                        beta=float(pqtl.get('slope', 0.0)),
                        p_value=float(pqtl.get('pValue', 1.0)),
                        effect_size=float(pqtl.get('effectSize', 0.0)),
                    )
            return None
        except Exception as e:
            logger.error(f"GTEx pQTL error: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_gene_expression(self, gene_id: str, tissue: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get gene expression data across tissues.

        Args:
            gene_id: Gene symbol or Ensembl ID (e.g., 'TCF7L2', 'ENSG00000148737')
            tissue: Optional tissue filter

        Returns:
            Dictionary with gene expression data or None
        """
        try:
            url = f"{self.BASE_URL}/expression/geneExpression"
            params = {'gencodeId': gene_id}
            if tissue:
                params['tissueSiteDetailId'] = tissue

            response = self.client.get(url, params=params)

            if response.status_code == 200:
                return response.json()

            return None

        except Exception as e:
            logger.error(f"GTEx gene expression error for {gene_id}: {e}")
            return None

    def close(self):
        """Close HTTP client."""
        self.client.close()


class ENCODEClient:
    """Client for ENCODE database (ATAC-seq, ChIP-seq)."""

    BASE_URL = "https://www.encodeproject.org"

    def __init__(self, timeout: int = 30):
        """Initialize ENCODE client."""
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def search_experiments(
        self,
        assay_type: str,
        biosample: str,
        limit: int = 10
    ) -> Optional[Dict[str, Any]]:
        """
        Search for experiments in ENCODE.

        Args:
            assay_type: Type of assay (e.g., 'ATAC-seq', 'ChIP-seq')
            biosample: Biosample name
            limit: Max results

        Returns:
            Search results dict or None
        """
        try:
            url = f"{self.BASE_URL}/search/"
            params = {
                'type': 'Experiment',
                'assay_title': assay_type,
                'biosample_ontology.term_name': biosample,
                'limit': limit,
                'format': 'json',
            }

            response = self.client.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            return data

        except Exception as e:
            logger.error(f"ENCODE search error: {e}")
            return None

    def close(self):
        """Close HTTP client."""
        self.client.close()


class dbSNPClient:
    """Client for dbSNP (NCBI SNP database)."""

    BASE_URL = "https://api.ncbi.nlm.nih.gov/variation/v0"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        """
        Initialize dbSNP client.

        Args:
            api_key: NCBI API key (optional, increases rate limits)
            timeout: Request timeout
        """
        self.api_key = api_key
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_snp(self, snp_id: str) -> Optional[Dict[str, Any]]:
        """
        Get SNP information.

        Args:
            snp_id: SNP ID (e.g., 'rs7903146')

        Returns:
            SNP information dict or None
        """
        try:
            # Remove 'rs' prefix if present
            rs_number = snp_id.replace('rs', '')

            url = f"{self.BASE_URL}/beta/refsnp/{rs_number}"
            headers = {}
            if self.api_key:
                headers['api_key'] = self.api_key

            response = self.client.get(url, headers=headers)

            if response.status_code == 200:
                return response.json()

            return None

        except Exception as e:
            logger.error(f"dbSNP error for {snp_id}: {e}")
            return None

    def close(self):
        """Close HTTP client."""
        self.client.close()


class EnsemblClient:
    """Client for Ensembl REST API."""

    BASE_URL = "https://rest.ensembl.org"

    def __init__(self, timeout: int = 30):
        """Initialize Ensembl client."""
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_variant_consequences(
        self,
        variant_id: str,
        species: str = "human"
    ) -> Optional[Dict[str, Any]]:
        """
        Get variant effect predictions.

        Args:
            variant_id: Variant ID (e.g., 'rs699')
            species: Species name

        Returns:
            Variant consequences dict or None
        """
        try:
            url = f"{self.BASE_URL}/vep/{species}/id/{variant_id}"
            headers = {'Content-Type': 'application/json'}

            response = self.client.get(url, headers=headers)

            if response.status_code == 200:
                return response.json()

            return None

        except Exception as e:
            logger.error(f"Ensembl VEP error for {variant_id}: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_vep_annotation(self, variant: str, species: str = "human") -> Optional[Dict[str, Any]]:
        """
        Get VEP (Variant Effect Predictor) annotation for a variant.

        Args:
            variant: Variant in format "chr:pos:ref:alt" or "rsID"
            species: Species name (default: "human")

        Returns:
            VEP annotation dict or None
        """
        try:
            # VEP region endpoint: /vep/{species}/region/{region}
            url = f"{self.BASE_URL}/vep/{species}/region/{variant}"
            headers = {'Content-Type': 'application/json'}

            response = self.client.get(url, headers=headers)

            if response.status_code == 200:
                return response.json()

            return None

        except Exception as e:
            logger.error(f"Ensembl VEP annotation error for {variant}: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_gene(self, gene_symbol: str, species: str = "human") -> Optional[Dict[str, Any]]:
        """
        Look up gene by symbol.

        Args:
            gene_symbol: Gene symbol (e.g., 'TCF7L2')
            species: Species name (default: "human")

        Returns:
            Gene information dict or None
        """
        try:
            url = f"{self.BASE_URL}/lookup/symbol/{species}/{gene_symbol}"
            headers = {'Content-Type': 'application/json'}

            response = self.client.get(url, headers=headers)

            if response.status_code == 200:
                return response.json()

            return None

        except Exception as e:
            logger.error(f"Ensembl gene lookup error for {gene_symbol}: {e}")
            return None

    def close(self):
        """Close HTTP client."""
        self.client.close()


class HMDBClient:
    """Client for HMDB (Human Metabolome Database)."""

    BASE_URL = "https://hmdb.ca"

    def __init__(self, timeout: int = 30):
        """Initialize HMDB client."""
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout, follow_redirects=True)

    def search_metabolite(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Search for metabolite by name.

        Args:
            name: Metabolite name

        Returns:
            Metabolite information or None

        Note: HMDB API is limited. This is a placeholder for future implementation
        or web scraping approach.
        """
        logger.warning("HMDB API is limited. Consider using KEGG or local database.")
        return None

    def get_metabolite(self, hmdb_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metabolite by HMDB ID.

        Args:
            hmdb_id: HMDB identifier (e.g., 'HMDB0000001')

        Returns:
            Metabolite information or None

        Note: HMDB API is limited. This is a placeholder for future implementation.
        """
        logger.warning(f"HMDB API implementation pending for {hmdb_id}. Consider using KEGG or local database.")
        return None

    def close(self):
        """Close HTTP client."""
        self.client.close()


class MetaboLightsClient:
    """Client for MetaboLights (public metabolomics repository)."""

    BASE_URL = "https://www.ebi.ac.uk/metabolights/ws"

    def __init__(self, timeout: int = 30):
        """Initialize MetaboLights client."""
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_study(self, study_id: str) -> Optional[Dict[str, Any]]:
        """
        Get study information.

        Args:
            study_id: MetaboLights study ID (e.g., 'MTBLS1')

        Returns:
            Study information dict or None
        """
        try:
            url = f"{self.BASE_URL}/studies/{study_id}"
            response = self.client.get(url)

            if response.status_code == 200:
                return response.json()

            return None

        except Exception as e:
            logger.error(f"MetaboLights error for {study_id}: {e}")
            return None

    def close(self):
        """Close HTTP client."""
        self.client.close()


class UniProtClient:
    """Client for UniProt (protein database)."""

    BASE_URL = "https://rest.uniprot.org"

    def __init__(self, timeout: int = 30):
        """Initialize UniProt client."""
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout, follow_redirects=True)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_protein(self, uniprot_id: str) -> Optional[Dict[str, Any]]:
        """
        Get protein information.

        Args:
            uniprot_id: UniProt accession (e.g., 'P04637')

        Returns:
            Protein information dict or None
        """
        try:
            url = f"{self.BASE_URL}/uniprotkb/{uniprot_id}.json"
            response = self.client.get(url)

            if response.status_code == 200:
                return response.json()

            return None

        except Exception as e:
            logger.error(f"UniProt error for {uniprot_id}: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def search_by_gene(self, gene_name: str) -> Optional[Dict[str, Any]]:
        """
        Search proteins by gene name.

        Args:
            gene_name: Gene symbol (e.g., 'TP53')

        Returns:
            Search results dict or None
        """
        try:
            url = f"{self.BASE_URL}/uniprotkb/search"
            params = {
                'query': f'gene:{gene_name}',
                'format': 'json',
                'size': 10
            }
            response = self.client.get(url, params=params)

            if response.status_code == 200:
                return response.json()

            return None

        except Exception as e:
            logger.error(f"UniProt gene search error for {gene_name}: {e}")
            return None

    def close(self):
        """Close HTTP client."""
        self.client.close()


class PDBClient:
    """Client for PDB (Protein Data Bank)."""

    BASE_URL = "https://data.rcsb.org/rest/v1"

    def __init__(self, timeout: int = 30):
        """Initialize PDB client."""
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_structure(self, pdb_id: str) -> Optional[Dict[str, Any]]:
        """
        Get protein structure information.

        Args:
            pdb_id: PDB ID (e.g., '1ABC')

        Returns:
            Structure information dict or None
        """
        try:
            url = f"{self.BASE_URL}/core/entry/{pdb_id}"
            response = self.client.get(url)

            if response.status_code == 200:
                return response.json()

            return None

        except Exception as e:
            logger.error(f"PDB error for {pdb_id}: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def search_structures(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Search protein structures by query.

        Args:
            query: Search query (protein name, gene, organism, etc.)

        Returns:
            Search results dict or None
        """
        try:
            # Use RCSB search API
            search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
            search_query = {
                "query": {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "value": query
                    }
                },
                "return_type": "entry",
                "request_options": {
                    "results_verbosity": "minimal",
                    "return_all_hits": False,
                    "results_content_type": ["experimental"]
                }
            }

            response = self.client.post(
                search_url,
                json=search_query,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                return response.json()

            return None

        except Exception as e:
            logger.error(f"PDB search error for '{query}': {e}")
            return None

    def close(self):
        """Close HTTP client."""
        self.client.close()
