"""
Unit tests for Materials domain API clients (Phase 9).

Tests 5 materials science API clients:
- MaterialsProject: Computed material properties database
- NOMAD: Materials data repository
- AFLOW: Automatic FLOW for Materials Discovery
- Citrination: Materials informatics platform
- PerovskiteDB: Perovskite solar cell experimental data

Coverage target: 35 tests (5 clients × 7 tests each)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import pandas as pd
from kosmos.domains.materials.apis import (
    MaterialsProjectClient,
    NOMADClient,
    AflowClient,
    CitrinationClient,
    PerovskiteDBClient,
    MaterialProperties,
    NomadEntry,
    AflowMaterial,
    CitrinationData,
    PerovskiteExperiment
)


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for API testing"""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"success": True, "data": "test"}
    mock_response.text = "test_data"
    mock_client.get = Mock(return_value=mock_response)
    mock_client.post = Mock(return_value=mock_response)
    mock_client.close = Mock()
    return mock_client


@pytest.mark.unit
class TestMaterialsProjectClient:
    """Test Materials Project API client"""

    def test_init_default(self):
        """Test MaterialsProjectClient initialization without API key"""
        client = MaterialsProjectClient()
        assert hasattr(client, 'client')
        assert client.api_key is None
        assert MaterialsProjectClient.BASE_URL == "https://api.materialsproject.org"

    def test_get_material_success(self, mock_httpx_client):
        """Test successful material retrieval by ID"""
        # Configure mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            'formula_pretty': 'Si',
            'formula': 'Si1',
            'energy_per_atom': -5.425,
            'band_gap': 1.14,
            'density': 2.33,
            'formation_energy_per_atom': 0.0,
            'is_stable': True
        }
        mock_httpx_client.get.return_value = mock_response

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = MaterialsProjectClient(api_key="test_key")
            result = client.get_material("mp-149")

            assert result is not None
            assert isinstance(result, MaterialProperties)
            assert result.material_id == "mp-149"
            assert result.formula == "Si"
            assert result.band_gap == 1.14

    def test_search_materials(self, mock_httpx_client):
        """Test material search with filters"""
        # Configure mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            'data': [
                {
                    'material_id': 'mp-149',
                    'formula_pretty': 'Si',
                    'band_gap': 1.14,
                    'density': 2.33
                },
                {
                    'material_id': 'mp-150',
                    'formula_pretty': 'Ge',
                    'band_gap': 0.74,
                    'density': 5.32
                }
            ]
        }
        mock_httpx_client.get.return_value = mock_response

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = MaterialsProjectClient(api_key="test_key")
            results = client.search_materials(
                elements=['Si'],
                band_gap_min=1.0,
                band_gap_max=2.0,
                limit=10
            )

            assert isinstance(results, list)
            assert len(results) == 2
            assert all(isinstance(r, MaterialProperties) for r in results)
            assert results[0].material_id == 'mp-149'

    def test_properties_retrieval(self, mock_httpx_client):
        """Test extraction of material properties"""
        mock_response = Mock()
        mock_response.json.return_value = {
            'formula_pretty': 'GaAs',
            'energy_per_atom': -3.82,
            'band_gap': 1.52,
            'density': 5.32,
            'formation_energy_per_atom': -0.52,
            'is_stable': True,
            'elasticity': {'K_VRH': 75.2, 'G_VRH': 59.1}
        }
        mock_httpx_client.get.return_value = mock_response

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = MaterialsProjectClient(api_key="test_key")
            result = client.get_material("mp-2534")

            assert result.energy_per_atom == -3.82
            assert result.formation_energy == -0.52
            assert result.is_stable is True
            assert result.elasticity is not None
            assert 'K_VRH' in result.elasticity

    def test_structure_data(self, mock_httpx_client):
        """Test crystal structure data retrieval"""
        mock_response = Mock()
        mock_response.json.return_value = {
            'formula_pretty': 'Fe',
            'structure': {
                'lattice': {'a': 2.87, 'b': 2.87, 'c': 2.87},
                'sites': [{'species': [{'element': 'Fe'}], 'xyz': [0, 0, 0]}]
            }
        }
        mock_httpx_client.get.return_value = mock_response

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = MaterialsProjectClient(api_key="test_key")
            result = client.get_material("mp-13")

            assert result.structure is not None
            assert 'lattice' in result.structure
            assert 'sites' in result.structure

    def test_api_key_authentication(self, mock_httpx_client):
        """Test API key is properly set in headers"""
        with patch('httpx.Client', return_value=mock_httpx_client) as mock_constructor:
            client = MaterialsProjectClient(api_key="my_api_key")

            # Verify Client was created with API key header
            mock_constructor.assert_called_once()
            call_kwargs = mock_constructor.call_args.kwargs
            assert 'headers' in call_kwargs
            assert call_kwargs['headers']['X-API-KEY'] == "my_api_key"

    def test_error_handling(self, mock_httpx_client):
        """Test error handling returns None on failure"""
        import httpx
        mock_httpx_client.get.side_effect = httpx.HTTPError("API Error")

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = MaterialsProjectClient(api_key="test_key")
            result = client.get_material("invalid-id")

            assert result is None


@pytest.mark.unit
class TestNOMADClient:
    """Test NOMAD API client"""

    def test_init_default(self):
        """Test NOMAD client initialization"""
        client = NOMADClient()
        assert hasattr(client, 'client')
        assert NOMADClient.BASE_URL == "https://nomad-lab.eu/prod/v1/api/v1"

    def test_search_entries(self, mock_httpx_client):
        """Test NOMAD entry search"""
        mock_response = Mock()
        mock_response.json.return_value = {
            'data': [
                {
                    'entry_id': 'entry123',
                    'upload_id': 'upload456',
                    'material': {
                        'material_name': 'Silicon',
                        'chemical_formula_hill': 'Si1'
                    },
                    'entry_type': 'calculation',
                    'properties': {'band_gap': 1.14}
                }
            ]
        }
        mock_httpx_client.post.return_value = mock_response

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = NOMADClient()
            results = client.search_materials(formula='Si', limit=10)

            assert isinstance(results, list)
            assert len(results) == 1
            assert isinstance(results[0], NomadEntry)
            assert results[0].entry_id == 'entry123'
            assert results[0].formula == 'Si1'

    def test_get_entry_data(self, mock_httpx_client):
        """Test retrieving specific NOMAD entry"""
        mock_response = Mock()
        mock_response.json.return_value = {
            'entry_id': 'test_entry',
            'upload_id': 'test_upload',
            'material': {
                'material_name': 'TiO2',
                'chemical_formula_hill': 'O2Ti1'
            },
            'entry_type': 'experiment',
            'properties': {'band_gap': 3.2}
        }
        mock_httpx_client.get.return_value = mock_response

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = NOMADClient()
            result = client.get_entry("test_entry")

            assert result is not None
            assert isinstance(result, NomadEntry)
            assert result.entry_id == "test_entry"
            assert result.material_name == "TiO2"

    def test_metadata_retrieval(self, mock_httpx_client):
        """Test metadata extraction from NOMAD entries"""
        mock_response = Mock()
        mock_response.json.return_value = {
            'data': [
                {
                    'entry_id': 'test',
                    'upload_id': 'upload',
                    'material': {'chemical_formula_hill': 'Fe2O3'},
                    'entry_type': 'calculation',
                    'properties': {'energy': -12.5}
                }
            ]
        }
        mock_httpx_client.post.return_value = mock_response

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = NOMADClient()
            results = client.search_materials(formula='Fe2O3')

            assert results[0].metadata is not None
            assert results[0].properties == {'energy': -12.5}
            assert results[0].data_type == 'calculation'

    def test_filtering_options(self, mock_httpx_client):
        """Test search filtering by elements and data type"""
        mock_response = Mock()
        mock_response.json.return_value = {'data': []}
        mock_httpx_client.post.return_value = mock_response

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = NOMADClient()
            results = client.search_materials(
                elements=['Ti', 'O'],
                data_type='experiment',
                limit=50
            )

            # Verify POST was called with correct query
            assert mock_httpx_client.post.called
            call_args = mock_httpx_client.post.call_args
            assert 'json' in call_args.kwargs
            query_data = call_args.kwargs['json']
            assert 'query' in query_data
            assert 'pagination' in query_data

    def test_pagination(self, mock_httpx_client):
        """Test pagination parameter handling"""
        mock_response = Mock()
        mock_response.json.return_value = {'data': []}
        mock_httpx_client.post.return_value = mock_response

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = NOMADClient()
            client.search_materials(formula='Si', limit=200)

            # Verify pagination was set
            call_args = mock_httpx_client.post.call_args
            query_data = call_args.kwargs['json']
            assert query_data['pagination']['page_size'] == 200

    def test_error_handling(self, mock_httpx_client):
        """Test error handling returns empty list on failure"""
        import httpx
        mock_httpx_client.post.side_effect = httpx.HTTPError("Connection error")

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = NOMADClient()
            results = client.search_materials(formula='invalid')

            assert results == []


@pytest.mark.unit
class TestAflowClient:
    """Test AFLOW API client"""

    def test_init_default(self):
        """Test AFLOW client initialization"""
        client = AflowClient()
        assert hasattr(client, 'client')
        assert AflowClient.BASE_URL == "http://aflowlib.org/API/aflux"

    def test_search_materials(self, mock_httpx_client):
        """Test AFLOW material search with AFLUX query"""
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                'auid': 'aflow:123abc',
                'compound': 'TiO2',
                'prototype': 'A2B_oP12_62_2c_c',
                'spacegroup_relax': 136,
                'energy_atom': -8.94,
                'Egap': 3.2,
                'density': 4.23
            }
        ]
        mock_httpx_client.get.return_value = mock_response

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = AflowClient()
            results = client.search_materials(compound='TiO2', limit=10)

            assert isinstance(results, list)
            assert len(results) == 1
            assert isinstance(results[0], AflowMaterial)
            assert results[0].compound == 'TiO2'
            assert results[0].band_gap == 3.2

    def test_properties_query(self, mock_httpx_client):
        """Test querying material properties"""
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                'auid': 'test',
                'compound': 'GaN',
                'energy_atom': -3.45,
                'Egap': 3.4,
                'density': 6.15
            }
        ]
        mock_httpx_client.get.return_value = mock_response

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = AflowClient()
            results = client.search_materials(compound='GaN')

            assert results[0].energy_per_atom == -3.45
            assert results[0].band_gap == 3.4
            assert results[0].density == 6.15

    def test_structure_retrieval(self, mock_httpx_client):
        """Test crystal structure retrieval"""
        mock_response = Mock()
        mock_response.json.return_value = {
            'auid': 'aflow:456def',
            'compound': 'Fe',
            'prototype': 'A_cI2_229_a',
            'spacegroup_relax': 229,
            'energy_atom': -8.31
        }
        mock_httpx_client.get.return_value = mock_response

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = AflowClient()
            result = client.get_material('aflow:456def')

            assert result is not None
            assert result.prototype == 'A_cI2_229_a'
            assert result.space_group == 229

    def test_composition_search(self, mock_httpx_client):
        """Test search by composition elements"""
        mock_response = Mock()
        mock_response.json.return_value = [
            {'auid': 'test1', 'compound': 'FeO'},
            {'auid': 'test2', 'compound': 'Fe2O3'}
        ]
        mock_httpx_client.get.return_value = mock_response

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = AflowClient()
            results = client.search_materials(elements=['Fe', 'O'])

            # Verify AFLUX query was constructed
            assert mock_httpx_client.get.called
            call_url = mock_httpx_client.get.call_args[0][0]
            assert 'species(Fe)' in call_url
            assert 'species(O)' in call_url

    def test_batch_queries(self, mock_httpx_client):
        """Test batch query with paging parameter"""
        mock_response = Mock()
        mock_response.json.return_value = [
            {'auid': f'aflow:test{i}', 'compound': f'Compound{i}'}
            for i in range(50)
        ]
        mock_httpx_client.get.return_value = mock_response

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = AflowClient()
            results = client.search_materials(compound='Si', limit=50)

            # Verify paging parameter
            call_url = mock_httpx_client.get.call_args[0][0]
            assert '$paging(50)' in call_url

    def test_error_handling(self, mock_httpx_client):
        """Test error handling returns None or empty list"""
        import httpx
        mock_httpx_client.get.side_effect = httpx.HTTPError("AFLOW error")

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = AflowClient()

            # get_material returns None on error
            result = client.get_material('invalid')
            assert result is None

            # search_materials returns empty list on error
            results = client.search_materials(compound='invalid')
            assert results == []


@pytest.mark.unit
class TestCitrinationClient:
    """Test Citrination API client"""

    def test_init_default(self):
        """Test Citrination client initialization"""
        client = CitrinationClient()
        assert hasattr(client, 'client')
        assert client.api_key is None
        assert CitrinationClient.BASE_URL == "https://citrination.com/api"

    def test_search_datasets(self, mock_httpx_client):
        """Test dataset search"""
        mock_response = Mock()
        mock_response.json.return_value = {
            'hits': [
                {
                    '_source': {
                        'uid': 'dataset123',
                        'chemicalFormula': 'LiFePO4',
                        'properties': [
                            {
                                'name': 'battery_capacity',
                                'scalars': [{'value': 170}]
                            }
                        ]
                    }
                }
            ]
        }
        mock_httpx_client.post.return_value = mock_response

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = CitrinationClient(api_key="test_key")
            results = client.search_datasets("battery materials", limit=10)

            assert isinstance(results, list)
            assert len(results) == 1
            assert isinstance(results[0], CitrinationData)
            assert results[0].dataset_id == 'dataset123'
            assert results[0].material_name == 'LiFePO4'

    def test_pif_retrieval(self, mock_httpx_client):
        """Test Physical Information File (PIF) data extraction"""
        mock_response = Mock()
        mock_response.json.return_value = {
            'hits': [
                {
                    '_source': {
                        'uid': 'pif_001',
                        'chemicalFormula': 'Al2O3',
                        'properties': [
                            {'name': 'hardness', 'scalars': [{'value': 9.0}]},
                            {'name': 'density', 'scalars': [{'value': 3.95}]}
                        ],
                        'conditions': {'temperature': 300}
                    }
                }
            ]
        }
        mock_httpx_client.post.return_value = mock_response

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = CitrinationClient(api_key="test_key")
            results = client.search_datasets("Al2O3")

            assert results[0].properties['hardness'] == 9.0
            assert results[0].properties['density'] == 3.95
            assert results[0].conditions == {'temperature': 300}

    def test_property_data(self, mock_httpx_client):
        """Test property data extraction from PIF"""
        mock_response = Mock()
        mock_response.json.return_value = {
            'hits': [
                {
                    '_source': {
                        'uid': 'test',
                        'chemicalFormula': 'TiO2',
                        'properties': [
                            {'name': 'band_gap', 'scalars': [{'value': 3.2}]},
                            {'name': 'refractive_index', 'scalars': [{'value': 2.6}]}
                        ]
                    }
                }
            ]
        }
        mock_httpx_client.post.return_value = mock_response

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = CitrinationClient(api_key="test_key")
            results = client.search_datasets("TiO2")

            props = results[0].properties
            assert 'band_gap' in props
            assert 'refractive_index' in props
            assert props['band_gap'] == 3.2

    def test_api_key_authentication(self, mock_httpx_client):
        """Test API key authentication header"""
        with patch('httpx.Client', return_value=mock_httpx_client) as mock_constructor:
            client = CitrinationClient(api_key="my_citrination_key")

            # Verify Client was created with API key header
            mock_constructor.assert_called_once()
            call_kwargs = mock_constructor.call_args.kwargs
            assert 'headers' in call_kwargs
            assert call_kwargs['headers']['X-API-Key'] == "my_citrination_key"

    def test_data_views(self, mock_httpx_client):
        """Test data extraction from hits/source structure"""
        mock_response = Mock()
        mock_response.json.return_value = {
            'hits': [
                {'_source': {'uid': '1', 'chemicalFormula': 'A', 'properties': []}},
                {'_source': {'uid': '2', 'chemicalFormula': 'B', 'properties': []}}
            ]
        }
        mock_httpx_client.post.return_value = mock_response

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = CitrinationClient(api_key="test_key")
            results = client.search_datasets("test", limit=10)

            assert len(results) == 2
            assert results[0].dataset_id == '1'
            assert results[1].dataset_id == '2'

    def test_error_handling(self, mock_httpx_client):
        """Test error handling returns empty list on failure"""
        import httpx
        mock_httpx_client.post.side_effect = httpx.HTTPError("Citrination error")

        with patch('httpx.Client', return_value=mock_httpx_client):
            client = CitrinationClient(api_key="test_key")
            results = client.search_datasets("invalid")

            assert results == []


@pytest.mark.unit
class TestPerovskiteDBClient:
    """Test Perovskite Database client (file-based)"""

    def test_init_default(self):
        """Test PerovskiteDB client initialization"""
        client = PerovskiteDBClient()
        assert client is not None

    def test_get_experiment_data(self, tmp_path):
        """Test loading experiment data from file"""
        # Create test CSV file
        csv_file = tmp_path / "test_data.csv"
        test_data = pd.DataFrame({
            'Composition': ['MAPbI3', 'FAPbI3'],
            'Temperature': [100, 150],
            'Jsc': [22.5, 23.1],
            'Voc': [1.05, 1.08],
            'Fill Factor': [0.75, 0.78],
            'Efficiency': [17.7, 19.4]
        })
        test_data.to_csv(csv_file, index=False)

        client = PerovskiteDBClient()
        df = client.load_dataset(str(csv_file))

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'Jsc' in df.columns
        assert df['Efficiency'].iloc[0] == 17.7

    def test_search_perovskites(self, tmp_path):
        """Test searching/filtering perovskite data"""
        pytest.importorskip("openpyxl")  # pandas' Excel engine; not a declared dependency
        # Create test data
        xlsx_file = tmp_path / "perovskites.xlsx"
        test_data = pd.DataFrame({
            'Material': ['MAPbI3', 'FAPbI3', 'CsPbI3'],
            'Jsc': [22.5, 23.1, 18.2],
            'Efficiency': [17.7, 19.4, 15.1]
        })
        test_data.to_excel(xlsx_file, index=False)

        client = PerovskiteDBClient()
        df = client.load_dataset(str(xlsx_file))

        # Test filtering
        high_efficiency = df[df['Efficiency'] > 17]
        assert len(high_efficiency) == 2

    def test_properties_filtering(self, tmp_path):
        """Test filtering by performance metrics"""
        csv_file = tmp_path / "data.csv"
        test_data = pd.DataFrame({
            'Jsc': [20.0, 22.0, 24.0],
            'Voc': [1.0, 1.05, 1.1],
            'Fill Factor': [0.70, 0.75, 0.80],
            'Efficiency': [14.0, 17.3, 21.1]
        })
        test_data.to_csv(csv_file, index=False)

        client = PerovskiteDBClient()
        df = client.load_dataset(str(csv_file))

        # Filter by Jsc > 21
        filtered = df[df['Jsc'] > 21]
        assert len(filtered) == 2
        assert filtered['Efficiency'].min() > 17

    def test_composition_queries(self):
        """Test parsing experiment composition data"""
        client = PerovskiteDBClient()

        # Test parse_experiments method
        df = pd.DataFrame({
            'PbI2': [1.0, 1.2],
            'MAI': [1.0, 0.8],
            'Pressure': [10, 20],
            'Temperature': [100, 120],
            'Jsc': [22.5, 21.8],
            'Voc': [1.05, 1.03],
            'Fill Factor': [0.75, 0.74],
            'Efficiency': [17.7, 16.6]
        })

        experiments = client.parse_experiments(
            df,
            composition_cols=['PbI2', 'MAI'],
            fabrication_cols=['Pressure', 'Temperature']
        )

        assert len(experiments) == 2
        assert isinstance(experiments[0], PerovskiteExperiment)
        assert 'PbI2' in experiments[0].composition
        assert 'Pressure' in experiments[0].fabrication_params

    def test_performance_metrics(self):
        """Test extraction of solar cell performance metrics"""
        client = PerovskiteDBClient()

        df = pd.DataFrame({
            'Jsc': [22.5, 23.1, 21.8],
            'Voc': [1.05, 1.08, 1.02],
            'Fill Factor': [0.75, 0.78, 0.73],
            'Efficiency': [17.7, 19.4, 16.2]
        })

        experiments = client.parse_experiments(df)

        assert experiments[0].jsc == 22.5
        assert experiments[0].voc == 1.05
        assert experiments[0].fill_factor == 0.75
        assert experiments[0].efficiency == 17.7
        assert experiments[1].efficiency == 19.4

    def test_error_handling(self):
        """Test error handling for missing files and unsupported formats"""
        client = PerovskiteDBClient()

        # Test missing file
        df = client.load_dataset("nonexistent_file.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

        # Test unsupported format
        df = client.load_dataset("data.txt")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
