"""
Unit tests for kosmos.agents.skill_loader module.

Tests:
- SkillLoader: skill discovery, loading, bundling
- Domain matching
- Prompt injection formatting
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from kosmos.agents.skill_loader import SkillLoader


# ============================================================================
# Fixtures
# ============================================================================

def _write_skill(skills_path, name, content):
    """Write one skill in the layout SkillLoader discovers: <skills_dir>/<name>/SKILL.md."""
    skill_folder = skills_path / name
    skill_folder.mkdir(parents=True, exist_ok=True)
    skill_file = skill_folder / "SKILL.md"
    skill_file.write_text(content, encoding="utf-8")
    return skill_file


@pytest.fixture
def skills_dir(temp_dir):
    """Create a mock skills directory structure.

    SkillLoader (since df310b5, issue #67) discovers one skill per top-level
    folder that contains a SKILL.md file; the folder name is the skill name.
    """
    skills_path = temp_dir / "scientific-skills"
    skills_path.mkdir(parents=True, exist_ok=True)

    _write_skill(skills_path, "pandas", """
# Pandas

Data manipulation and analysis library.

## Installation
```bash
pip install pandas
```

## Basic Usage
```python
import pandas as pd

df = pd.read_csv('data.csv')
df.head()
df.describe()
```

## Common Operations
- read_csv, to_csv
- groupby, merge, join
- fillna, dropna
""")

    _write_skill(skills_path, "scipy", """
# SciPy

Scientific computing library.

## Usage
```python
from scipy import stats

result = stats.ttest_ind(group1, group2)
```
""")

    _write_skill(skills_path, "scanpy", """
# Scanpy

Single-cell analysis toolkit.

## Usage
```python
import scanpy as sc

adata = sc.read_h5ad('data.h5ad')
sc.pp.filter_cells(adata, min_genes=200)
```
""")

    _write_skill(skills_path, "ensembl-database", """
# Ensembl Database

Genome database for vertebrates and model organisms.

## API Access
```python
from biomart import BiomartServer

server = BiomartServer("http://www.ensembl.org/biomart")
```
""")

    return skills_path


@pytest.fixture
def skill_loader_with_skills(skills_dir):
    """Create SkillLoader with mock skills directory."""
    return SkillLoader(skills_dir=str(skills_dir), auto_discover=True)


@pytest.fixture
def empty_skill_loader(temp_dir):
    """Create SkillLoader with empty skills directory."""
    empty_dir = temp_dir / "empty-skills"
    empty_dir.mkdir()
    return SkillLoader(skills_dir=str(empty_dir), auto_discover=True)


# ============================================================================
# SkillLoader Initialization Tests
# ============================================================================

class TestSkillLoaderInit:
    """Tests for SkillLoader initialization."""

    def test_init_with_skills_dir(self, skills_dir):
        """Test initialization with explicit skills directory."""
        loader = SkillLoader(skills_dir=str(skills_dir), auto_discover=True)

        assert loader.skills_dir == skills_dir
        assert len(loader.skills_cache) > 0

    def test_init_without_skills_dir(self):
        """Test initialization without skills directory."""
        loader = SkillLoader(skills_dir=None, auto_discover=False)

        assert loader.skills_dir is None
        assert loader.skills_cache == {}

    def test_init_with_auto_discover(self, skills_dir):
        """Test auto-discovery of skills."""
        loader = SkillLoader(skills_dir=str(skills_dir), auto_discover=True)

        # Should find every <name>/SKILL.md folder
        assert 'pandas' in loader.skills_cache
        assert 'scipy' in loader.skills_cache
        assert 'scanpy' in loader.skills_cache
        assert 'ensembl-database' in loader.skills_cache

    def test_init_without_auto_discover(self, skills_dir):
        """Test initialization without auto-discovery."""
        loader = SkillLoader(skills_dir=str(skills_dir), auto_discover=False)

        assert loader.skills_cache == {}

    def test_skill_bundles_defined(self):
        """Test that skill bundles are predefined."""
        loader = SkillLoader(auto_discover=False)

        assert 'single_cell_analysis' in loader.SKILL_BUNDLES
        assert 'genomics_analysis' in loader.SKILL_BUNDLES
        assert 'drug_discovery' in loader.SKILL_BUNDLES
        assert 'scanpy' in loader.SKILL_BUNDLES['single_cell_analysis']

    def test_common_skills_defined(self):
        """Test that common skills are predefined."""
        loader = SkillLoader(auto_discover=False)

        # df310b5 (#67): COMMON_SKILLS lists only skills that exist as skill
        # folders; pandas/numpy are plain imports, not skill files.
        assert 'scikit-learn' in loader.COMMON_SKILLS
        assert 'statsmodels' in loader.COMMON_SKILLS
        assert 'pandas' not in loader.COMMON_SKILLS
        assert 'numpy' not in loader.COMMON_SKILLS


# ============================================================================
# Skill Discovery Tests
# ============================================================================

class TestSkillDiscovery:
    """Tests for skill discovery."""

    def test_discover_skills(self, skill_loader_with_skills):
        """Test skill discovery from directory."""
        loader = skill_loader_with_skills

        # Should have discovered skills
        assert len(loader.skills_cache) == 4  # pandas, scipy, scanpy, ensembl

    def test_discovered_skill_metadata(self, skill_loader_with_skills):
        """Test that discovered skills have correct metadata."""
        loader = skill_loader_with_skills
        pandas_skill = loader.skills_cache.get('pandas')

        assert pandas_skill is not None
        assert 'path' in pandas_skill
        assert 'name' in pandas_skill
        assert 'category' in pandas_skill
        assert pandas_skill['name'] == 'pandas'
        assert pandas_skill['path'].endswith('SKILL.md')
        # Category is the skill folder's name
        assert pandas_skill['category'] == 'pandas'
        assert pandas_skill['loaded'] is False  # Lazy loading

    def test_discover_only_skill_folders(self, skills_dir):
        """Only top-level folders containing SKILL.md are discovered as skills."""
        # A loose .md file is not a skill
        (skills_dir / "loose.md").write_text("# Loose\n\nNot a skill folder.")
        # A folder without SKILL.md is not a skill
        (skills_dir / "notes").mkdir()
        (skills_dir / "notes" / "README.md").write_text("# Notes\n")
        # A new skill folder is discovered
        _write_skill(skills_dir, "pytorch", "# PyTorch\n\nDeep learning framework.")

        loader = SkillLoader(skills_dir=str(skills_dir), auto_discover=True)

        assert 'pytorch' in loader.skills_cache
        assert 'loose' not in loader.skills_cache
        assert 'notes' not in loader.skills_cache
        assert len(loader.skills_cache) == 5


# ============================================================================
# Skill Loading Tests
# ============================================================================

class TestSkillLoading:
    """Tests for loading individual skills."""

    def test_load_skill_by_name(self, skill_loader_with_skills):
        """Test loading skill by name."""
        loader = skill_loader_with_skills
        skill = loader.load_skill('pandas')

        assert skill is not None
        assert skill['name'] == 'pandas'
        assert skill['loaded'] is True
        assert 'content' in skill
        assert 'Pandas' in skill['content']

    def test_load_skill_from_cache(self, skill_loader_with_skills):
        """Test that loaded skills are cached."""
        loader = skill_loader_with_skills

        # Load twice
        skill1 = loader.load_skill('pandas')
        skill2 = loader.load_skill('pandas')

        # Should be same object (cached)
        assert skill1 is skill2

    def test_load_skill_not_found(self, skill_loader_with_skills):
        """Test loading non-existent skill."""
        loader = skill_loader_with_skills
        skill = loader.load_skill('nonexistent_skill')

        assert skill is None

    def test_load_skill_from_file(self, skill_loader_with_skills, skills_dir):
        """Test loading skill directly from file path."""
        loader = skill_loader_with_skills
        skill_path = skills_dir / "scipy" / "SKILL.md"

        skill = loader.load_skill_from_file(skill_path)

        # A SKILL.md file takes its name from the parent folder
        assert skill['name'] == 'scipy'
        assert 'SciPy' in skill['content']
        assert skill['loaded'] is True


# ============================================================================
# Skill Parsing Tests
# ============================================================================

class TestSkillParsing:
    """Tests for skill markdown parsing."""

    def test_parse_skill_description(self, skill_loader_with_skills):
        """Test extraction of skill description."""
        loader = skill_loader_with_skills
        skill = loader.load_skill('pandas')

        assert 'description' in skill
        assert len(skill['description']) > 0

    def test_parse_skill_examples(self, skill_loader_with_skills):
        """Test extraction of code examples."""
        loader = skill_loader_with_skills
        skill = loader.load_skill('pandas')

        assert 'examples' in skill
        assert len(skill['examples']) > 0
        assert 'import pandas' in skill['examples'][0]

    def test_parse_empty_skill(self, temp_dir):
        """Test parsing skill with minimal content."""
        skills_dir = temp_dir / "minimal-skills"
        skills_dir.mkdir()
        _write_skill(skills_dir, "empty", "# Empty Skill\n")

        loader = SkillLoader(skills_dir=str(skills_dir), auto_discover=True)
        skill = loader.load_skill('empty')

        assert skill is not None
        assert skill['examples'] == []


# ============================================================================
# Skill Bundle Tests
# ============================================================================

class TestSkillBundles:
    """Tests for skill bundles."""

    def test_get_available_bundles(self):
        """Test getting list of available bundles."""
        loader = SkillLoader(auto_discover=False)
        bundles = loader.get_available_bundles()

        assert 'single_cell_analysis' in bundles
        assert 'genomics_analysis' in bundles
        assert len(bundles) >= 8

    def test_get_bundle_skills(self):
        """Test getting skills in a bundle."""
        loader = SkillLoader(auto_discover=False)
        skills = loader.get_bundle_skills('single_cell_analysis')

        assert 'scanpy' in skills
        assert 'anndata' in skills

    def test_get_bundle_skills_nonexistent(self):
        """Test getting skills from non-existent bundle."""
        loader = SkillLoader(auto_discover=False)
        skills = loader.get_bundle_skills('nonexistent_bundle')

        assert skills == []


# ============================================================================
# Load Skills for Task Tests
# ============================================================================

class TestLoadSkillsForTask:
    """Tests for loading skills based on task requirements."""

    def test_load_skills_for_task_type(self, skill_loader_with_skills):
        """Test loading skills for a specific task type."""
        loader = skill_loader_with_skills
        skills_text = loader.load_skills_for_task(
            task_type='single_cell_analysis',
            include_examples=False,
            include_common=False
        )

        # Should include scanpy from single_cell_analysis bundle
        # (if it was found in the mock directory)
        assert isinstance(skills_text, str)

    def test_load_skills_for_domain(self, skill_loader_with_skills):
        """Test loading skills by domain."""
        loader = skill_loader_with_skills
        skills_text = loader.load_skills_for_task(
            domain='single_cell_analysis',
            include_common=False
        )

        assert isinstance(skills_text, str)

    def test_load_skills_with_specific_libraries(self, skill_loader_with_skills):
        """Test loading specific libraries."""
        loader = skill_loader_with_skills
        skills_text = loader.load_skills_for_task(
            libraries=['pandas', 'scipy'],
            include_common=False
        )

        # Should include requested libraries if available
        assert isinstance(skills_text, str)

    def test_load_skills_with_common(self, skill_loader_with_skills):
        """Test that common skills are included when requested."""
        loader = skill_loader_with_skills
        skills_text = loader.load_skills_for_task(
            task_type='single_cell_analysis',
            include_common=True
        )

        # Should be formatted for prompt
        assert 'Scientific Skills' in skills_text or skills_text == ''

    def test_load_skills_with_examples(self, skill_loader_with_skills):
        """Test including code examples."""
        loader = skill_loader_with_skills
        skills_text = loader.load_skills_for_task(
            libraries=['pandas'],
            include_examples=True,
            include_common=False
        )

        # If pandas was loaded, should include example
        if 'pandas' in skills_text.lower():
            assert '```python' in skills_text

    def test_load_skills_empty_request(self, skill_loader_with_skills):
        """Test loading with no specific requirements."""
        loader = skill_loader_with_skills
        skills_text = loader.load_skills_for_task(
            include_common=False
        )

        # Should return empty or minimal
        assert isinstance(skills_text, str)


# ============================================================================
# Prompt Formatting Tests
# ============================================================================

class TestPromptFormatting:
    """Tests for skill prompt formatting."""

    def test_format_skills_for_prompt(self, skill_loader_with_skills):
        """Test formatting skills for prompt injection."""
        loader = skill_loader_with_skills

        # Load some skills
        skills = [loader.load_skill('pandas')]
        skills = [s for s in skills if s is not None]

        if skills:
            formatted = loader._format_skills_for_prompt(skills, include_examples=False)

            assert '# Scientific Skills' in formatted
            assert 'pandas' in formatted

    def test_format_skills_with_examples(self, skill_loader_with_skills):
        """Test formatting with code examples."""
        loader = skill_loader_with_skills

        skills = [loader.load_skill('pandas')]
        skills = [s for s in skills if s is not None]

        if skills:
            formatted = loader._format_skills_for_prompt(skills, include_examples=True)

            assert '**Example Usage**' in formatted
            assert '```python' in formatted

    def test_format_empty_skills(self, skill_loader_with_skills):
        """Test formatting empty skill list."""
        loader = skill_loader_with_skills
        formatted = loader._format_skills_for_prompt([], include_examples=False)

        assert formatted == ""

    def test_format_limits_skills(self, skill_loader_with_skills):
        """Test that formatting limits number of skills."""
        loader = skill_loader_with_skills

        # Create many skills
        skills = []
        for i in range(20):
            skills.append({
                'name': f'skill_{i}',
                'description': f'Description {i}',
                'examples': []
            })

        formatted = loader._format_skills_for_prompt(skills)

        # Should limit to 15 skills
        skill_count = formatted.count('## skill_')
        assert skill_count <= 15


# ============================================================================
# Search Tests
# ============================================================================

class TestSkillSearch:
    """Tests for skill searching."""

    def test_search_skills_by_name(self, skill_loader_with_skills):
        """Test searching skills by name."""
        loader = skill_loader_with_skills
        results = loader.search_skills('pandas')

        assert len(results) >= 1
        assert any(s['name'] == 'pandas' for s in results if s)

    def test_search_skills_partial_match(self, skill_loader_with_skills):
        """Test searching with partial name."""
        loader = skill_loader_with_skills
        results = loader.search_skills('scan')

        # Should find scanpy
        assert any('scanpy' in str(s) for s in results if s)

    def test_search_skills_no_results(self, skill_loader_with_skills):
        """Test search with no matches."""
        loader = skill_loader_with_skills
        results = loader.search_skills('xyznonexistent')

        assert results == []

    def test_search_skills_case_insensitive(self, skill_loader_with_skills):
        """Test case-insensitive search."""
        loader = skill_loader_with_skills
        results = loader.search_skills('PANDAS')

        assert len(results) >= 1


# ============================================================================
# Statistics Tests
# ============================================================================

class TestSkillStatistics:
    """Tests for skill statistics."""

    def test_get_statistics(self, skill_loader_with_skills):
        """Test getting skill statistics."""
        loader = skill_loader_with_skills
        stats = loader.get_statistics()

        assert 'total_skills' in stats
        assert 'skills_by_category' in stats
        assert 'predefined_bundles' in stats
        assert stats['total_skills'] >= 4  # Our mock skills

    def test_get_statistics_empty(self, empty_skill_loader):
        """Test statistics for empty skill loader."""
        loader = empty_skill_loader
        stats = loader.get_statistics()

        assert stats['total_skills'] == 0


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestSkillLoaderEdgeCases:
    """Tests for edge cases and error handling."""

    def test_find_skills_directory_fallback(self):
        """Test that _find_skills_directory handles missing dirs."""
        loader = SkillLoader(auto_discover=False)
        result = loader._find_skills_directory()

        # Should return None or valid path (depends on environment)
        assert result is None or Path(result).exists()

    def test_load_skill_with_encoding_error(self, temp_dir):
        """Test loading skill with problematic encoding."""
        skills_dir = temp_dir / "encoding-skills"
        skills_dir.mkdir()

        # Create file with special characters
        _write_skill(skills_dir, "special", "# Special\n\nContent with émojis: 🎉")

        loader = SkillLoader(skills_dir=str(skills_dir), auto_discover=True)
        skill = loader.load_skill('special')

        assert skill is not None
        assert '🎉' in skill['content']

    def test_skills_dir_not_exists(self, temp_dir):
        """Test with non-existent skills directory."""
        nonexistent = temp_dir / "does_not_exist"
        loader = SkillLoader(skills_dir=str(nonexistent), auto_discover=True)

        assert loader.skills_cache == {}

    def test_concurrent_skill_loading(self, skill_loader_with_skills):
        """Test concurrent skill loading doesn't cause issues."""
        import threading

        loader = skill_loader_with_skills
        results = []

        def load_skill(name):
            skill = loader.load_skill(name)
            results.append(skill)

        threads = [
            threading.Thread(target=load_skill, args=('pandas',)),
            threading.Thread(target=load_skill, args=('scipy',)),
            threading.Thread(target=load_skill, args=('scanpy',))
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 3
        assert all(r is not None for r in results)
        assert sorted(r['name'] for r in results) == ['pandas', 'scanpy', 'scipy']
