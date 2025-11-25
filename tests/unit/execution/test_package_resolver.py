"""
Tests for package dependency resolution.

Tests package extraction, resolution, and installation logic
without requiring actual Docker containers.
"""

import pytest
from unittest.mock import Mock, MagicMock
from kosmos.execution.package_resolver import (
    PackageResolver,
    PackageRequirement,
    extract_imports_from_code,
    resolve_package_name,
    is_stdlib_module,
    IMPORT_TO_PIP,
    STDLIB_MODULES,
)


class TestImportExtraction:
    """Tests for extracting imports from code."""

    def test_extract_simple_imports(self):
        """Test extraction of simple import statements."""
        code = """
import pandas as pd
import numpy as np
"""
        resolver = PackageResolver(Mock(), "test-container")
        imports = resolver.extract_imports(code)

        assert "pandas" in imports
        assert "numpy" in imports

    def test_extract_from_imports(self):
        """Test extraction of from...import statements."""
        code = """
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind
"""
        resolver = PackageResolver(Mock(), "test-container")
        imports = resolver.extract_imports(code)

        assert "sklearn" in imports
        assert "scipy" in imports

    def test_extract_nested_imports(self):
        """Test extraction handles nested module paths."""
        code = """
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
"""
        resolver = PackageResolver(Mock(), "test-container")
        imports = resolver.extract_imports(code)

        # Should extract top-level modules only
        assert "sklearn" in imports
        assert "matplotlib" in imports

    def test_excludes_stdlib_modules(self):
        """Test that stdlib modules are excluded."""
        code = """
import os
import sys
import json
import re
import pandas
"""
        resolver = PackageResolver(Mock(), "test-container")
        imports = resolver.extract_imports(code)

        assert "os" not in imports
        assert "sys" not in imports
        assert "json" not in imports
        assert "re" not in imports
        assert "pandas" in imports

    def test_handles_syntax_errors(self):
        """Test fallback regex for invalid Python."""
        code = """
import pandas
import numpy
x = [1, 2, 3  # missing bracket
"""
        resolver = PackageResolver(Mock(), "test-container")
        imports = resolver.extract_imports(code)

        # Should still extract imports via regex
        assert "pandas" in imports
        assert "numpy" in imports

    def test_handles_empty_code(self):
        """Test handling of empty code."""
        resolver = PackageResolver(Mock(), "test-container")
        imports = resolver.extract_imports("")

        assert len(imports) == 0

    def test_handles_code_without_imports(self):
        """Test code without any imports."""
        code = """
x = 1 + 2
y = x * 3
print(y)
"""
        resolver = PackageResolver(Mock(), "test-container")
        imports = resolver.extract_imports(code)

        assert len(imports) == 0

    def test_extract_imports_convenience_function(self):
        """Test the convenience function for import extraction."""
        code = "import pandas\nimport numpy"
        imports = extract_imports_from_code(code)

        assert "pandas" in imports
        assert "numpy" in imports


class TestPackageResolution:
    """Tests for resolving import names to pip packages."""

    def test_resolve_simple_packages(self):
        """Test resolution of packages with same import/pip name."""
        resolver = PackageResolver(Mock(), "test-container")
        imports = {"pandas", "numpy", "scipy"}
        packages = resolver.resolve_packages(imports)

        names = {p.name for p in packages}
        assert "pandas" in names
        assert "numpy" in names
        assert "scipy" in names

    def test_resolve_different_pip_names(self):
        """Test resolution where import name differs from pip name."""
        resolver = PackageResolver(Mock(), "test-container")
        imports = {"sklearn", "PIL", "cv2", "yaml"}
        packages = resolver.resolve_packages(imports)

        names = {p.name for p in packages}
        assert "scikit-learn" in names
        assert "Pillow" in names
        assert "opencv-python" in names
        assert "pyyaml" in names

    def test_resolve_preserves_import_name(self):
        """Test that import name is preserved in requirement."""
        resolver = PackageResolver(Mock(), "test-container")
        imports = {"sklearn"}
        packages = resolver.resolve_packages(imports)

        assert len(packages) == 1
        assert packages[0].name == "scikit-learn"
        assert packages[0].import_name == "sklearn"

    def test_resolve_package_name_function(self):
        """Test convenience function for package name resolution."""
        assert resolve_package_name("sklearn") == "scikit-learn"
        assert resolve_package_name("PIL") == "Pillow"
        assert resolve_package_name("pandas") == "pandas"
        assert resolve_package_name("unknown_pkg") == "unknown_pkg"

    def test_no_duplicate_packages(self):
        """Test that duplicate imports don't create duplicate packages."""
        resolver = PackageResolver(Mock(), "test-container")
        # Both would resolve to same pip package
        imports = {"pandas", "pandas"}
        packages = resolver.resolve_packages(imports)

        assert len(packages) == 1


class TestStdlibDetection:
    """Tests for stdlib module detection."""

    def test_common_stdlib_modules(self):
        """Test detection of common stdlib modules."""
        assert is_stdlib_module("os")
        assert is_stdlib_module("sys")
        assert is_stdlib_module("json")
        assert is_stdlib_module("re")
        assert is_stdlib_module("pathlib")
        assert is_stdlib_module("typing")
        assert is_stdlib_module("datetime")
        assert is_stdlib_module("collections")

    def test_non_stdlib_modules(self):
        """Test that external packages are not detected as stdlib."""
        assert not is_stdlib_module("pandas")
        assert not is_stdlib_module("numpy")
        assert not is_stdlib_module("sklearn")
        assert not is_stdlib_module("scipy")

    def test_stdlib_modules_set_is_comprehensive(self):
        """Test that STDLIB_MODULES contains expected modules."""
        expected_modules = [
            "os", "sys", "json", "re", "math", "random",
            "datetime", "time", "pathlib", "typing",
            "collections", "itertools", "functools",
            "subprocess", "multiprocessing", "threading",
            "asyncio", "concurrent", "socket", "http"
        ]

        for module in expected_modules:
            assert module in STDLIB_MODULES, f"{module} should be in STDLIB_MODULES"


class TestPackageRequirement:
    """Tests for PackageRequirement dataclass."""

    def test_basic_requirement(self):
        """Test basic requirement creation."""
        req = PackageRequirement(name="pandas")

        assert req.name == "pandas"
        assert req.version is None
        assert req.pip_spec == "pandas"

    def test_requirement_with_version(self):
        """Test requirement with version."""
        req = PackageRequirement(name="pandas", version="2.0.0")

        assert req.pip_spec == "pandas==2.0.0"

    def test_requirement_equality(self):
        """Test requirement equality."""
        req1 = PackageRequirement(name="pandas", version="2.0.0")
        req2 = PackageRequirement(name="pandas", version="2.0.0")
        req3 = PackageRequirement(name="pandas", version="2.1.0")

        assert req1 == req2
        assert req1 != req3

    def test_requirement_hashing(self):
        """Test that requirements can be hashed (for sets)."""
        req1 = PackageRequirement(name="pandas", version="2.0.0")
        req2 = PackageRequirement(name="pandas", version="2.0.0")

        # Should be able to add to set
        requirements = {req1, req2}
        assert len(requirements) == 1


class TestImportToPipMapping:
    """Tests for the IMPORT_TO_PIP mapping."""

    def test_common_mappings(self):
        """Test common import to pip name mappings."""
        mappings = [
            ("cv2", "opencv-python"),
            ("PIL", "Pillow"),
            ("sklearn", "scikit-learn"),
            ("skimage", "scikit-image"),
            ("yaml", "pyyaml"),
            ("Bio", "biopython"),
        ]

        for import_name, pip_name in mappings:
            assert IMPORT_TO_PIP.get(import_name) == pip_name, \
                f"Expected {import_name} -> {pip_name}"

    def test_deep_learning_mappings(self):
        """Test deep learning framework mappings."""
        assert IMPORT_TO_PIP.get("torch") == "torch"
        assert IMPORT_TO_PIP.get("tensorflow") == "tensorflow"
        assert IMPORT_TO_PIP.get("keras") == "keras"

    def test_bioinformatics_mappings(self):
        """Test bioinformatics package mappings."""
        assert IMPORT_TO_PIP.get("Bio") == "biopython"
        assert IMPORT_TO_PIP.get("scanpy") == "scanpy"
        assert IMPORT_TO_PIP.get("anndata") == "anndata"


class TestRequirementsTxtGeneration:
    """Tests for requirements.txt generation."""

    def test_generate_requirements_txt(self):
        """Test generating requirements.txt content."""
        code = """
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
"""
        resolver = PackageResolver(Mock(), "test-container")
        requirements = resolver.get_requirements_txt(code)

        assert "pandas" in requirements
        assert "numpy" in requirements
        assert "scikit-learn" in requirements

    def test_empty_code_requirements(self):
        """Test requirements for code without imports."""
        code = "x = 1 + 2"
        resolver = PackageResolver(Mock(), "test-container")
        requirements = resolver.get_requirements_txt(code)

        # Should only have header comment
        assert "Auto-generated" in requirements


class TestPackageResolverCaching:
    """Tests for package resolver caching."""

    def test_installed_cache(self):
        """Test that installed packages are cached."""
        resolver = PackageResolver(Mock(), "test-container")

        # Manually add to cache
        resolver._installed_cache.add("pandas")

        # Should be in cache
        assert "pandas" in resolver._installed_cache

    def test_failed_cache(self):
        """Test that failed packages are cached."""
        resolver = PackageResolver(Mock(), "test-container")

        # Manually add to failed cache
        resolver._failed_cache.add("broken-package")

        # Should be in failed cache
        assert "broken-package" in resolver._failed_cache

    def test_clear_cache(self):
        """Test cache clearing."""
        resolver = PackageResolver(Mock(), "test-container")

        resolver._installed_cache.add("pandas")
        resolver._failed_cache.add("broken")

        resolver.clear_cache()

        assert len(resolver._installed_cache) == 0
        assert len(resolver._failed_cache) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
