"""
Automatic package dependency resolution and installation.

Features:
- Parse imports from generated code
- Resolve package names (import name -> pip name)
- Install missing packages in sandbox
- Handle version conflicts
- Cache installed packages

This module ensures generated scientific code has all required
dependencies available in the execution environment.
"""

import re
import ast
from dataclasses import dataclass
from typing import List, Set, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# Mapping of import names to pip package names
# Some packages have different import names than their pip names
IMPORT_TO_PIP: Dict[str, str] = {
    # Computer Vision / Image Processing
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "skimage": "scikit-image",

    # Machine Learning
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "keras": "keras",

    # Deep Learning
    "torch": "torch",
    "torchvision": "torchvision",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",

    # Data Science
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "statsmodels": "statsmodels",

    # Visualization
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "plotly": "plotly",
    "bokeh": "bokeh",
    "altair": "altair",

    # Bioinformatics
    "Bio": "biopython",
    "scanpy": "scanpy",
    "anndata": "anndata",
    "pysam": "pysam",

    # Chemistry
    "rdkit": "rdkit-pypi",

    # NLP
    "nltk": "nltk",
    "spacy": "spacy",
    "transformers": "transformers",
    "gensim": "gensim",

    # Utilities
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "lxml": "lxml",
    "requests": "requests",
    "httpx": "httpx",
    "aiohttp": "aiohttp",

    # Data formats
    "openpyxl": "openpyxl",
    "xlrd": "xlrd",
    "pyarrow": "pyarrow",
    "h5py": "h5py",
    "netCDF4": "netcdf4",

    # Scientific
    "sympy": "sympy",
    "networkx": "networkx",
    "igraph": "python-igraph",

    # Statistics
    "pingouin": "pingouin",
    "lifelines": "lifelines",

    # Jupyter
    "nbformat": "nbformat",
    "nbconvert": "nbconvert",

    # Feature engineering
    "shap": "shap",
    "lime": "lime",

    # Pathway analysis
    "gseapy": "gseapy",
}


# Standard library modules that don't need installation
STDLIB_MODULES: Set[str] = {
    # Built-in modules
    "abc", "aifc", "argparse", "array", "ast", "asynchat", "asyncio",
    "asyncore", "atexit", "audioop", "base64", "bdb", "binascii",
    "binhex", "bisect", "builtins", "bz2", "calendar", "cgi", "cgitb",
    "chunk", "cmath", "cmd", "code", "codecs", "codeop", "collections",
    "colorsys", "compileall", "concurrent", "configparser", "contextlib",
    "contextvars", "copy", "copyreg", "cProfile", "crypt", "csv",
    "ctypes", "curses", "dataclasses", "datetime", "dbm", "decimal",
    "difflib", "dis", "distutils", "doctest", "email", "encodings",
    "enum", "errno", "faulthandler", "fcntl", "filecmp", "fileinput",
    "fnmatch", "fractions", "ftplib", "functools", "gc", "getopt",
    "getpass", "gettext", "glob", "graphlib", "grp", "gzip", "hashlib",
    "heapq", "hmac", "html", "http", "idlelib", "imaplib", "imghdr",
    "imp", "importlib", "inspect", "io", "ipaddress", "itertools",
    "json", "keyword", "lib2to3", "linecache", "locale", "logging",
    "lzma", "mailbox", "mailcap", "marshal", "math", "mimetypes",
    "mmap", "modulefinder", "multiprocessing", "netrc", "nis",
    "nntplib", "numbers", "operator", "optparse", "os", "ossaudiodev",
    "parser", "pathlib", "pdb", "pickle", "pickletools", "pipes",
    "pkgutil", "platform", "plistlib", "poplib", "posix", "posixpath",
    "pprint", "profile", "pstats", "pty", "pwd", "py_compile",
    "pyclbr", "pydoc", "queue", "quopri", "random", "re", "readline",
    "reprlib", "resource", "rlcompleter", "runpy", "sched", "secrets",
    "select", "selectors", "shelve", "shlex", "shutil", "signal",
    "site", "smtpd", "smtplib", "sndhdr", "socket", "socketserver",
    "spwd", "sqlite3", "ssl", "stat", "statistics", "string",
    "stringprep", "struct", "subprocess", "sunau", "symbol", "symtable",
    "sys", "sysconfig", "syslog", "tabnanny", "tarfile", "telnetlib",
    "tempfile", "termios", "test", "textwrap", "threading", "time",
    "timeit", "tkinter", "token", "tokenize", "trace", "traceback",
    "tracemalloc", "tty", "turtle", "turtledemo", "types", "typing",
    "unicodedata", "unittest", "urllib", "uu", "uuid", "venv",
    "warnings", "wave", "weakref", "webbrowser", "winreg", "winsound",
    "wsgiref", "xdrlib", "xml", "xmlrpc", "zipapp", "zipfile",
    "zipimport", "zlib", "_thread",
}


# Packages typically pre-installed in scientific environments
PREINSTALLED_PACKAGES: Set[str] = {
    "numpy", "pandas", "scipy", "matplotlib", "seaborn",
    "scikit-learn", "statsmodels", "pydantic",
}


@dataclass
class PackageRequirement:
    """A required package with optional version."""
    name: str
    version: Optional[str] = None
    import_name: str = ""
    optional: bool = False

    def __hash__(self):
        return hash((self.name, self.version))

    def __eq__(self, other):
        if not isinstance(other, PackageRequirement):
            return False
        return self.name == other.name and self.version == other.version

    @property
    def pip_spec(self) -> str:
        """Get pip install specification."""
        if self.version:
            return f"{self.name}=={self.version}"
        return self.name


class PackageResolver:
    """
    Resolves and installs package dependencies.

    Usage:
        resolver = PackageResolver(docker_client, container_id)

        # Extract and install dependencies
        success = await resolver.ensure_dependencies(code)

        # Or step by step
        imports = resolver.extract_imports(code)
        packages = resolver.resolve_packages(imports)
        results = await resolver.install_packages(packages)
    """

    def __init__(self, docker_client, container_id: str):
        """
        Initialize package resolver.

        Args:
            docker_client: Docker client instance
            container_id: Container to install packages in
        """
        self.docker_client = docker_client
        self.container_id = container_id
        self._installed_cache: Set[str] = set()
        self._failed_cache: Set[str] = set()

    def extract_imports(self, code: str) -> Set[str]:
        """
        Extract import statements from code.

        Args:
            code: Python source code

        Returns:
            Set of top-level module names
        """
        imports = set()

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Get top-level module
                        module = alias.name.split('.')[0]
                        imports.add(module)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Get top-level module
                        module = node.module.split('.')[0]
                        imports.add(module)
        except SyntaxError as e:
            logger.warning(f"Syntax error parsing code for imports: {e}")
            # Fallback to regex for invalid Python
            imports = self._extract_imports_regex(code)

        # Filter out stdlib modules
        external_imports = imports - STDLIB_MODULES

        logger.debug(f"Extracted imports: {external_imports}")
        return external_imports

    def _extract_imports_regex(self, code: str) -> Set[str]:
        """Fallback regex-based import extraction."""
        imports = set()

        # Match import statements
        import_pattern = r'^(?:from|import)\s+([\w\.]+)'
        for match in re.finditer(import_pattern, code, re.MULTILINE):
            module = match.group(1).split('.')[0]
            imports.add(module)

        return imports

    def resolve_packages(self, imports: Set[str]) -> List[PackageRequirement]:
        """
        Resolve import names to pip packages.

        Args:
            imports: Set of import names

        Returns:
            List of PackageRequirements
        """
        packages = []
        seen = set()

        for imp in imports:
            # Get pip package name
            pip_name = IMPORT_TO_PIP.get(imp, imp)

            if pip_name in seen:
                continue
            seen.add(pip_name)

            packages.append(PackageRequirement(
                name=pip_name,
                import_name=imp,
                optional=pip_name not in PREINSTALLED_PACKAGES
            ))

        logger.debug(f"Resolved packages: {[p.name for p in packages]}")
        return packages

    async def check_installed(self, package: PackageRequirement) -> bool:
        """
        Check if a package is installed.

        Args:
            package: Package to check

        Returns:
            True if installed
        """
        if package.name in self._installed_cache:
            return True

        if package.name in self._failed_cache:
            return False

        container = self.docker_client.containers.get(self.container_id)

        # Try to import the package
        import_name = package.import_name or package.name
        result = container.exec_run(
            ["python3", "-c", f"import {import_name}"],
            workdir="/workspace"
        )

        installed = result.exit_code == 0
        if installed:
            self._installed_cache.add(package.name)

        return installed

    async def install_packages(
        self,
        packages: List[PackageRequirement],
        skip_installed: bool = True
    ) -> Dict[str, bool]:
        """
        Install packages in container.

        Args:
            packages: List of packages to install
            skip_installed: Skip already installed packages

        Returns:
            Dict mapping package name to success status
        """
        container = self.docker_client.containers.get(self.container_id)
        results = {}

        for pkg in packages:
            # Skip if already installed
            if skip_installed:
                if pkg.name in self._installed_cache:
                    results[pkg.name] = True
                    continue

                if await self.check_installed(pkg):
                    results[pkg.name] = True
                    continue

            # Skip known failures
            if pkg.name in self._failed_cache:
                results[pkg.name] = False
                continue

            # Try to install
            cmd = ["pip", "install", "--quiet", "--no-cache-dir", pkg.pip_spec]

            logger.info(f"Installing package: {pkg.pip_spec}")
            result = container.exec_run(cmd, workdir="/workspace")

            success = result.exit_code == 0
            results[pkg.name] = success

            if success:
                self._installed_cache.add(pkg.name)
                logger.info(f"Installed {pkg.name}")
            else:
                self._failed_cache.add(pkg.name)
                output = result.output.decode('utf-8', errors='replace')
                logger.warning(f"Failed to install {pkg.name}: {output[:200]}")

        return results

    async def ensure_dependencies(self, code: str) -> Tuple[bool, List[str]]:
        """
        Extract, resolve, and install all dependencies for code.

        Args:
            code: Python source code

        Returns:
            Tuple of (success, list of failed packages)
        """
        imports = self.extract_imports(code)

        if not imports:
            return True, []

        packages = self.resolve_packages(imports)
        results = await self.install_packages(packages)

        failed = [name for name, success in results.items() if not success]

        return len(failed) == 0, failed

    def get_requirements_txt(self, code: str) -> str:
        """
        Generate requirements.txt content for code.

        Args:
            code: Python source code

        Returns:
            requirements.txt content
        """
        imports = self.extract_imports(code)
        packages = self.resolve_packages(imports)

        lines = [f"# Auto-generated requirements for code execution"]
        for pkg in sorted(packages, key=lambda p: p.name):
            lines.append(pkg.pip_spec)

        return "\n".join(lines)

    def clear_cache(self):
        """Clear installed/failed package caches."""
        self._installed_cache.clear()
        self._failed_cache.clear()


def extract_imports_from_code(code: str) -> Set[str]:
    """
    Convenience function to extract imports from code.

    Args:
        code: Python source code

    Returns:
        Set of import names
    """
    resolver = PackageResolver(None, "")
    return resolver.extract_imports(code)


def resolve_package_name(import_name: str) -> str:
    """
    Get pip package name for an import.

    Args:
        import_name: Python import name

    Returns:
        pip package name
    """
    return IMPORT_TO_PIP.get(import_name, import_name)


def is_stdlib_module(module_name: str) -> bool:
    """
    Check if a module is in the standard library.

    Args:
        module_name: Module name to check

    Returns:
        True if stdlib module
    """
    return module_name in STDLIB_MODULES
