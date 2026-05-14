"""Dependency checker.

Reads the project's pyproject.toml and determines which configured
GitHub library repositories should actually be indexed, based on whether
their package name is declared as a dependency of the project.

Only libraries that are real dependencies make sense to index for reuse
detection: if the package is not in pyproject.toml, you cannot import
from it, so any reuse suggestion would be unusable.
"""

import logging
import re
from pathlib import Path

from .config import GitHubRepoConfig

logger = logging.getLogger(__name__)


# Try Python 3.11+ stdlib tomllib, fall back to a regex parser if unavailable.
try:
    import tomllib  # type: ignore[import-not-found]
    _HAS_TOMLLIB = True
except ImportError:  # pragma: no cover - only on Python < 3.11
    _HAS_TOMLLIB = False


class DependencyChecker:
    """Checks if a package is declared as a dependency in pyproject.toml."""

    def __init__(self, pyproject_path: Path):
        self.pyproject_path = pyproject_path
        self._dependencies: set[str] | None = None

    def get_declared_dependencies(self) -> set[str]:
        """Return the set of package names declared in pyproject.toml.

        Covers PEP 621 [project] dependencies and Poetry
        [tool.poetry.dependencies] / [tool.poetry.dev-dependencies].
        Result is cached after the first call.
        """
        if self._dependencies is not None:
            return self._dependencies

        if not self.pyproject_path.exists():
            logger.warning(
                "pyproject.toml not found at %s; treating dependency set as empty",
                self.pyproject_path,
            )
            self._dependencies = set()
            return self._dependencies

        try:
            data = self._load_toml(self.pyproject_path)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", self.pyproject_path, e)
            self._dependencies = set()
            return self._dependencies

        deps: set[str] = set()

        # Poetry-style dependencies
        poetry = data.get("tool", {}).get("poetry", {})
        for section in ("dependencies", "dev-dependencies"):
            for name in (poetry.get(section) or {}).keys():
                if name.lower() == "python":
                    continue
                deps.add(self._normalize_name(name))

        # Poetry 1.2+ groups
        groups = poetry.get("group", {})
        if isinstance(groups, dict):
            for group in groups.values():
                if isinstance(group, dict):
                    for name in (group.get("dependencies") or {}).keys():
                        if name.lower() == "python":
                            continue
                        deps.add(self._normalize_name(name))

        # PEP 621 dependencies
        project = data.get("project", {})
        for spec in project.get("dependencies", []) or []:
            name = self._extract_name_from_pep508(spec)
            if name:
                deps.add(self._normalize_name(name))

        # PEP 621 optional dependencies
        for group_specs in (project.get("optional-dependencies") or {}).values():
            for spec in group_specs:
                name = self._extract_name_from_pep508(spec)
                if name:
                    deps.add(self._normalize_name(name))

        self._dependencies = deps
        return deps

    def is_dependency(self, package_name: str) -> bool:
        """Return True if the given package is declared in pyproject.toml."""
        return self._normalize_name(package_name) in self.get_declared_dependencies()

    def filter_indexable_repos(
        self, repos: list[GitHubRepoConfig]
    ) -> list[GitHubRepoConfig]:
        """Return only repos whose package name is in pyproject.toml.

        Only libraries that are actual project dependencies should be
        indexed for reuse detection. For example, if pyproject.toml has
        'pdt-common' as a dependency, the repo 'pdt-common-lib' will be
        included because stripping the '-lib' suffix yields a match.

        Matching order:
        1. Configured package_name directly
        2. package_name with '-lib' suffix stripped
        3. Repo name derived from URL
        4. Repo name with '-lib' suffix stripped
        """
        indexable: list[GitHubRepoConfig] = []
        for repo in repos:
            if self._matches_dependency(repo):
                logger.info(
                    "Repo %s (package %s) is a project dependency; will index",
                    repo.url,
                    repo.package_name,
                )
                indexable.append(repo)
            else:
                logger.info(
                    "Skipping repo %s: package %s is not in %s dependencies",
                    repo.url,
                    repo.package_name,
                    self.pyproject_path,
                )
        return indexable

    def _matches_dependency(self, repo: GitHubRepoConfig) -> bool:
        """Check if a repo matches any declared dependency.

        Tries the explicit package_name first. If that fails, derives
        candidate names from the repo URL (with and without '-lib' suffix)
        and checks those against pyproject.toml dependencies.

        The '-lib' suffix convention: a repo named 'pdt-common-lib' maps
        to the PyPI package 'pdt-common'. So if 'pdt-common' is in
        pyproject.toml, the 'pdt-common-lib' repo should be indexed.
        """
        # 1. Direct match on configured package_name
        if self.is_dependency(repo.package_name):
            return True

        # 2. Strip '-lib' suffix from package_name (pdt-common-lib -> pdt-common)
        if repo.package_name.endswith("-lib"):
            base_pkg = repo.package_name[: -len("-lib")]
            if self.is_dependency(base_pkg):
                return True

        # 3. Derive name from repo URL (last path segment)
        repo_name = repo.url.rstrip("/").split("/")[-1]
        if self.is_dependency(repo_name):
            return True

        # 4. Strip '-lib' suffix from repo name and check
        if repo_name.endswith("-lib"):
            base_name = repo_name[: -len("-lib")]
            if self.is_dependency(base_name):
                return True

        return False

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize a package name per PEP 503 (lowercase, hyphens for separators)."""
        return re.sub(r"[-_.]+", "-", name).lower()

    @staticmethod
    def _extract_name_from_pep508(spec: str) -> str | None:
        """Extract the package name from a PEP 508 requirement spec."""
        match = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)", spec)
        return match.group(1) if match else None

    @staticmethod
    def _load_toml(path: Path) -> dict:
        """Load a TOML file using stdlib tomllib if available, else a regex parser."""
        if _HAS_TOMLLIB:
            with open(path, "rb") as f:
                return tomllib.load(f)
        return _parse_toml_minimal(path.read_text(encoding="utf-8"))


def _parse_toml_minimal(text: str) -> dict:
    """Very small TOML reader for environments without tomllib.

    Only extracts the structure needed by DependencyChecker: section headers
    and the keys directly under them. Values are kept as raw strings; we
    only care about which keys exist under [tool.poetry.dependencies] etc.
    """
    result: dict = {}
    current_section: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("[") and line.endswith("]"):
            section_name = line[1:-1].strip()
            current_section = [p.strip() for p in section_name.split(".")]
            # Walk into nested dicts, creating as we go
            cursor = result
            for part in current_section:
                cursor = cursor.setdefault(part, {})
            continue

        if "=" in line and current_section:
            key, _, value = line.partition("=")
            key = key.strip().strip('"').strip("'")
            cursor = result
            for part in current_section:
                cursor = cursor.setdefault(part, {})
            cursor[key] = value.strip()

    return result
