"""Configuration for the reuse detection system."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class GitHubRepoConfig:
    """Configuration for a single GitHub library repository.

    Maps a GitHub URL to a package name. The package name is checked
    against pyproject.toml dependencies; the repo is only indexed if
    the package is an actual dependency of the project.
    """

    url: str
    package_name: str

    @classmethod
    def from_dict(cls, data: dict | str) -> "GitHubRepoConfig":
        """Build from either a dict {url, package_name} or a plain URL string.

        When given a plain URL string, the package name is inferred from
        the last path segment (e.g. .../pdt-common-lib -> pdt-common-lib).
        """
        if isinstance(data, str):
            url = data
            package_name = url.rstrip("/").split("/")[-1]
            return cls(url=url, package_name=package_name)

        return cls(
            url=data["url"],
            package_name=data["package_name"],
        )


@dataclass
class DetectionConfig:
    """Configuration for the reuse detection system.

    All thresholds, patterns, and resource limits are configurable.
    Defaults are provided for all fields so the system works out of the box.
    """

    github_repositories: list[GitHubRepoConfig] = field(default_factory=list)
    similarity_threshold: float = 0.75
    block_threshold: float = 0.85
    warn_threshold: float = 0.70
    min_block_lines: int = 3
    cache_dir: Path = field(default_factory=lambda: Path(".reuse-cache"))
    index_refresh_interval_hours: int = 24
    exclude_patterns: list[str] = field(
        default_factory=lambda: ["test_*", "*_test.py", "conftest.py", "__pycache__"]
    )
    include_patterns: list[str] = field(default_factory=lambda: ["*.py"])
    max_suggestions_per_block: int = 3
    top_k: int = 50
    llm_timeout_seconds: float = 10.0
    max_llm_calls_per_commit: int = 20
    pyproject_path: Path = field(default_factory=lambda: Path("pyproject.toml"))

    @classmethod
    def from_file(cls, path: Path) -> "DetectionConfig":
        """Load configuration from a YAML file, falling back to defaults."""
        if not path.exists():
            return cls()

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        # Convert cache_dir string to Path if present
        if "cache_dir" in data:
            data["cache_dir"] = Path(data["cache_dir"])

        # Convert pyproject_path string to Path if present
        if "pyproject_path" in data:
            data["pyproject_path"] = Path(data["pyproject_path"])

        # Convert github_repositories entries to GitHubRepoConfig objects.
        # Accepts either a list of strings (URLs) or a list of dicts
        # {url, package_name}.
        if "github_repositories" in data:
            raw = data["github_repositories"] or []
            data["github_repositories"] = [
                GitHubRepoConfig.from_dict(item) for item in raw
            ]

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_file(self, path: Path) -> None:
        """Save configuration to a YAML file."""
        data = {
            "github_repositories": [
                {"url": r.url, "package_name": r.package_name}
                for r in self.github_repositories
            ],
            "similarity_threshold": self.similarity_threshold,
            "block_threshold": self.block_threshold,
            "warn_threshold": self.warn_threshold,
            "min_block_lines": self.min_block_lines,
            "cache_dir": str(self.cache_dir),
            "index_refresh_interval_hours": self.index_refresh_interval_hours,
            "exclude_patterns": self.exclude_patterns,
            "include_patterns": self.include_patterns,
            "max_suggestions_per_block": self.max_suggestions_per_block,
            "top_k": self.top_k,
            "llm_timeout_seconds": self.llm_timeout_seconds,
            "max_llm_calls_per_commit": self.max_llm_calls_per_commit,
            "pyproject_path": str(self.pyproject_path),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
