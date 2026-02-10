"""Project path helpers for scripts and modules."""

from pathlib import Path


def default_paths() -> dict[str, Path]:
    """Return commonly used repository paths.

    Returns
    -------
    dict[str, Path]
        Mapping of logical names to absolute paths rooted at repository base.
    """
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "repo_root": repo_root,
        "data_raw": repo_root / "data" / "raw",
        "data_processed": repo_root / "data" / "processed",
        "data_sample": repo_root / "data" / "sample",
        "results_tables": repo_root / "results" / "tables",
        "results_figures": repo_root / "results" / "figures",
        "configs": repo_root / "configs",
    }
