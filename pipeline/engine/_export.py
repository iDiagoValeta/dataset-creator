"""JSONL/JSON I/O, dataset splitting, artifact cleanup, and reproducibility metadata."""

import json
import platform
import random
import subprocess
from collections.abc import Sequence
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from engine._config import BASE_DIR
from engine._text import deduplicate_preserve_order


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write rows as JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_rows(
    rows: Sequence[dict[str, Any]],
    split: tuple[float, float, float],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into train/val/test using deterministic shuffle."""
    train_ratio, val_ratio, test_ratio = split
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_rows = shuffled[:train_end]
    val_rows = shuffled[train_end:val_end]
    test_rows = shuffled[val_end:]
    return train_rows, val_rows, test_rows


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Write metadata as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _assert_safe_cleanup_root(path: Path) -> Path:
    """Resolve and validate cleanup roots before deleting generated artifacts."""
    resolved = path.resolve()
    allowed_roots = [
        (BASE_DIR / "output").resolve(),
        (BASE_DIR / "run_logs").resolve(),
    ]
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        allowed = ", ".join(str(root) for root in allowed_roots)
        raise RuntimeError(f"Ruta de limpieza no permitida: {resolved}. Permitidas: {allowed}")
    return resolved


def _remove_empty_dirs(root: Path) -> int:
    """Remove empty directories below root, deepest first."""
    removed = 0
    if not root.exists():
        return removed
    dirs = [p for p in root.rglob("*") if p.is_dir()]
    for directory in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
        try:
            directory.rmdir()
            removed += 1
        except OSError:
            continue
    return removed


def clean_generated_artifacts(output_path: Path, debug_dir: Path, dry_run: bool = False) -> dict[str, int]:
    """Clean generated JSON/JSONL artifacts from output and run-log directories."""
    roots = deduplicate_preserve_order([str(output_path.parent), str(debug_dir)])
    files: list[Path] = []
    for root_text in roots:
        root = _assert_safe_cleanup_root(Path(root_text))
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.name == ".gitkeep":
                continue
            if path.suffix.lower() in {".json", ".jsonl"}:
                files.append(path)

    files = sorted(set(files))
    if dry_run:
        for path in files:
            print(f"[DRY-RUN] borraria {path}")
        return {"files": len(files), "dirs": 0}

    for path in files:
        path.unlink()

    removed_dirs = 0
    for root_text in roots:
        root = _assert_safe_cleanup_root(Path(root_text))
        removed_dirs += _remove_empty_dirs(root)
    return {"files": len(files), "dirs": removed_dirs}


def _package_version(name: str) -> str | None:
    """Return installed version for a package, or None if not available."""
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _current_git_commit() -> str | None:
    """Return short git commit for the repo, or None if git is unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(BASE_DIR),
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        return out.decode().strip() or None
    except Exception:
        return None


def build_reproducibility_info() -> dict[str, Any]:
    """Collect runtime/version info for metadata reproducibility."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "ollama": _package_version("ollama"),
            "pypdf": _package_version("pypdf"),
            "pymupdf4llm": _package_version("pymupdf4llm"),
        },
        "git_commit": _current_git_commit(),
    }
