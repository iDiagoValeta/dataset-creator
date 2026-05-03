"""JSONL/JSON I/O, dataset splitting, artifact cleanup, and reproducibility metadata."""

import json
import platform
import random
import subprocess
from collections import defaultdict
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
    """Split rows into train/val/test using deterministic topic-aware shuffle."""
    train_ratio, val_ratio, _test_ratio = split
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    target_val = val_end - train_end
    target_test = total - val_end

    if total and any("topic_id" in row for row in shuffled) and (target_val or target_test):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in shuffled:
            groups[str(row.get("topic_id", "__missing__"))].append(row)

        rng = random.Random(seed)
        group_keys = list(groups)
        rng.shuffle(group_keys)
        train_rows: list[dict[str, Any]] = []
        val_rows: list[dict[str, Any]] = []
        test_rows: list[dict[str, Any]] = []

        for key in group_keys:
            group = groups[key]
            rng.shuffle(group)
            remaining = list(group)
            if target_val and len(remaining) >= 3 and len(val_rows) < target_val:
                val_rows.append(remaining.pop())
            if target_test and len(remaining) >= 2 and len(test_rows) < target_test:
                test_rows.append(remaining.pop())
            train_rows.extend(remaining)

        while len(val_rows) < target_val and train_rows:
            val_rows.append(train_rows.pop())
        while len(test_rows) < target_test and train_rows:
            test_rows.append(train_rows.pop())

        rng.shuffle(train_rows)
        rng.shuffle(val_rows)
        rng.shuffle(test_rows)
        return train_rows, val_rows, test_rows

    train_rows = shuffled[:train_end]
    val_rows = shuffled[train_end:val_end]
    test_rows = shuffled[val_end:]
    return train_rows, val_rows, test_rows


def build_dataset_audit(
    accepted_rows: Sequence[dict[str, Any]],
    rejected_rows: Sequence[dict[str, Any]],
    train_rows: Sequence[dict[str, Any]],
    val_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    expected_topic_ids: Sequence[str] | None = None,
    min_items_per_topic: int = 2,
) -> dict[str, Any]:
    """Build lightweight quality and coverage audit metrics for metadata."""
    # Use composite keys "{document}::{topic_id}" so that the same topic_id in different
    # documents does not collapse into a single counter entry.
    def _composite_key(row: dict[str, Any]) -> str:
        doc = str(row.get("document", "__missing__"))
        tid = str(row.get("topic_id", "__missing__"))
        return f"{doc}::{tid}"

    accepted_by_topic: dict[str, int] = {}
    rejected_by_topic: dict[str, int] = {}
    split_by_topic = {"train": {}, "val": {}, "test": {}}

    for row in accepted_rows:
        key = _composite_key(row)
        accepted_by_topic[key] = accepted_by_topic.get(key, 0) + 1
    for row in rejected_rows:
        key = _composite_key(row)
        rejected_by_topic[key] = rejected_by_topic.get(key, 0) + 1
    for split_name, rows_for_split in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        for row in rows_for_split:
            key = _composite_key(row)
            bucket = split_by_topic[split_name]
            bucket[key] = bucket.get(key, 0) + 1

    expected_topic_keys = {
        str(topic_id)
        for topic_id in (expected_topic_ids or [])
        if "::" in str(topic_id)
    }
    # Bare expected_topic_ids are not composite-aware, so only explicit
    # "{document}::{topic_id}" values are added to the key space.
    all_topics = sorted(set(accepted_by_topic) | set(rejected_by_topic) | expected_topic_keys)
    topics_without_accepted = [key for key in all_topics if accepted_by_topic.get(key, 0) == 0]
    low_accepted_topics = [
        key
        for key in all_topics
        if 0 < accepted_by_topic.get(key, 0) < min_items_per_topic
    ]

    coverage_total = len(all_topics)
    topics_with_accepted = coverage_total - len(topics_without_accepted)
    coverage_ratio = (
        round(topics_with_accepted / coverage_total, 4) if coverage_total else 0.0
    )

    multi_doc_in_splits = {}
    for split_name, rows_for_split in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        documents = {str(row.get("document", "")) for row in rows_for_split if row.get("document")}
        multi_doc_in_splits[split_name] = len(documents) > 1

    warnings: list[str] = []
    if topics_without_accepted:
        warnings.append(f"topics_without_accepted: {', '.join(topics_without_accepted)}")
    if low_accepted_topics:
        warnings.append(f"topics_below_{min_items_per_topic}_items: {', '.join(low_accepted_topics)}")
    for split_name, rows_for_split in (("val", val_rows), ("test", test_rows)):
        if rows_for_split and len(split_by_topic[split_name]) == 1 and len(all_topics) > 1:
            warnings.append(f"{split_name}_split_has_single_topic")

    pipeline_success = (
        coverage_total > 0
        and len(topics_without_accepted) <= max(1, coverage_total // 5)
    )

    return {
        "accepted_by_topic": accepted_by_topic,
        "rejected_by_topic": rejected_by_topic,
        "split_by_topic": split_by_topic,
        "topics_without_accepted": topics_without_accepted,
        "low_accepted_topics": low_accepted_topics,
        "coverage_ratio": coverage_ratio,
        "multi_doc_in_splits": multi_doc_in_splits,
        "pipeline_success": pipeline_success,
        "warnings": warnings,
    }


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
