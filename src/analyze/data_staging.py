"""Stage the seed dataset's files for the analyst before the agent starts.

Published Pennsieve datasets are public through Discover, with no account and
no AWS credentials (`pennsieve_ai_utils.tools.discover_files`). When the
analyst had to find that route itself, it sometimes spent hundreds of tool
calls on S3 and authenticated APIs, and sometimes never reached the data. So
before the agent starts, this module:

1. Writes `data/files_index.json`, every published file with its stored path,
   size and type, so the agent can choose files without probing an API.
2. Downloads a bounded sample into `data/`: small tables (CSV, TSV, XLSX)
   first, then the dominant data type, spread across folders.
3. Gives the prompt a section that names the staged files and the command for
   downloading more.

A dataset whose file index cannot be read raises `DataStagingError`, so the
stage stops instead of letting the agent guess. A single sample file that
fails is recorded and skipped, but if every sample download fails, that is
raised too.

Settings:
  VP_ANALYZE_STAGE_FILES   most data files to pre-download (default 12)
  VP_ANALYZE_STAGE_MAX_MB  total pre-download size in MiB (default 200)
  VP_ANALYZE_STAGE_DATA=0  write the index only, download nothing
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pennsieve_ai_utils.blackboard import Blackboard, Hypothesis
from pennsieve_ai_utils.tools.discover_files import DiscoverFiles

MAX_FILES = int(os.environ.get("VP_ANALYZE_STAGE_FILES", "12"))
MAX_BYTES = int(float(os.environ.get("VP_ANALYZE_STAGE_MAX_MB", "200")) * 1024 * 1024)
TABLE_SUFFIXES = (".csv", ".tsv", ".xlsx")
TABLE_FILE_CAP = 25 * 1024 * 1024
# File types that never help an analysis, skipped when sampling data files.
SKIP_SUFFIXES = {".md", ".txt", ".pdf", ".png", ".jpg", ".jpeg", ".html", ".json", ".xml", ".docx"}


class DataStagingError(RuntimeError):
    """The seed dataset's files could not be indexed or downloaded."""


@dataclass
class StagingResult:
    dataset_id: str
    version: str
    data_dir: Path
    index_path: Path
    total_files: int
    file_types: dict[str, int]
    staged: list[dict] = field(default_factory=list)   # {path, bytes}
    failed: list[str] = field(default_factory=list)


def stage(bb: Blackboard, hypothesis: Hypothesis, hyp_workdir: Path,
          files: DiscoverFiles | None = None) -> StagingResult:
    """Index the seed dataset and download a sample into `hyp_workdir/data`."""
    seed = next((d for d in bb.dataset_summaries if d.id == bb.subsystem), None)
    if seed is None or not seed.id.startswith("pennsieve:"):
        raise DataStagingError(f"seed dataset {bb.subsystem!r} is not a published Pennsieve dataset")
    try:
        files = files or DiscoverFiles(seed.id)
        index = files.index()
    except Exception as exc:
        raise DataStagingError(f"could not read the file index of {seed.id}: {exc}") from exc

    data_dir = hyp_workdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    index_path = data_dir / "files_index.json"
    index_path.write_text(json.dumps({
        "dataset_id": files.dataset_id, "version": files.version, "doi": seed.doi,
        "title": seed.title, "file_count": len(index), "files": index}, indent=1))
    result = StagingResult(
        dataset_id=files.dataset_id, version=files.version, data_dir=data_dir,
        index_path=index_path, total_files=len(index),
        file_types=dict(Counter((f["fileType"] or Path(f["path"]).suffix.lstrip(".") or "?").upper()
                                for f in index).most_common(12)))

    if os.environ.get("VP_ANALYZE_STAGE_DATA", "1") != "0":
        for path, size in select_sample(index, hypothesis):
            try:
                files.download([path], data_dir)
                result.staged.append({"path": path, "bytes": size})
            except Exception as exc:  # noqa: BLE001 - one bad file is recorded; all failing raises below
                result.failed.append(path)
                bb.log("data_staging", "file_error", f"{path}: {type(exc).__name__}: {exc}")
        if result.failed and not result.staged:
            raise DataStagingError(f"every sample download failed ({len(result.failed)} files); "
                                   f"first: {result.failed[0]}")

    bb.log("data_staging", "done",
           f"dataset {result.dataset_id} v{result.version}: {result.total_files} files indexed, "
           f"{len(result.staged)} downloaded ({sum(s['bytes'] for s in result.staged) / 1e6:.1f} MB), "
           f"{len(result.failed)} failed")
    print(f"  [staging] {result.total_files} files indexed, {len(result.staged)} downloaded, "
          f"{len(result.failed)} failed -> {data_dir}", flush=True)
    return result


def select_sample(index: list[dict], hypothesis: Hypothesis) -> list[tuple[str, int]]:
    """Small tables first, then a folder-spread sample of the data type the
    hypothesis names (or the most common one), within the size budget."""
    budget = MAX_BYTES
    chosen: list[tuple[str, int]] = []

    def take(path: str, size: int) -> bool:
        nonlocal budget
        if size > budget:
            return False
        chosen.append((path, size))
        budget -= size
        return True

    for f in index:
        if f["path"].lower().endswith(TABLE_SUFFIXES) and f["size"] <= TABLE_FILE_CAP:
            take(f["path"], f["size"])

    data_files = [f for f in index if Path(f["path"]).suffix.lower() not in SKIP_SUFFIXES
                  and not f["path"].lower().endswith(TABLE_SUFFIXES)]
    if not data_files:
        return chosen
    text = f"{hypothesis.claim} {hypothesis.rationale} {hypothesis.suggested_experiment}".lower()
    by_type = Counter(Path(f["path"]).suffix.lower() for f in data_files)
    named = [ext for ext in by_type
             if ext.lstrip(".") and re.search(rf"\b{re.escape(ext.lstrip('.'))}\b", text)]
    target = named[0] if named else by_type.most_common(1)[0][0]
    groups: dict[str, list[dict]] = defaultdict(list)
    for f in sorted((f for f in data_files if Path(f["path"]).suffix.lower() == target),
                    key=lambda f: f["path"]):
        groups[str(Path(f["path"]).parent)].append(f)
    # Round-robin across folders, so the sample covers subjects and conditions
    # instead of the first folder alphabetically.
    queues = list(groups.values())
    picked = 0
    while queues and picked < MAX_FILES:
        for q in list(queues):
            if picked >= MAX_FILES:
                break
            f = q.pop(0)
            if take(f["path"], f["size"]):
                picked += 1
            if not q:
                queues.remove(q)
    return chosen


def prompt_section(result: StagingResult) -> str:
    """The analyst prompt's data-access instructions."""
    fetch = (f"python -m pennsieve_ai_utils.tools.discover_files fetch {result.dataset_id} "
             f"--version {result.version} --dest data")
    lines = [
        "## Data access: read this before touching the dataset",
        "",
        (f"The seed dataset is the published Pennsieve dataset **{result.dataset_id}** "
         f"(version {result.version}). Published datasets are public through Pennsieve "
         "Discover. No account, API key or AWS credentials are needed, and none are available."),
        "",
        (f"- `data/files_index.json` lists all {result.total_files} published files, with "
         "`path`, `size` and `fileType`. File types: "
         + ", ".join(f"{k} ({v})" for k, v in result.file_types.items()) + "."),
    ]
    if result.staged:
        lines.append(f"- These {len(result.staged)} files are already downloaded under `data/`:")
        lines += [f"  - `data/{s['path']}` ({s['bytes'] / 1e6:.2f} MB)" for s in result.staged[:40]]
        if len(result.staged) > 40:
            lines.append(f"  - and {len(result.staged) - 40} more (see `data/files_index.json`)")
    lines += [
        "- To download more, use the paths from `data/files_index.json`:",
        "",
        "```",
        f"{fetch} <path> [<path> ...]",
        f'{fetch} --glob "files/<folder>/*.fcs" --limit 10',
        "```",
        "",
        ("Read data from `data/`, and have the notebook download any files it needs with the "
         "command above, so that it runs again from a clean folder."),
        "",
        ("**Do not** use AWS credentials, `boto3`, the `aws` command or `s3://` and S3 URLs, and "
         "do not try the authenticated Pennsieve API. They are not available here, and any "
         "attempt is wasted turns."),
    ]
    return "\n".join(lines)
