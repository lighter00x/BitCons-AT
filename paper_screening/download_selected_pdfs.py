#!/usr/bin/env python3
"""Download fine-screened PDFs with retries and a validation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def safe_name(value: str, limit: int = 110) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_.")
    return value[:limit] or "paper"


def valid_pdf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as handle:
        return handle.read(5) == b"%PDF-"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, retries: int, timeout: int) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 BitCons literature downloader/1.0"},
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with temporary.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            if not valid_pdf(temporary):
                raise ValueError("downloaded content is not a valid PDF")
            temporary.replace(destination)
            return
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = error
            if temporary.exists():
                temporary.unlink()
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(str(last_error))


def openreview_attachment_url(note_id: str) -> str:
    return (
        "https://api2.openreview.net/notes/attachment?"
        + urllib.parse.urlencode({"id": note_id, "name": "pdf"})
    )


def resolve_download_url(paper: dict[str, Any], timeout: int) -> str:
    """Prefer public API attachment URLs over browser-challenged mirrors."""
    pdf_url = paper["pdf"]
    parsed_pdf = urllib.parse.urlparse(pdf_url)
    if parsed_pdf.netloc == "openreview.net":
        forum_query = urllib.parse.parse_qs(
            urllib.parse.urlparse(paper.get("url", "")).query
        )
        if forum_query.get("id"):
            return openreview_attachment_url(forum_query["id"][0])

    raw_match = re.match(
        r"https://raw\.githubusercontent\.com/mlresearch/(v\d+)/main/assets/([^/]+)/[^/]+\.pdf$",
        pdf_url,
    )
    if raw_match:
        volume, slug = raw_match.groups()
        page_url = f"https://proceedings.mlr.press/{volume}/{slug}.html"
        request = urllib.request.Request(
            page_url, headers={"User-Agent": "Mozilla/5.0 BitCons literature downloader/1.0"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            page = response.read().decode("utf-8", errors="replace")
        forum_match = re.search(r"openreview\.net/forum\?id=([^\"&<]+)", page)
        if forum_match:
            return openreview_attachment_url(forum_match.group(1))
    return pdf_url


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    papers: list[dict[str, Any]] = json.loads(args.input.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, paper in enumerate(papers, start=1):
        filename = (
            f"{index:03d}_{paper['priority']}_{paper['venue']}{paper['year']}_"
            f"{safe_name(paper['title'])}.pdf"
        )
        destination = args.output_dir / filename
        entry = {
            "paper_id": paper["paper_id"],
            "title": paper["title"],
            "priority": paper["priority"],
            "source_url": paper["pdf"],
            "filename": filename,
        }
        try:
            status = "existing" if valid_pdf(destination) else "downloaded"
            if status == "downloaded":
                resolved_url = resolve_download_url(paper, args.timeout)
                entry["resolved_url"] = resolved_url
                download(resolved_url, destination, args.retries, args.timeout)
            entry.update({
                "status": status,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            })
            print(f"[{index:02d}/{len(papers)}] OK {filename}", flush=True)
        except Exception as error:  # Keep the remaining downloads progressing.
            entry.update({"status": "failed", "error": str(error)})
            print(f"[{index:02d}/{len(papers)}] FAIL {paper['title']}: {error}", flush=True)
        manifest.append(entry)
        (args.output_dir / "download_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    success = sum(entry["status"] in {"downloaded", "existing"} for entry in manifest)
    failed = len(manifest) - success
    print(json.dumps({"total": len(manifest), "success": success, "failed": failed}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
