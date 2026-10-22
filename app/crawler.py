import asyncio
import ipaddress
import json
import os
import re
import socket
import time
from collections import deque
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree

import httpx
from bs4 import BeautifulSoup

from .config import settings
from .store import add_document, remove_document
from .taxonomy import classify_category, infer_department, extract_academic_year

running = False
_data_dir = Path(settings.data_path)
_status_path = _data_dir / "crawl_status.json"
_checkpoint_path = _data_dir / "crawl_checkpoint.json"
last_sync = {"status": "never", "processed": 0, "failed": 0, "started_at": None, "finished_at": None}

def _write_json(path: Path, value: dict):
    """Atomically persist crawler state so restarts do not erase progress."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(temp, path)
    except OSError:
        # The source index remains usable if status/checkpoint storage is unavailable.
        pass

def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None

_saved_status = _read_json(_status_path)
if isinstance(_saved_status, dict):
    last_sync = _saved_status
    try:
        stale_running_status = time.time() - _status_path.stat().st_mtime > 600
    except OSError:
        stale_running_status = True
    if last_sync.get("status") == "running" and stale_running_status:
        last_sync["status"] = "interrupted"
        _write_json(_status_path, last_sync)

def has_pending_checkpoint():
    checkpoint = _read_json(_checkpoint_path)
    if isinstance(checkpoint, dict) and checkpoint.get("queue") and not (checkpoint.get("status") == "running" and last_sync.get("status") == "running"):
        return True
    report = checkpoint if isinstance(checkpoint, dict) else last_sync
    return bool(_retryable_urls(report.get("errors", []))) and report.get("status") in {"partial", "interrupted"}

def _retryable_urls(errors):
    return list(dict.fromkeys(item.get("url") for item in errors or []
                              if item.get("url") and "Response exceeded the 8 MB streaming limit" in item.get("error", "")))

def status_snapshot():
    """Return the latest status, including progress written by a worker process."""
    saved = _read_json(_status_path)
    if not isinstance(saved, dict): return dict(last_sync)
    try:
        stale = time.time() - _status_path.stat().st_mtime > 600
    except OSError:
        stale = False
    if saved.get("status") == "running" and stale:
        saved["status"] = "interrupted"
        _write_json(_status_path, saved)
        last_sync.update(saved)
    return saved
AGENT = "PCCOEStudentAssistant"

@lru_cache(maxsize=1024)
def _public_host(host: str) -> bool:
    try:
        results = socket.getaddrinfo(host, None)
        return bool(results) and all(ipaddress.ip_address(item[4][0]).is_global for item in results)
    except (OSError, ValueError):
        return False

def safe_url(url: str) -> bool:
    p = urlsplit(url)
    domains = [d.strip().lower() for d in settings.allowed_domains.split(",") if d.strip()]
    if p.scheme not in ("http", "https") or not p.hostname or p.username or p.password:
        return False
    try:
        port=p.port
    except ValueError:
        return False
    if port not in (None, 80, 443):
        return False
    host = p.hostname.lower()
    if not any(host == d or host.endswith("." + d) for d in domains):
        return False
    if not _public_host(host):
        return False
    return True

def canonical(url: str) -> str:
    p = urlsplit(url)
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path or "/", p.query, ""))

def classify(url: str, title: str) -> str:
    return classify_category(url,title)

def extract_html(soup: BeautifulSoup, base_url: str = "") -> str:
    for tag in soup(["script", "style", "nav", "footer", "noscript", "svg"]):
        tag.decompose()
    # Preserve semantic headers inside profile cards/articles; only discard
    # the site-wide masthead that repeats on every crawled page.
    for tag in soup.find_all("header"):
        classes=" ".join(tag.get("class",[])) if isinstance(tag.get("class"),list) else str(tag.get("class",""))
        is_site_header=(tag.parent and tag.parent.name=="body") or re.search(r"site|masthead|navbar|navigation",classes+" "+str(tag.get("id","")),re.I)
        if tag.find_parent(["main","article"]) is None and is_site_header:
            tag.decompose()
    for tag in soup.find_all(attrs={"class": re.compile(r"(^|[-_ ])(menu|navbar|navigation|breadcrumb|social)([-_ ]|$)", re.I)}):
        tag.decompose()
    for tag in soup.find_all(attrs={"id": re.compile(r"(menu|navbar|navigation|breadcrumb|social)", re.I)}):
        tag.decompose()
    sections = []
    # Profile and directory pages often store card fields in generic divs
    # (including labels such as Designation and Email) that are not exposed by
    # the semantic tags below. Capture leaf profile cards as intact records so
    # chunking cannot separate a teacher's name from their designation.
    profile_block = re.compile(r"(?:faculty|teacher|staff|person|member|profile).*(?:card|item|detail|info)|(?:card|item|detail|info).*(?:faculty|teacher|staff|person|member|profile)", re.I)
    block_candidates = soup.find_all(["article", "section", "div", "li"], attrs={"class": profile_block})
    block_candidates += soup.find_all(["article", "section", "div", "li"], attrs={"id": profile_block})
    captured_blocks = set()
    for node in block_candidates:
        # Prefer the smallest matching container; a parent directory wrapper
        # would concatenate every card into one oversized, poorly searchable chunk.
        nested = node.find_all(["article", "section", "div", "li"], attrs={"class": profile_block})
        nested += node.find_all(["article", "section", "div", "li"], attrs={"id": profile_block})
        if nested:
            continue
        value = node.get_text("\n", strip=True)
        key = re.sub(r"\s+", " ", value).casefold()
        if len(value) < 30 or len(value) > 5000 or key in captured_blocks:
            continue
        if not re.search(r"designation|professor|lecturer|teacher|faculty|email|qualification", value, re.I):
            continue
        captured_blocks.add(key)
        sections.append("Faculty profile record:\n" + value)
    for node in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "a", "button", "dt", "dd", "tr"]):
        if node.name=="tr":
            cells=[cell.get_text(" ",strip=True) for cell in node.find_all(["th","td"],recursive=False)]
            value=" | ".join(cell for cell in cells if cell)
        else:
            value = node.get_text(" ", strip=True)
        if value:
            if node.name=="a" and node.get("href"):
                sections.append(f"Link: {value} | {urljoin(base_url, node.get('href'))}")
            else:
                sections.append(value)
    # Keep useful pages that are mostly plain text without semantic tags.
    if not sections:
        sections.append(soup.get_text(" ", strip=True))
    return "\n".join(dict.fromkeys(sections))

async def load_robots(url: str, client: httpx.AsyncClient, cache: dict):
    p = urlsplit(url)
    origin = f"{p.scheme}://{p.netloc}"
    if origin in cache:
        return cache[origin]
    robots_url = origin + "/robots.txt"
    async with asyncio.timeout(15):
        async with client.stream("GET", robots_url) as response:
            body = await read_limited(response, 1_000_000) if response.status_code == 200 else b""
    parser = RobotFileParser()
    parser.set_url(robots_url)
    if response.status_code == 404:
        parser.parse(["User-agent: *", "Allow: /"])
        sitemaps = []
    elif response.status_code == 200:
        text = body.decode("utf-8", errors="replace")
        parser.parse(text.splitlines())
        sitemaps = [line.partition(":")[2].strip() for line in text.splitlines() if line.lower().startswith("sitemap:")]
    else:
        raise RuntimeError(f"robots.txt returned HTTP {response.status_code} for {origin}")
    cache[origin] = (parser, sitemaps)
    return cache[origin]

async def read_limited(response, limit):
    chunks=[]; total=0
    async for chunk in response.aiter_bytes():
        total+=len(chunk)
        if total>limit:
            raise ValueError(f"Response exceeded the {limit // 1_000_000} MB streaming limit")
        chunks.append(chunk)
    return b"".join(chunks)

async def add_sitemap(url: str, client: httpx.AsyncClient, queue: deque, visited: set, depth=0, errors=None):
    if depth > 4 or url in visited or not safe_url(url) or len(queue)>=settings.max_crawl_pages:
        return
    visited.add(url)
    try:
        async with asyncio.timeout(30):
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    return
                body = await read_limited(response, 4_000_000)
        if url.lower().endswith(".gz"):
            import zlib
            decompressor=zlib.decompressobj(16+zlib.MAX_WBITS)
            body=decompressor.decompress(body,32_000_001)
            if len(body)>32_000_000 or decompressor.unconsumed_tail: return
            body+=decompressor.flush()
            if len(body)>32_000_000: return
        root = ElementTree.fromstring(body)
        for element in root.iter():
            if len(queue)>=settings.max_crawl_pages: break
            if not element.tag.endswith("loc") or not element.text:
                continue
            candidate = canonical(element.text.strip())
            if not safe_url(candidate):
                continue
            if candidate.lower().endswith((".xml", ".xml.gz")):
                await add_sitemap(candidate, client, queue, visited, depth + 1, errors)
            elif candidate not in queue:
                queue.append(candidate)
    except Exception as exc:
        if errors is not None: errors.append({"url":url,"error":str(exc)[:240]})
        return

async def crawl():
    global running, last_sync
    if running:
        return {"status": "already_running"}
    running = True
    started = datetime.now(timezone.utc)
    seen, sitemap_seen, robots_cache = set(), set(), {}
    checkpoint = _read_json(_checkpoint_path)
    resume = isinstance(checkpoint, dict) and checkpoint.get("status") in {"running", "interrupted", "partial"} and bool(checkpoint.get("queue"))
    if resume:
        queue = deque(checkpoint.get("queue", []))
        seen = set(checkpoint.get("seen", []))
        sitemap_seen = set(checkpoint.get("sitemap_seen", []))
        # Retry files that only failed because of the old 8 MB download cap.
        # The current cap is larger, so those official PDFs should not be lost
        # from a resumable crawl just because an earlier batch rejected them.
        retry_urls = _retryable_urls(checkpoint.get("errors", []))
        queue = deque(dict.fromkeys(retry_urls + list(queue)))
    else:
        retry_urls = _retryable_urls(last_sync.get("errors", [])) if last_sync.get("status") in {"partial", "interrupted"} else []
        if retry_urls:
            queue = deque(retry_urls)
        else:
            seed_urls = [canonical(settings.college_url)]
            seed_urls.extend(canonical(value.strip()) for value in settings.seed_urls.split(",") if value.strip())
            queue = deque(dict.fromkeys(seed_urls))
    processed = int(checkpoint.get("processed", 0)) if resume else 0
    failed = int(checkpoint.get("failed", 0)) if resume else 0
    updated = int(checkpoint.get("updated_sources", 0)) if resume else 0
    unchanged = int(checkpoint.get("unchanged_sources", 0)) if resume else 0
    removed = int(checkpoint.get("removed_chunks", 0)) if resume else 0
    host_counts = checkpoint.get("hosts", {}) if resume else {}
    skipped_robots = int(checkpoint.get("skipped", 0)) if resume else 0
    robots_blocked = int(checkpoint.get("robots_blocked", 0)) if resume else 0
    unsupported = int(checkpoint.get("unsupported", 0)) if resume else 0
    skipped_details = checkpoint.get("skipped_details", []) if resume else []
    sitemap_errors = checkpoint.get("sitemap_error_details", []) if resume else []
    failures = checkpoint.get("errors", []) if resume else []
    started = datetime.fromisoformat(checkpoint["started_at"]) if resume and checkpoint.get("started_at") else started
    starting_seen_count = len(seen)
    last_sync = {"status": "running", "processed": processed, "failed": failed, "started_at": started.isoformat(), "resuming": resume, "queued": len(queue)}
    _write_json(_status_path, last_sync)

    last_checkpoint_processed = processed
    def checkpoint_state(status="running", force=False):
        nonlocal last_checkpoint_processed
        state = {"status": status, "queue": list(queue), "seen": list(seen), "sitemap_seen": list(sitemap_seen),
                 "processed": processed, "failed": failed, "updated_sources": updated, "unchanged_sources": unchanged,
                 "removed_chunks": removed, "skipped": skipped_robots, "robots_blocked": robots_blocked,
                 "unsupported": unsupported, "skipped_details": skipped_details[-50:], "hosts": host_counts,
                 "started_at": started.isoformat(), "errors": failures[-100:], "sitemap_error_details": sitemap_errors[-50:]}
        # The detailed queue snapshot can be large. Persist it every 25 pages;
        # the compact public status is still refreshed after each page.
        if force or status != "running" or processed - last_checkpoint_processed >= 25:
            _write_json(_checkpoint_path, state)
            last_checkpoint_processed = processed
        public = {k: v for k, v in state.items() if k not in {"queue", "seen", "sitemap_seen"}}
        public.update({"status": status, "discovered": len(seen), "queued": len(queue), "sitemap_errors": len(sitemap_errors)})
        _write_json(_status_path, public)
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=False, headers={"User-Agent": AGENT + "/1.0 (polite academic crawler)"}) as client:
            parser, sitemaps = await load_robots(settings.college_url, client, robots_cache)
            for sitemap in sitemaps:
                await add_sitemap(sitemap, client, queue, sitemap_seen, errors=sitemap_errors)
            checkpoint_state(force=True)
            while queue and len(seen) - starting_seen_count < settings.max_crawl_pages:
                url = canonical(queue.popleft())
                if url in seen or not safe_url(url):
                    continue
                # Save the current item at the front so a process restart retries it safely.
                queue.appendleft(url)
                checkpoint_state()
                queue.popleft()
                try:
                    parser, host_sitemaps = await load_robots(url, client, robots_cache)
                    for sitemap in host_sitemaps:
                        await add_sitemap(sitemap, client, queue, sitemap_seen, errors=sitemap_errors)
                except Exception as exc:
                    failed += 1
                    failures.append({"url": url, "error": str(exc)[:240]})
                    continue
                if not parser.can_fetch(AGENT, url):
                    skipped_robots += 1
                    robots_blocked += 1
                    skipped_details.append({"url":url,"reason":"Disallowed by host robots.txt"})
                    continue
                seen.add(url)
                try:
                    async with asyncio.timeout(45):
                        async with client.stream("GET", url) as response:
                            if response.is_redirect:
                                target = canonical(urljoin(url, response.headers.get("location", "")))
                                if safe_url(target) and target not in queue:
                                    queue.append(target)
                                continue
                            response.raise_for_status()
                            body = await read_limited(response, 40_000_000)
                    content_type = response.headers.get("content-type", "").lower()
                    changed = 0
                    file_path=url.lower().split("?",1)[0]
                    extension=Path(urlsplit(file_path).path).suffix.lower()
                    if "pdf" in content_type or extension==".pdf" or body.startswith(b"%PDF"):
                        from .ingest import ingest
                        filename = urlsplit(url).path.rsplit("/", 1)[-1] or "PCCOE document.pdf"
                        changed = await asyncio.to_thread(ingest, filename, body, url)
                    elif extension in {".docx", ".pptx", ".xlsx"} and body.startswith(b"PK"):
                        from .ingest import ingest
                        filename = urlsplit(url).path.rsplit("/", 1)[-1] or f"PCCOE document{extension}"
                        changed = await asyncio.to_thread(ingest, filename, body, url)
                    elif extension==".csv" or "text/csv" in content_type:
                        from .ingest import ingest
                        filename = urlsplit(url).path.rsplit("/", 1)[-1] or "PCCOE data.csv"
                        changed = await asyncio.to_thread(ingest, filename, body, url)
                    elif "html" in content_type or "xhtml" in content_type:
                        soup = BeautifulSoup(body, "html.parser")
                        title = soup.title.get_text(" ", strip=True) if soup.title else url
                        description=soup.find("meta",attrs={"name":re.compile(r"^description$",re.I)})
                        page_description=description.get("content","").strip() if description else ""
                        # Save links before removing navigation/footer markup for clean indexing.
                        # Department websites and syllabus PDFs are commonly linked from menus.
                        child_links = [canonical(urljoin(url, anchor["href"])) for anchor in soup.find_all("a", href=True)]
                        text = extract_html(soup, url)
                        if page_description: text="Page description: "+page_description+"\n"+text
                        if text:
                            checked_at=datetime.now(timezone.utc).isoformat()
                            changed = await asyncio.to_thread(add_document, text, {"title": title, "url": url, "category": classify_category(url,title,text), "department": infer_department(url,title), "academic_year": extract_academic_year(url+" "+title+" "+text), "type": "HTML", "last_checked_at": checked_at, "last_successful_sync_at": checked_at})
                        for child in child_links:
                            if safe_url(child) and child not in seen and child not in queue:
                                queue.append(child)
                    else:
                        skipped_robots += 1
                        unsupported += 1
                        skipped_details.append({"url":url,"reason":f"Unsupported content type: {content_type or 'unknown'}"})
                        continue
                    processed += 1
                    recovered = sum(1 for item in failures if item.get("url") == url)
                    if recovered:
                        failures[:] = [item for item in failures if item.get("url") != url]
                        failed = max(0, failed - recovered)
                    host = urlsplit(url).hostname or "unknown"
                    host_counts[host] = host_counts.get(host, 0) + 1
                    if changed:
                        updated += 1
                    else:
                        unchanged += 1
                except httpx.HTTPStatusError as exc:
                    failed += 1
                    if exc.response.status_code in (404, 410):
                        removed += await asyncio.to_thread(remove_document, url)
                    failures.append({"url": url, "error": f"HTTP {exc.response.status_code}"})
                except Exception as exc:
                    failed += 1
                    failures.append({"url": url, "error": str(exc)[:240]})
                last_sync = {"status": "running", "processed": processed, "updated_sources": updated, "unchanged_sources": unchanged, "removed_chunks": removed, "failed": failed, "skipped": skipped_robots, "robots_blocked": robots_blocked, "unsupported": unsupported, "skipped_details": skipped_details[-50:], "sitemap_errors": len(sitemap_errors), "discovered": len(seen), "hosts": host_counts.copy(), "started_at": started.isoformat()}
                checkpoint_state()
                await asyncio.sleep(0.25)
    except Exception as exc:
        failed += 1
        failures.append({"url": settings.college_url, "error": str(exc)[:240]})
    finally:
        running = False
        duration = (datetime.now(timezone.utc) - started).total_seconds()
        capped = bool(queue) and len(seen) - starting_seen_count >= settings.max_crawl_pages
        status = "failed" if failed and not processed else ("partial" if failed or capped else "complete")
        last_sync = {"status": status, "processed": processed, "updated_sources": updated, "unchanged_sources": unchanged, "removed_chunks": removed, "failed": failed, "skipped": skipped_robots, "robots_blocked": robots_blocked, "unsupported": unsupported, "skipped_details": skipped_details[-50:], "sitemap_errors": len(sitemap_errors), "discovered": len(seen), "hosts": host_counts, "started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(), "duration_seconds": round(duration, 2), "errors": failures[:20], "sitemap_error_details": sitemap_errors[:10]}
        checkpoint_state(status)
        if not queue:
            try: _checkpoint_path.unlink(missing_ok=True)
            except OSError: pass
    return last_sync

if __name__ == "__main__":
    # Supports a recoverable standalone refresh for deployments without the
    # admin dashboard. The same persistent status/checkpoint files are used.
    asyncio.run(crawl())
