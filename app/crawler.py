import asyncio
import ipaddress
import json
import os
import re
import socket
import threading
import time
from collections import Counter, deque
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree

import httpx
from bs4 import BeautifulSoup

from .config import settings
from .store import add_document, collection, remove_document, _terms
from .taxonomy import classify_category, infer_department, extract_academic_year

running = False
_stop_requested = False
_query_lock = threading.Lock()
_active_queries = 0
_data_dir = Path(settings.data_path)
_status_path = _data_dir / "crawl_status.json"
_checkpoint_path = _data_dir / "crawl_checkpoint.json"
last_sync = {"status": "never", "processed": 0, "failed": 0, "started_at": None, "finished_at": None}
_crawl_continuation_task = None
_dns_cache = {}
_dns_cache_lock = threading.Lock()
_STATIC_ASSET_EXTENSIONS = {
    ".avif", ".bmp", ".css", ".eot", ".gif", ".ico", ".jpeg", ".jpg",
    ".js", ".map", ".mp3", ".mp4", ".ogg", ".otf", ".png", ".svg",
    ".ttf", ".webm", ".webp", ".woff", ".woff2",
}

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
    if isinstance(checkpoint, dict) and checkpoint.get("queue") and checkpoint.get("status") in {"running","interrupted","partial","stopped"}:
        return True
    report = checkpoint if isinstance(checkpoint, dict) else last_sync
    return bool(_retryable_urls(report.get("errors", []))) and report.get("status") in {"partial", "interrupted", "stopped"}

def request_stop():
    """Ask the active crawl to stop after its current request and checkpoint."""
    global _stop_requested
    if not running:
        return {"status":"not_running","running":False}
    _stop_requested=True
    return {"status":"stopping","running":True}

def recover_interrupted_worker():
    """Recover a persisted crawl marker after this API process has restarted.

    The in-memory ``running`` flag is process-local. A recent on-disk "running"
    status from a previous process must not suppress startup resume indefinitely.
    """
    global last_sync
    saved = _read_json(_status_path)
    if not isinstance(saved, dict) or saved.get("status") != "running" or running:
        return False
    saved["status"] = "interrupted"
    saved["interrupted_at"] = datetime.now(timezone.utc).isoformat()
    last_sync = saved
    _write_json(_status_path, saved)
    return True

def _retryable_urls(errors):
    retryable=[]
    for item in errors or []:
        url=item.get("url")
        error=str(item.get("error","")).strip()
        retry_after=item.get("retry_after")
        if retry_after:
            try:
                if datetime.fromisoformat(retry_after)>datetime.now(timezone.utc):
                    continue
            except (TypeError,ValueError):
                pass
        # Retry interrupted/empty-timeout failures and transient server errors.
        # Permanent 4xx responses remain terminal; robots.txt exclusions never
        # enter this list because they are recorded as skips, not failures.
        transient=(not error or "Response exceeded the 8 MB streaming limit" in error
                   or re.search(
                       r"Timeout|ConnectError|ReadError|RemoteProtocolError|"
                       r"Temporary failure in name resolution|Name or service not known|"
                       r"RateLimitError|RESOURCE_EXHAUSTED|quota|rate.?limit|"
                       r"robots\.txt returned HTTP 30[1278]|"
                       r"HTTP (?:408|425|429|5\d\d)",
                       error,re.I,
                   ))
        if url and transient: retryable.append(url)
    return list(dict.fromkeys(retryable))

def _prioritize_unseen_seeds(queue, seed_urls, seen, retry_urls=()):
    """Move every not-yet-seen configured seed ahead of backlog and retries.

    A seed may already be in a saved crawl queue from an older run. Merely
    prepending seeds absent from that queue leaves high-value sources buried
    behind stale failures, so remove seed duplicates before rebuilding order.
    """
    priority=[url for url in dict.fromkeys(seed_urls) if url not in seen]
    priority_set=set(priority)
    return deque(dict.fromkeys(priority+list(retry_urls)+[url for url in queue if url not in priority_set]))

def _error_summary(exc):
    message=" ".join(str(exc).split())
    return (f"{type(exc).__name__}: {message}" if message else type(exc).__name__)[:240]

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

@contextmanager
def prioritize_queries():
    """Temporarily yield crawler work to interactive retrieval requests."""
    global _active_queries
    with _query_lock:
        _active_queries += 1
    try:
        yield
    finally:
        with _query_lock:
            _active_queries = max(0, _active_queries - 1)
AGENT = "PCCOEStudentAssistant"

def _public_host(host: str) -> bool:
    now=time.monotonic()
    with _dns_cache_lock:
        cached=_dns_cache.get(host)
        if cached and cached[0]>now:
            return cached[1]
    try:
        results = socket.getaddrinfo(host, None)
        public=bool(results) and all(ipaddress.ip_address(item[4][0]).is_global for item in results)
    except (OSError, ValueError):
        public=False
    # DNS failures are usually transient here. Cache them briefly, but keep
    # successful public-host checks warm for normal crawl throughput.
    ttl=300 if public else 15
    with _dns_cache_lock:
        _dns_cache[host]=(time.monotonic()+ttl,public)
    return public

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

def _crawlable_url(url: str) -> bool:
    """Keep page/document/data links while excluding non-text static assets."""
    filename=urlsplit(url).path.rsplit("/",1)[-1].lower()
    if re.search(r"(?:^|[-_.])manifest\.json$",filename):
        return False
    extension=("."+filename.rsplit(".",1)[-1]) if "." in filename else ""
    return extension not in _STATIC_ASSET_EXTENSIONS

def _strip_js_comments(source):
    """Remove JavaScript comments without damaging quoted URLs or text."""
    out=[]; i=0; quote=None; escaped=False
    while i<len(source):
        char=source[i]; nxt=source[i+1] if i+1<len(source) else ""
        if quote:
            out.append(char)
            if escaped: escaped=False
            elif char=="\\": escaped=True
            elif char==quote: quote=None
            i+=1; continue
        if char in {"'",'"',"`"}:
            quote=char; out.append(char); i+=1; continue
        if char=="/" and nxt=="/":
            while i<len(source) and source[i] not in "\r\n": i+=1
            out.append("\n"); continue
        if char=="/" and nxt=="*":
            i+=2
            while i+1<len(source) and source[i:i+2]!="*/":
                if source[i] in "\r\n": out.append(source[i])
                i+=1
            i=min(len(source),i+2); continue
        out.append(char); i+=1
    return "".join(out)

def _js_profile_records(source):
    """Extract active name/role/contact objects from common JS roster arrays."""
    clean=_strip_js_comments(source)
    array=re.search(r"\b(?:let|var|const)\s+([\w$]*(?:faculty|gallary|gallery|staff|teacher|member)[\w$]*)\s*=\s*\[",clean,re.I)
    if not array: return []
    i=array.end(); level=1; quote=None; escaped=False; object_start=None; objects=[]
    while i<len(clean) and level:
        char=clean[i]
        if quote:
            if escaped: escaped=False
            elif char=="\\": escaped=True
            elif char==quote: quote=None
            i+=1; continue
        if char in {"'",'"',"`"}: quote=char; i+=1; continue
        if char=="[": level+=1
        elif char=="]":
            level-=1
            if level==0 and object_start is not None:
                objects.append(clean[object_start:i]); object_start=None
        elif char=="{" and level==1: object_start=i
        elif char=="}" and level==1 and object_start is not None:
            objects.append(clean[object_start:i+1]); object_start=None
        i+=1
    records=[]
    for obj in objects:
        fields={}
        for key in ("name","title","designation","email","phone","mobile","qualification","experience","AOI","areaOfInterest"):
            match=re.search(r"(?:^|[,{\s])(?:['\"]?"+re.escape(key)+r"['\"]?)\s*:\s*(['\"])((?:\\.|.)*?)\1",obj,re.I|re.S)
            if match:
                value=match.group(2).replace(r"\'","'").replace(r'\"','"').replace(r"\\","\\").strip()
                fields[key.lower()]=value
        name=fields.get("name","").strip()
        designation=fields.get("designation","").strip()
        if not name and fields.get("title"):
            name=re.sub(r"\s*\([^)]*\)\s*$","",fields["title"]).strip()
        if not name or not designation or re.search(r"teaching assistant|admin|non[- ]teaching",designation,re.I):
            continue
        email=re.sub(r"\s*@\s*","@",fields.get("email","")).strip(" ,;")
        phone=fields.get("phone") or fields.get("mobile","")
        line=f"{name}\nDesignation | {designation}"
        if email: line+=f"\nEmail | {email}"
        if phone: line+=f"\nPhone | {phone}"
        for key,label in (("qualification","Qualification"),("experience","Experience"),("aoi","Area of Interest"),("areaofinterest","Area of Interest")):
            value=fields.get(key,"").strip()
            if value: line+=f"\n{label} | {value}"
        records.append(line)
    # Stable order and de-duplication: some source scripts repeat cards in
    # filtered arrays. Ignore repeated same-name records and retain first card.
    unique={}
    for record in records:
        name=record.splitlines()[0].casefold()
        unique.setdefault(name,record)
    return list(unique.values())

_JS_RECORD_KEYS={
    "name":"Name","title":"Title","designation":"Designation","role":"Role",
    "email":"Email","phone":"Phone","mobile":"Phone","contact":"Contact",
    "qualification":"Qualification","experience":"Experience","aoi":"Area of Interest",
    "areaofinterest":"Area of Interest","department":"Department","program":"Program",
    "course":"Course","subject":"Subject","code":"Code","unit":"Unit",
    "description":"Description","content":"Content","heading":"Heading","date":"Date",
    "event":"Event","club":"Club","amount":"Amount","fee":"Fee","package":"Package",
    "company":"Company","placement":"Placement","url":"URL","href":"Link",
    "venue":"Venue","semester":"Semester","academicYear":"Academic year",
}

def _js_object_records(source, limit=200):
    """Extract useful scalar records from data objects embedded in JS bundles."""
    clean=_strip_js_comments(source)
    records=[]; seen=set(); i=0; quote=None; escaped=False; starts=[]
    while i<len(clean) and len(records)<limit:
        char=clean[i]
        if quote:
            if escaped: escaped=False
            elif char=="\\": escaped=True
            elif char==quote: quote=None
            i+=1; continue
        if char in {"'",'"',"`"}: quote=char; i+=1; continue
        if char=="{": starts.append(i)
        elif char=="}" and starts:
            start=starts.pop()
            obj=clean[start:i+1]
            fields=[]
            for key,label in _JS_RECORD_KEYS.items():
                match=re.search(r"(?:^|[,{\s])(?:['\"]?"+re.escape(key)+r"['\"]?)\s*:\s*(['\"])((?:\\.|.)*?)\1",obj,re.I|re.S)
                if match:
                    value=match.group(2).replace(r"\'","'").replace(r'\"','"').replace(r"\\","\\").strip()
                    if value and len(value)<1200 and not value.startswith(("http://", "https://")) or (value and key.lower() in {"url","href"}):
                        fields.append((label," ".join(value.split())))
            labels={label.casefold() for label,_ in fields}
            identifying=labels & {"name","title","designation","role","course","subject","code","unit","description","content","heading","date","event","club","amount","fee","company","placement"}
            if len(fields)>=2 and identifying:
                line="\n".join(f"{label} | {value}" for label,value in fields)
                normalized=re.sub(r"\W+"," ",line.casefold()).strip()
                if normalized not in seen:
                    records.append(line); seen.add(normalized)
        i+=1
    return records

def _json_payload_text(raw, max_records=1200):
    """Turn JSON-LD and common client-app state blobs into readable records."""
    output=[]; seen=set(); visited=0
    def walk(value):
        nonlocal visited
        if visited>=max_records: return
        if isinstance(value,dict):
            pairs=[]
            for key,item in value.items():
                if isinstance(item,(str,int,float,bool)) and str(item).strip():
                    pairs.append((str(key),str(item).strip()))
            meaningful={key.casefold() for key,_ in pairs}&{key.casefold() for key in _JS_RECORD_KEYS}
            if len(pairs)>=2 and meaningful:
                line="\n".join(f"{key} | {val}" for key,val in pairs[:32])
                norm=re.sub(r"\W+"," ",line.casefold()).strip()
                if norm not in seen:
                    output.append(line); seen.add(norm); visited+=1
            for item in value.values():
                if isinstance(item,(dict,list)): walk(item)
        elif isinstance(value,list):
            for item in value: walk(item)
    if not raw or len(raw)>4_000_000: return ""
    try: payload=json.loads(raw)
    except (ValueError,TypeError): return ""
    walk(payload)
    return "\n\n".join(output)

def _json_data_text(soup, max_records=1200):
    """Turn JSON-LD and common client-app state blobs into readable records."""
    output=[]
    for tag in soup.find_all("script"):
        kind=(tag.get("type") or "").lower()
        identity=(tag.get("id") or "").lower()
        if kind not in {"application/ld+json","application/json"} and not re.search(r"next_data|nuxt|initial_state|app_state|__data",identity):
            continue
        parsed=_json_payload_text(tag.string or tag.get_text(),max_records=max_records)
        if parsed: output.append(parsed)
    return "\n\n".join(output)

def _embedded_links(soup, base_url):
    """Find page/document links stored in normal markup or embedded state."""
    links=[]
    for tag in soup.find_all(["a","area","iframe","source","link"],href=True):
        links.append(urljoin(base_url,tag.get("href","")))
    for tag in soup.find_all(["iframe","source","script"],src=True):
        links.append(urljoin(base_url,tag.get("src","")))
    for tag in soup.find_all("script"):
        raw=tag.string or tag.get_text()
        if not raw: continue
        for match in re.finditer(r"(?:href|url|file|src|download|document)\s*[:=]\s*['\"]([^'\"]{2,500})['\"]",raw,re.I):
            value=match.group(1).replace(r"\/","/")
            if value.startswith(("/","http://","https://")):
                links.append(urljoin(base_url,value))
    result=[]
    for link in links:
        try: candidate=canonical(link)
        except (ValueError,TypeError): continue
        suffix=Path(urlsplit(candidate).path).suffix.lower()
        if suffix and suffix not in {".php",".html",".htm",".asp",".aspx",".jsp",".pdf",".docx",".pptx",".xlsx",".csv",".json",".xml"}:
            continue
        if safe_url(candidate) and candidate not in result: result.append(candidate)
    return result

def _dynamic_script_urls(soup, base_url, visible_text, robots):
    """Select a bounded set of same-host data scripts for sparse/JS pages."""
    host=urlsplit(base_url).hostname
    page_hint=re.search(r"faculty|profile|staff|teacher|people|member|roster|club|event|placement|syllabus|course|academic|department|program|notice|fee|calendar|research|publication",base_url,re.I)
    sparse=len((visible_text or "").strip())<700
    urls=[]
    filename_hint=re.compile(r"faculty|fac_|profile|staff|teacher|people|member|roster|data|record|course|syllab|club|event|placement|notice|fee|academic|department|program|calendar|list|content",re.I)
    app_bundle=re.compile(r"(?:^|/)(?:main|app|index|page|bundle)(?:[._-][^/]*)?\.js(?:$|\?)",re.I)
    for tag in soup.find_all("script",src=True):
        candidate=canonical(urljoin(base_url,tag.get("src","")))
        name=urlsplit(candidate).path.rsplit("/",1)[-1]
        same_host=urlsplit(candidate).hostname==host
        relevant=filename_hint.search(name) or (sparse and page_hint and app_bundle.search(urlsplit(candidate).path))
        if same_host and relevant and safe_url(candidate) and (robots is None or robots.can_fetch(AGENT,candidate)):
            if candidate not in urls: urls.append(candidate)
        if len(urls)>=6: break
    return urls

def refresh_live_source(url: str, timeout: float = 12, query: str | None = None, related_limit: int = 0):
    """Fetch and parse an official URL; optionally follow query-relevant links."""
    url=canonical(url)
    if not safe_url(url): return 0
    try:
        with httpx.Client(timeout=timeout,follow_redirects=False,headers={"User-Agent":AGENT+"/1.0 (polite academic crawler)"}) as client:
            origin=f"{urlsplit(url).scheme}://{urlsplit(url).netloc}"
            robots=RobotFileParser(); robots.set_url(origin+"/robots.txt")
            try:
                policy=client.get(origin+"/robots.txt")
                if policy.status_code==404: robots.parse(["User-agent: *","Allow: /"])
                elif policy.status_code==200: robots.parse(policy.text.splitlines())
                else: return 0
            except httpx.HTTPError: return 0
            if not robots.can_fetch(AGENT,url): return 0
            response=client.get(url)
            if response.is_redirect:
                target=canonical(urljoin(url,response.headers.get("location","")))
                if not safe_url(target) or not robots.can_fetch(AGENT,target): return 0
                url=target; response=client.get(url)
            response.raise_for_status()
            content_type=response.headers.get("content-type","").lower()
            extension=Path(urlsplit(url).path).suffix.lower()
            if "pdf" in content_type or extension==".pdf" or response.content.startswith(b"%PDF"):
                from .ingest import ingest
                filename=urlsplit(url).path.rsplit("/",1)[-1] or "PCCOE document.pdf"
                return ingest(filename,response.content,url)
            if extension in {".docx",".pptx",".xlsx",".csv"} or "text/csv" in content_type:
                from .ingest import ingest
                filename=urlsplit(url).path.rsplit("/",1)[-1] or f"PCCOE document{extension}"
                return ingest(filename,response.content,url)
            if extension==".json" or "application/json" in content_type:
                text=_json_payload_text(response.text)
                if not text: return 0
                checked=datetime.now(timezone.utc).isoformat()
                title=urlsplit(url).path.rsplit("/",1)[-1] or "PCCOE structured data"
                return add_document(text,{"title":title,"url":url,"source_key":url,"source_id":url,"category":classify_category(url,title,text),"department":infer_department(url,title),"academic_year":extract_academic_year(url+" "+title+" "+text),"type":"JSON","extraction":"live-json","last_checked_at":checked,"last_successful_sync_at":checked})
            if "html" not in content_type and "xhtml" not in content_type: return 0
            soup=BeautifulSoup(response.content,"html.parser")
            title=soup.title.get_text(" ",strip=True) if soup.title else url
            description=soup.find("meta",attrs={"name":re.compile(r"^description$",re.I)})
            page_description=description.get("content","").strip() if description else ""
            visible_text=soup.get_text(" ",strip=True)
            related_links=[]
            if query and related_limit>0:
                query_terms=_terms(query)
                for anchor in soup.find_all("a",href=True):
                    label=anchor.get_text(" ",strip=True)
                    try: target=canonical(urljoin(url,anchor.get("href","")))
                    except (ValueError,TypeError): continue
                    if target==url or not safe_url(target): continue
                    target_suffix=Path(urlsplit(target).path).suffix.lower()
                    if target_suffix and target_suffix not in {".php",".html",".htm",".asp",".aspx",".jsp",".pdf",".docx",".pptx",".xlsx",".csv",".json",".xml"}: continue
                    link_terms=_terms(label+" "+urlsplit(target).path.replace("/"," ").replace("-"," ").replace("_"," "))
                    overlap=len(query_terms&link_terms)
                    if overlap:
                        related_links.append((overlap/len(query_terms),target))
                related_links=sorted(set(related_links),reverse=True)[:related_limit]
            robots_for_scripts=robots
            script_urls=_dynamic_script_urls(soup,url,visible_text,robots_for_scripts)
            dynamic_parts=[]
            structured=_json_data_text(soup)
            if structured: dynamic_parts.append("Structured JSON page data:\n"+structured)
            text=extract_html(soup,url)
            for script_url in dict.fromkeys(script_urls):
                try:
                    js=client.get(script_url)
                    if js.status_code==200 and len(js.content)<=4_000_000:
                        records=_js_profile_records(js.text)
                        if not records: records=_js_object_records(js.text)
                        if records: dynamic_parts.append("Structured JavaScript page data from "+script_url+":\n"+"\n\n".join(records))
                except httpx.HTTPError:
                    continue
            if dynamic_parts:
                text+="\n\nDynamically embedded page records:\n"+"\n\n".join(dynamic_parts)
            if page_description: text="Page description: "+page_description+"\n"+text
            if not text.strip(): return 0
            checked=datetime.now(timezone.utc).isoformat()
            added=add_document(text,{"title":title,"url":url,"source_key":url,"source_id":url,"category":classify_category(url,title,text),"department":infer_department(url,title),"academic_year":extract_academic_year(url+" "+title+" "+text),"type":"HTML","extraction":"live-http+javascript" if dynamic_parts else "live-http","last_checked_at":checked,"last_successful_sync_at":checked})
            # A results page often links to the syllabus/report/form that has
            # the answer. Follow only a small number of links whose anchor or
            # path overlaps this query; keep the same robots/domain checks.
            for _,target in related_links:
                try: added+=refresh_live_source(target,timeout=timeout)
                except Exception: continue
            return added
    except Exception:
        return 0

def canonical(url: str) -> str:
    p = urlsplit(url)
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path or "/", p.query, ""))

_CORE_ADMISSION_SEEDS = (
    "https://www.pccoepune.com/admission-home.php",
    "https://www.pccoepune.com/against-cap-admissions.php",
    "https://www.pccoepune.com/institute-level-admissions.php",
    "https://www.pccoepune.com/cut-off.php",
)

def _seed_urls():
    configured=[settings.college_url,*settings.seed_urls.split(","),*_CORE_ADMISSION_SEEDS]
    return list(dict.fromkeys(canonical(value.strip()) for value in configured if value.strip()))

def _retry_unindexed_faculty_rosters(seed_urls, seen):
    """Revisit seeded faculty PDFs that an older parser marked visited but never indexed."""
    retry=[]
    for url in seed_urls:
        filename=Path(urlsplit(url).path).name
        if url not in seen or not re.search(r"(?:faculty|staff)[-_ ]?list.*\.pdf$",filename,re.I):
            continue
        indexed=collection.get(where={"source_key":url},include=["metadatas"],limit=1)
        if not indexed.get("ids"):
            retry.append(url)
    return retry

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
    status_code,body=await _read_scoped_redirects(robots_url,client,1_000_000,15)
    parser = RobotFileParser()
    parser.set_url(robots_url)
    if status_code == 404:
        parser.parse(["User-agent: *", "Allow: /"])
        sitemaps = []
    elif status_code == 200:
        text = body.decode("utf-8", errors="replace")
        parser.parse(text.splitlines())
        sitemaps = [line.partition(":")[2].strip() for line in text.splitlines() if line.lower().startswith("sitemap:")]
    else:
        raise RuntimeError(f"robots.txt returned HTTP {status_code} for {origin}")
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

async def _read_scoped_redirects(url,client,limit,timeout_seconds):
    """Read one response while following only redirects to allowed PCCOE hosts."""
    current=url
    for redirect_count in range(6):
        if not safe_url(current):
            raise RuntimeError(f"Refusing out-of-scope redirect target: {current}")
        async with asyncio.timeout(timeout_seconds):
            async with client.stream("GET",current) as response:
                if response.is_redirect:
                    location=response.headers.get("location")
                    if not location:
                        raise RuntimeError(f"Redirect has no Location header: {current}")
                    target=canonical(urljoin(current,location))
                    if redirect_count==5:
                        raise RuntimeError(f"Too many redirects while fetching {url}")
                    if not safe_url(target):
                        raise RuntimeError(f"Refusing out-of-scope redirect target: {target}")
                    current=target
                    continue
                body=await read_limited(response,limit) if response.status_code==200 else b""
                return response.status_code,body
    raise RuntimeError(f"Too many redirects while fetching {url}")

async def add_sitemap(url: str, client: httpx.AsyncClient, queue: deque, visited: set, depth=0, errors=None):
    if depth > 4 or url in visited or not safe_url(url) or len(queue)>=settings.max_crawl_pages:
        return
    visited.add(url)
    try:
        status_code,body=await _read_scoped_redirects(url,client,4_000_000,30)
        if status_code!=200:
            if errors is not None: errors.append({"url":url,"error":f"HTTP {status_code}"})
            if status_code in (408,425,429) or status_code>=500:
                visited.discard(url)
            return
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
            if not safe_url(candidate) or not _crawlable_url(candidate):
                continue
            if candidate.lower().endswith((".xml", ".xml.gz")):
                await add_sitemap(candidate, client, queue, visited, depth + 1, errors)
            elif candidate not in queue:
                queue.append(candidate)
    except Exception as exc:
        if errors is not None: errors.append({"url":url,"error":str(exc)[:240]})
        visited.discard(url)
        return

async def crawl():
    global running, last_sync, _crawl_continuation_task, _stop_requested
    if running:
        return {"status": "already_running"}
    running = True
    _stop_requested=False
    started = datetime.now(timezone.utc)
    seen, sitemap_seen, robots_cache = set(), set(), {}
    checkpoint = _read_json(_checkpoint_path)
    resume = isinstance(checkpoint, dict) and checkpoint.get("status") in {"running", "interrupted", "partial", "stopped"} and bool(checkpoint.get("queue"))
    if resume:
        queue = deque(checkpoint.get("queue", []))
        seen = set(checkpoint.get("seen", []))
        sitemap_seen = set(checkpoint.get("sitemap_seen", []))
        seed_urls = _seed_urls()
        # Retry transient failures from an interrupted/partial run before the
        # remaining frontier, but put every unprocessed configured seed first,
        # even if a prior crawl already had that URL queued.
        # Keep permanent HTTP errors and robots skips out.
        retry_urls = _retryable_urls(checkpoint.get("errors", []))
        try:
            missing_roster_urls=_retry_unindexed_faculty_rosters(seed_urls,seen)
        except Exception:
            running=False
            raise
        queue = _prioritize_unseen_seeds(queue, seed_urls, seen, retry_urls+missing_roster_urls)
        seen.difference_update(retry_urls+missing_roster_urls)
    else:
        retry_urls = _retryable_urls(last_sync.get("errors", [])) if last_sync.get("status") in {"partial", "interrupted"} else []
        seed_urls = _seed_urls()
        queue = deque(dict.fromkeys(retry_urls+seed_urls))
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
    retry_counts = checkpoint.get("retry_counts", {}) if resume else {}
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
                 "retry_counts": retry_counts,
                 "started_at": started.isoformat(), "errors": failures[-100:], "sitemap_error_details": sitemap_errors[-50:]}
        # The detailed queue snapshot can be large. Persist it every 25 pages;
        # the compact public status is still refreshed after each page.
        if force or status != "running" or processed - last_checkpoint_processed >= 25:
            _write_json(_checkpoint_path, state)
            last_checkpoint_processed = processed
        public = {k: v for k, v in state.items() if k not in {"queue", "seen", "sitemap_seen", "retry_counts"}}
        pending_hosts=Counter(urlsplit(url).hostname or "unknown" for url in queue)
        pending_types=Counter("PDF" if Path(urlsplit(url).path).suffix.lower()==".pdf" else "HTML/data" for url in queue)
        public.update({"status": status, "discovered": len(seen), "queued": len(queue),
                       "pending_hosts":dict(sorted(pending_hosts.items(),key=lambda row:(-row[1],row[0]))),
                       "pending_types":dict(pending_types),"sitemap_errors": len(sitemap_errors)})
        _write_json(_status_path, public)
    def record_failure(url,exc,terminal=False):
        nonlocal failed
        failed += 1
        error=_error_summary(exc)
        failure={"url":url,"error":error}
        rate_limited=bool(re.search(r"429|RateLimitError|RESOURCE_EXHAUSTED|quota|rate.?limit",error,re.I))
        if rate_limited:
            failure["retry_after"]=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
        failures.append(failure)
        attempts=int(retry_counts.get(url,0))
        if not terminal and not rate_limited and attempts<2:
            retry_counts[url]=attempts+1
            seen.discard(url)
            if url not in queue: queue.append(url)
        else:
            retry_counts.pop(url,None)
    try:
        # Give slow, multi-megabyte official PDFs enough time to download while
        # preserving bounded connection/read timeouts and a single polite
        # request stream. A short default 20s read timeout was discarding many
        # otherwise valid PCCOE reports as blank TimeoutError failures.
        timeout=httpx.Timeout(connect=15.0,read=75.0,write=30.0,pool=15.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, headers={"User-Agent": AGENT + "/1.0 (polite academic crawler)"}) as client:
            parser, sitemaps = await load_robots(settings.college_url, client, robots_cache)
            for sitemap in sitemaps:
                await add_sitemap(sitemap, client, queue, sitemap_seen, errors=sitemap_errors)
            checkpoint_state(force=True)
            while queue and len(seen) - starting_seen_count < settings.max_crawl_pages and not _stop_requested:
                while _active_queries:
                    await asyncio.sleep(0.05)
                url = canonical(queue.popleft())
                if url in seen or not safe_url(url) or not _crawlable_url(url):
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
                    record_failure(url,exc)
                    continue
                if not parser.can_fetch(AGENT, url):
                    skipped_robots += 1
                    robots_blocked += 1
                    skipped_details.append({"url":url,"reason":"Disallowed by host robots.txt"})
                    continue
                seen.add(url)
                try:
                    async with client.stream("GET", url) as response:
                        if response.is_redirect:
                            target = canonical(urljoin(url, response.headers.get("location", "")))
                            if safe_url(target) and _crawlable_url(target) and target not in queue:
                                queue.append(target)
                            continue
                        response.raise_for_status()
                        # A valid 8–10 MB official faculty roster can take
                        # several minutes from the source host. A fixed 90s
                        # wall-clock cutoff repeatedly discarded it despite
                        # successful streaming. Keep inactivity timeouts on
                        # each read, while deriving a bounded transfer budget
                        # from the advertised size (or file type when absent).
                        try:
                            content_length=int(response.headers.get("content-length", "0"))
                        except ValueError:
                            content_length=0
                        is_document=Path(urlsplit(url).path).suffix.lower() in {".pdf",".docx",".pptx",".xlsx",".csv"}
                        rate_budget=(content_length/(15*1024)*1.5) if content_length else (900 if is_document else 90)
                        transfer_timeout=min(3600,max(90,rate_budget if is_document else 90))
                        async with asyncio.timeout(transfer_timeout):
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
                    elif extension==".json" or "application/json" in content_type:
                        text=_json_payload_text(body.decode("utf-8",errors="replace"))
                        if text:
                            checked_at=datetime.now(timezone.utc).isoformat()
                            filename=urlsplit(url).path.rsplit("/",1)[-1] or "PCCOE structured data"
                            changed=await asyncio.to_thread(add_document,text,{"title":filename,"url":url,"category":classify_category(url,filename,text),"department":infer_department(url,filename),"academic_year":extract_academic_year(url+" "+filename+" "+text),"type":"JSON","extraction":"json-data","last_checked_at":checked_at,"last_successful_sync_at":checked_at})
                    elif "html" in content_type or "xhtml" in content_type:
                        soup = BeautifulSoup(body, "html.parser")
                        title = soup.title.get_text(" ", strip=True) if soup.title else url
                        description=soup.find("meta",attrs={"name":re.compile(r"^description$",re.I)})
                        page_description=description.get("content","").strip() if description else ""
                        # Save links before removing navigation/footer markup for clean indexing.
                        # Department websites and syllabus PDFs are commonly linked from menus.
                        visible_text=soup.get_text(" ",strip=True)
                        child_links = [canonical(urljoin(url, anchor["href"])) for anchor in soup.find_all("a", href=True)]
                        child_links.extend(_embedded_links(soup,url))
                        dynamic_parts=[]
                        structured=_json_data_text(soup)
                        if structured: dynamic_parts.append("Structured JSON page data:\n"+structured)
                        script_urls=_dynamic_script_urls(soup,url,visible_text,parser)
                        for script_url in script_urls:
                            try:
                                if not parser.can_fetch(AGENT,script_url): continue
                                async with asyncio.timeout(12):
                                    async with client.stream("GET",script_url) as script_response:
                                        if script_response.status_code!=200: continue
                                        script_body=await read_limited(script_response,4_000_000)
                                script_text=script_body.decode("utf-8",errors="replace")
                                records=_js_profile_records(script_text)
                                if not records: records=_js_object_records(script_text)
                                if records: dynamic_parts.append("Structured JavaScript page data from "+script_url+":\n"+"\n\n".join(records))
                            except Exception as exc:
                                # A data script is auxiliary: retain the page
                                # even when one JavaScript asset is unavailable.
                                continue
                        text = extract_html(soup, url)
                        if dynamic_parts:
                            text+="\n\nDynamically embedded page records:\n"+"\n\n".join(dynamic_parts)
                        if page_description: text="Page description: "+page_description+"\n"+text
                        if text:
                            checked_at=datetime.now(timezone.utc).isoformat()
                            changed = await asyncio.to_thread(add_document, text, {"title": title, "url": url, "category": classify_category(url,title,text), "department": infer_department(url,title), "academic_year": extract_academic_year(url+" "+title+" "+text), "type": "HTML", "extraction":"http+structured-js" if dynamic_parts else "http-html", "last_checked_at": checked_at, "last_successful_sync_at": checked_at})
                        for child in child_links:
                            if safe_url(child) and _crawlable_url(child) and child not in seen and child not in queue:
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
                    retry_counts.pop(url,None)
                    host = urlsplit(url).hostname or "unknown"
                    host_counts[host] = host_counts.get(host, 0) + 1
                    if changed:
                        updated += 1
                    else:
                        unchanged += 1
                except httpx.HTTPStatusError as exc:
                    status_code=exc.response.status_code
                    terminal=status_code<500 and status_code not in (408,425,429)
                    if status_code in (404, 410):
                        removed += await asyncio.to_thread(remove_document, url)
                    record_failure(url,RuntimeError(f"HTTP {status_code}"),terminal=terminal)
                except Exception as exc:
                    record_failure(url,exc)
                last_sync = {"status": "running", "processed": processed, "updated_sources": updated, "unchanged_sources": unchanged, "removed_chunks": removed, "failed": failed, "skipped": skipped_robots, "robots_blocked": robots_blocked, "unsupported": unsupported, "skipped_details": skipped_details[-50:], "sitemap_errors": len(sitemap_errors), "discovered": len(seen), "hosts": host_counts.copy(), "started_at": started.isoformat()}
                checkpoint_state()
                await asyncio.sleep(0.25)
    except Exception as exc:
        failed += 1
        failures.append({"url": settings.college_url, "error": _error_summary(exc)})
    finally:
        running = False
        duration = (datetime.now(timezone.utc) - started).total_seconds()
        capped = bool(queue) and len(seen) - starting_seen_count >= settings.max_crawl_pages and not _stop_requested
        stopped=_stop_requested and bool(queue)
        status = "stopped" if stopped else ("failed" if failed and not processed else ("partial" if failed or capped else "complete"))
        last_sync = {"status": status, "processed": processed, "updated_sources": updated, "unchanged_sources": unchanged, "removed_chunks": removed, "failed": failed, "skipped": skipped_robots, "robots_blocked": robots_blocked, "unsupported": unsupported, "skipped_details": skipped_details[-50:], "sitemap_errors": len(sitemap_errors), "discovered": len(seen), "hosts": host_counts, "started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(), "duration_seconds": round(duration, 2), "capped": capped, "queued": len(queue), "errors": failures[-100:], "sitemap_error_details": sitemap_errors[-50:]}
        checkpoint_state(status)
        if not queue:
            try: _checkpoint_path.unlink(missing_ok=True)
            except OSError: pass
        _stop_requested=False
    if capped and (_crawl_continuation_task is None or _crawl_continuation_task.done()):
        _crawl_continuation_task=asyncio.create_task(_continue_capped_crawl())
    return last_sync

async def _continue_capped_crawl():
    global _crawl_continuation_task
    await asyncio.sleep(2)
    _crawl_continuation_task=None
    await crawl()

if __name__ == "__main__":
    # Supports a recoverable standalone refresh for deployments without the
    # admin dashboard. The same persistent status/checkpoint files are used.
    asyncio.run(crawl())
