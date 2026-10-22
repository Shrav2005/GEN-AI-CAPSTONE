import hashlib, os, sqlite3
import re
from pathlib import Path
from contextlib import contextmanager
import chromadb
from .config import settings

os.makedirs(settings.chroma_path, exist_ok=True)
client = chromadb.PersistentClient(path=settings.chroma_path)
collection = client.get_or_create_collection("pccoe_sources", metadata={"hnsw:space":"cosine"})

@contextmanager
def _lexical_db():
    """Persistent FTS index for exact terms alongside Chroma's semantic index."""
    path=Path(settings.data_path)
    path.mkdir(parents=True,exist_ok=True)
    conn=sqlite3.connect(str(path/"pccoe_lexical.sqlite3"),timeout=15)
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS source_chunks USING fts5(doc_id UNINDEXED, body, title, category, department, url, academic_year, tokenize='unicode61')")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def _normalized_terms(value):
    return " ".join(sorted(_terms(value)))

def _fts_upsert(doc_id,document,metadata):
    try:
        with _lexical_db() as conn:
            conn.execute("DELETE FROM source_chunks WHERE doc_id=?",(doc_id,))
            conn.execute("INSERT INTO source_chunks(doc_id,body,title,category,department,url,academic_year) VALUES(?,?,?,?,?,?,?)",(
                doc_id,_normalized_terms(document),_normalized_terms(metadata.get("title","")),
                _normalized_terms(metadata.get("category","")),_normalized_terms(metadata.get("department","")),
                _normalized_terms(metadata.get("url","")),_normalized_terms(metadata.get("academic_year") or "")))
    except (sqlite3.Error, OSError):
        # Chroma remains the source of truth; lexical indexing is an additional
        # retrieval signal and should never interrupt a successful crawl.
        return

def _fts_delete(doc_ids):
    if not doc_ids: return
    try:
        with _lexical_db() as conn:
            conn.executemany("DELETE FROM source_chunks WHERE doc_id=?",[(value,) for value in doc_ids])
    except (sqlite3.Error,OSError):
        return

def _fts_search(query,limit):
    terms=sorted(_terms(query))
    if not terms: return []
    expression=" OR ".join('"'+term.replace('"','')+'"' for term in terms)
    try:
        with _lexical_db() as conn:
            count=conn.execute("SELECT count(*) FROM source_chunks").fetchone()[0]
            vector_count=collection.count()
            if count!=vector_count:
                # Reconcile legacy Chroma records, including an existing index
                # that predates this FTS database. Batch reads avoid loading the
                # whole corpus into memory during a first query.
                conn.execute("DELETE FROM source_chunks")
                for offset in range(0,vector_count,1000):
                    stored=collection.get(limit=1000,offset=offset,include=["documents","metadatas"])
                    conn.executemany("INSERT INTO source_chunks(doc_id,body,title,category,department,url,academic_year) VALUES(?,?,?,?,?,?,?)",[
                        (doc_id,_normalized_terms(doc),_normalized_terms(meta.get("title","")),_normalized_terms(meta.get("category","")),_normalized_terms(meta.get("department","")),_normalized_terms(meta.get("url","")),_normalized_terms(meta.get("academic_year") or ""))
                        for doc_id,doc,meta in zip(stored.get("ids",[]),stored.get("documents",[]),stored.get("metadatas",[]))])
            rows=conn.execute("SELECT doc_id,bm25(source_chunks,0,1,4,2,1,2,1) AS rank FROM source_chunks WHERE source_chunks MATCH ? ORDER BY rank LIMIT ?",(expression,limit)).fetchall()
            return rows
    except (sqlite3.Error,OSError):
        return []

_STOPWORDS = {"what", "where", "when", "which", "who", "does", "is", "are", "the", "and", "for", "from", "about", "tell", "show", "give", "find", "can", "you", "me", "college", "information", "please", "with", "that", "this", "have", "has", "how", "much", "many", "document", "documents", "available", "list", "get", "provide", "pccoe", "pccoepune", "per", "of", "in", "on", "at", "to", "a", "an", "my", "your", "their", "its", "all", "year"}

_SYNONYMS = {"fourth": "final", "4th": "final", "curriculum": "syllabus", "curricula": "syllabus", "subjects": "subject", "systems": "system", "3rd": "third", "ty": "third", "te": "third", "2nd": "second", "sy": "second", "se": "second", "1st": "first", "fy": "first", "fe": "first", "fees": "fee", "tuition": "fee", "charges": "fee", "cost": "fee", "costs": "fee", "price": "fee", "prices": "fee", "engg": "engineering", "curriculums": "syllabus", "courses": "course", "branches": "branch", "departments": "department", "semesters": "semester", "programmes": "program", "programs": "program"}
_SHORT_TERMS = {"ai", "it", "cs", "cse", "entc", "gpa", "cgpa", "ml", "ds", "fe", "me", "be", "hod", "mtech", "btech", "ty", "te", "sy", "se", "fy", "3rd", "2nd", "1st"}

_QUERY_TOPICS=(
    ("Tuition and fees",r"\b(fees?|tuition|cost|charges?|payment|refund)\b"),
    ("Admissions",r"\b(admission|eligibility|cap|entrance|cut.?off|application)\b"),
    ("Scholarships and financial aid",r"\b(scholarship|tfws|ebc|freeship|financial aid)\b"),
    ("Syllabus and curriculum",r"\b(syllabus|curriculum|course structure|course content|units?)\b"),
    ("Academic calendars and schedules",r"\b(calendar|timetable|schedule|semester start)\b"),
    ("Examinations and assessment",r"\b(exam|examination|assessment|revaluation|result)\b"),
    ("Academic regulations and attendance",r"\b(attendance|progression|regulation|promotion|backlog)\b"),
    ("Placements and careers",r"\b(placement|internship|package|salary|recruitment|career)\b"),
    ("Faculty and official contacts",r"\b(faculty|staff|professor|teacher|hod|head of department|email|phone|contact)\b"),
    ("Hostel and accommodation",r"\b(hostel|accommodation|mess|warden)\b"),
    ("Campus facilities",r"\b(facility|facilities|library|laboratory|labs?|wifi|medical|infrastructure)\b"),
    ("Notices and circulars",r"\b(notice|circular|announcement|deadline|last date)\b"),
    ("Student welfare and safety",r"\b(grievance|anti.?ragging|counselling|welfare|safety|complaint)\b"),
    ("Programs and departments",r"\b(programs?|courses?|branches|departments?|ug|pg|undergraduate|postgraduate|intake)\b"),
    ("Research, projects and innovation",r"\b(research|project|incubation|innovation|patent)\b"),
    ("Alumni",r"\b(alumni|alumnus|alumna)\b"),
    ("Student activities and events",r"\b(clubs?|chapters?|societies|student organizations?|events?)\b"),
    ("ERP and student services",r"\b(erp|student portal|login|portal)\b"),
    ("Transportation and campus directions",r"\b(address|directions?|campus|visitor|transport|bus route)\b"),
)

def _query_topics(query):
    return {category for category,pattern in _QUERY_TOPICS if re.search(pattern,query,re.I)}

def _terms(value):
    words = set()
    for word in re.findall(r"[a-zA-Z0-9]+", value.lower()):
        word = _SYNONYMS.get(word, word)
        if (len(word) <= 3 and word not in _SHORT_TERMS and word not in {"third","second","first"}) or word in _STOPWORDS:
            continue
        if word.endswith("ies") and len(word) > 5:
            word = word[:-3] + "y"
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 5:
            word = word[:-1]
        words.add(word)
    return words

def _chunks(text, size=1150, overlap=160):
    """Chunk by paragraph/sentence boundaries so course units and table rows stay readable."""
    paragraphs = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    if not paragraphs:
        paragraphs = [" ".join(text.split())]
    units = []
    for paragraph in paragraphs:
        if len(paragraph) <= size:
            units.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?])\s+|(?<=\|)\s+", paragraph)
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= size:
                current = (current + " " + sentence).strip()
                continue
            if current:
                units.append(current)
            while len(sentence) > size:
                cut = sentence.rfind(" ", 0, size)
                cut = cut if cut > size // 2 else size
                units.append(sentence[:cut].strip())
                sentence = sentence[cut:].strip()
            current = sentence
        if current:
            units.append(current)
    chunks, current = [], ""
    for unit in units:
        if current and len(current) + len(unit) + 1 > size:
            chunks.append(current)
            carry=max(0,min(overlap,size-len(unit)-1))
            current = (current[-carry:] + " " + unit).strip() if carry else unit
        else:
            current = (current + "\n" + unit).strip()
    if current:
        chunks.append(current)
    return chunks

def add_document(text, metadata):
    """Index one source incrementally; replace stale chunks only after changed chunks upsert."""
    text="\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())
    if not text: return 0
    chunks=_chunks(text)
    url=metadata.get("url") or "local://"+metadata.get("title","untitled")
    source_key=metadata.get("source_key") or url
    source_id=metadata.get("source_id") or source_key
    existing=collection.get(where={"source_id":source_id},include=["metadatas"])
    # One-time compatibility for legacy records only; other page records from this
    # same PDF must never be mistaken for the current page's chunks.
    if not existing.get("ids"):
        legacy=collection.get(where={"url":url},include=["metadatas"])
        pairs=[(doc_id,meta) for doc_id,meta in zip(legacy.get("ids",[]),legacy.get("metadatas",[])) if not meta.get("source_id")]
        existing={"ids":[pair[0] for pair in pairs],"metadatas":[pair[1] for pair in pairs]}
    old={m.get("chunk"): (doc_id,m.get("chunk_hash")) for doc_id,m in zip(existing.get("ids",[]),existing.get("metadatas",[]))}
    changed_docs=[]; changed_metas=[]; changed_ids=[]; metadata_ids=[]; metadata_values=[]; metadata_documents=[]
    document_hash=hashlib.sha256(text.encode()).hexdigest()
    for i,chunk in enumerate(chunks):
        chunk_hash=hashlib.sha256(chunk.encode()).hexdigest()
        stable_id=hashlib.sha256((source_id+"::"+str(i)).encode()).hexdigest()
        metadatum={**metadata,"url":url,"source_key":source_key,"source_id":source_id,"chunk":i,"chunk_hash":chunk_hash,"content_hash":document_hash,"source_version":metadata.get("source_version") or document_hash,"chunk_count":len(chunks)}
        if i in old and old[i][1]==chunk_hash and old[i][0]==stable_id:
            metadata_ids.append(stable_id); metadata_values.append(metadatum); metadata_documents.append(chunk)
            continue
        changed_ids.append(stable_id)
        changed_docs.append(chunk)
        changed_metas.append(metadatum)
    if changed_ids:
        collection.upsert(ids=changed_ids,documents=changed_docs,metadatas=changed_metas)
        for doc_id,document,metadatum in zip(changed_ids,changed_docs,changed_metas): _fts_upsert(doc_id,document,metadatum)
    if metadata_ids:
        collection.update(ids=metadata_ids,metadatas=metadata_values)
        for doc_id,document,metadatum in zip(metadata_ids,metadata_documents,metadata_values): _fts_upsert(doc_id,document,metadatum)
    stable_ids={i:hashlib.sha256((source_id+"::"+str(i)).encode()).hexdigest() for i in range(len(chunks))}
    stale=[doc_id for index,(doc_id,_) in old.items() if index>=len(chunks) or doc_id!=stable_ids[index]]
    if stale:
        collection.delete(ids=stale); _fts_delete(stale)
    return len(changed_ids)

def remove_document(source_url):
    ids=set()
    for field in ("url", "source_key"):
        existing=collection.get(where={field:source_url},include=[])
        ids.update(existing.get("ids",[]))
    if ids: collection.delete(ids=list(ids)); _fts_delete(ids)
    return len(ids)

def remove_source_id(source_id):
    existing=collection.get(where={"source_id":source_id},include=[])
    ids=existing.get("ids",[])
    if ids: collection.delete(ids=ids); _fts_delete(ids)
    return len(ids)

def search(query, limit=5):
    if collection.count()==0: return []
    # Dense retrieval is good at paraphrases, while lexical matching is better
    # for exact course codes, dates, amounts and document titles. Retrieve a
    # broad candidate pool, then fuse both signals instead of trusting either.
    dense_result=collection.query(query_texts=[query],n_results=min(max(limit * 40, 200),collection.count()),include=["documents","metadatas","distances"])
    dense_rows={}
    for index,(doc_id,doc,meta,distance) in enumerate(zip(dense_result["ids"][0],dense_result["documents"][0],dense_result["metadatas"][0],dense_result["distances"][0])):
        dense_rows[doc_id]={"text":doc,"metadata":meta,"score":1-float(distance),"dense_rank":index+1}
    lexical=_fts_search(query,min(max(limit*40,200),collection.count()))
    lexical_ids=[doc_id for doc_id,_ in lexical]
    lexical_ranks={doc_id:index+1 for index,(doc_id,_) in enumerate(lexical)}
    missing=[doc_id for doc_id in lexical_ids if doc_id not in dense_rows]
    if missing:
        lexical_docs=collection.get(ids=missing,include=["documents","metadatas"])
        for doc_id,doc,meta in zip(lexical_docs.get("ids",[]),lexical_docs.get("documents",[]),lexical_docs.get("metadatas",[])):
            dense_rows[doc_id]={"text":doc,"metadata":meta,"score":0.0,"dense_rank":None}
    rows=[]
    terms=_terms(query)
    expected_topics=_query_topics(query)
    query_phrase=" ".join(re.findall(r"[a-zA-Z0-9]+",query.lower()))
    academic_year=re.search(r"\b20\d{2}\s*[-–/]\s*(?:\d{2}|20\d{2})\b",query)
    for doc_id,row in dense_rows.items():
        doc=row["text"]; meta=row["metadata"]
        title_terms=_terms(str(meta.get("title",""))+" "+str(meta.get("url","")))
        evidence_terms=_terms(doc) | title_terms | _terms(str(meta.get("category",""))+" "+str(meta.get("department","")))
        matched=terms & evidence_terms
        overlap=len(matched)
        row["lexical_overlap"]=overlap
        coverage=overlap / len(terms) if terms else 0
        title_coverage=len(terms & title_terms)/len(terms) if terms else 0
        row["coverage"]=coverage
        content_lower=re.sub(r"[^a-z0-9 ]+"," ",doc.lower())
        phrase_match=bool(query_phrase and len(query_phrase)>5 and query_phrase in content_lower)
        dense=max(0.0,min(1.0,row["score"]))
        lexical_rank=lexical_ranks.get(doc_id)
        lexical_signal=(1/(1+0.04*(lexical_rank-1))) if lexical_rank else 0
        row["rank_score"]=(0.48*dense)+(0.30*coverage)+(0.15*title_coverage)+(0.04*lexical_signal)+(0.03*phrase_match)
        category=str(meta.get("category", ""))
        if expected_topics:
            if category in expected_topics:
                row["rank_score"]+=0.18
            else:
                row["rank_score"]*=0.68
        row["lexical_rank"]=lexical_rank
        if academic_year:
            expected=re.sub(r"\s+","",academic_year.group(0)).replace("–","-").replace("/","-")
            indexed_year=re.sub(r"\s+","",str(meta.get("academic_year") or "")).replace("–","-").replace("/","-")
            if indexed_year and expected not in indexed_year and indexed_year not in expected:
                row["rank_score"]*=0.72
        rows.append(row)
    query_lower=query.lower()
    department_hints=((r"\bcomputer\b|\bcomp\b","computer engineering"),(r"\binformation technology\b|\bit\b","information technology"),(r"\bentc\b","electronics and telecommunication"),(r"\bcivil\b","civil engineering"),(r"\bmechanical\b|\bmech\b","mechanical engineering"),(r"\bai\s*[/&-]?\s*ml\b|\baiml\b|\bartificial intelligence and machine learning\b","artificial intelligence and machine learning"),(r"\bvlsi\b|\belectronics engineering\b","electronics engineering"))
    expected_department=None
    for pattern,department in department_hints:
        if re.search(pattern,query_lower):
            expected_department=department; break
    if expected_department:
        department_rows=[row for row in rows if expected_department in row["metadata"].get("department","").lower()]
        # Institute-wide tables (e.g. the official Heads & Deans roster) carry
        # useful department-specific data despite institution-wide metadata.
        institution_rows=[row for row in rows if row["metadata"].get("department","").lower() in {"institution-wide",""} and expected_department in row["text"].lower()]
        if department_rows:
            exact_rows=[row for row in department_rows if row["metadata"].get("department","").lower()==expected_department]
            rows=(exact_rows or department_rows)+institution_rows
        elif institution_rows:
            rows=institution_rows
    return sorted(rows,key=lambda row:row["rank_score"],reverse=True)[:limit]
