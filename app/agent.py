"""One retrieval-grounded PCCOE knowledge agent (no model training required)."""
import json
import logging
import re
from functools import lru_cache
import httpx
from .config import settings
from .store import collection, search, _terms
from .taxonomy import DEPARTMENT_HOSTS, resolve_department, infer_department

logger=logging.getLogger(__name__)

_ROMAN_HINDI_MARATHI = {
    "kaun":"who", "kon":"who", "kya":"what", "kay":"what", "konte":"which", "konti":"which",
    "konta":"which", "kitna":"how much", "kitni":"how much", "kiti":"how much", "kitne":"how many",
    "kaise":"how", "kasa":"how", "kashi":"how", "kab":"when", "kadhi":"when", "kahan":"where",
    "kuthe":"where", "kuthे":"where", "kyun":"why", "ka":"of", "ki":"of", "ke":"of", "cha":"of",
    "chi":"of", "che":"of", "mein":"in", "madhye":"in", "me":"in", "madhe":"in", "par":"on",
    "ani":"and", "aur":"and", "ya":"this", "ye":"this", "is":"this", "hai":"is", "hain":"are",
    "aahe":"is", "aahet":"are", "ahe":"is", "ahet":"are", "milega":"get", "milega":"get",
    "kaise":"how", "batao":"tell", "bata":"tell", "sanga":"tell", "sang":"tell", "dya":"give",
    "do":"give", "chahiye":"need", "pahije":"need", "sathi":"for", "suru":"start", "shuru":"start",
    "hoga":"will", "honge":"will", "kare":"do", "karu":"do", "karaycha":"do", "kya":"what",
}
_LANGUAGE_MARKERS=set(_ROMAN_HINDI_MARATHI)|{"aahe","aahet","madhye","madhye","sathi","kagapatre","kagadpatre"}
_SEARCH_FILLER={
    "a","an","the","is","are","was","were","be","been","being","do","does","did",
    "can","could","would","should","will","please","kindly","tell","show","give","find",
    "me","my","we","you","i","us","our","your","what","which","who","whom","when",
    "where","why","how","many","much","some","any","about","from","with","for","of",
    "in","on","at","to","by","and","or","as","this","that","there","pccoe",
    "pccoepune","college","institute",
}

_ROLE_EVIDENCE = re.compile(r"\bHOD\b|head\s+of\s+(?:the\s+)?department|department\s+head",re.I)
_NAMED_ROLE_PERSON = re.compile(
    r"\b(?:Dr|Prof|Professor)\.?\s*(?:Dr\.?\s*)?(?:(?:[A-Z]\.?(?:\s+|$)){0,3}[A-Z][A-Za-zÀ-ÿ.'’\-]*(?:\s+[A-Z][A-Za-zÀ-ÿ.'’\-]*){0,2})\b"
)

def _has_department_role_line(text,department):
    if not department: return False
    aliases={department.casefold()}
    aliases.update(alias.casefold() for alias,label in DEPARTMENT_HOSTS.items() if label.casefold()==department.casefold())
    aliases={alias for alias in aliases if len(alias)>2}
    for line in (text or "").splitlines():
        if (_ROLE_EVIDENCE.search(line)
                and any(re.search(r"(?<![a-z])"+re.escape(alias)+r"(?![a-z])",line,re.I) for alias in aliases)
                and _NAMED_ROLE_PERSON.search(line)):
            return True
    return False

def _has_named_role(text):
    lines=[line.strip() for line in (text or "").splitlines() if line.strip()]
    for index,line in enumerate(lines):
        if not _ROLE_EVIDENCE.search(line): continue
        nearby=" ".join(lines[max(0,index-1):min(len(lines),index+2)])
        if _NAMED_ROLE_PERSON.search(nearby): return True
    return False

def _authoritative_role_source(hit):
    meta=hit.get("metadata",{})
    identity=(str(meta.get("title", "")+" "+meta.get("url", ""))).casefold()
    return any(term in identity for term in (
        "academic-council", "heads-and-deans", "faculty-profile", "faculty profile",
        "faculty.php", "faculty_staff", "contact", "hod_desk", "hod-desk",
        "department-head", "department head",
    ))

def _keyword_query(question):
    """Make a second, compact search view without assuming a fixed question type."""
    tokens=re.findall(r"[A-Za-z0-9][A-Za-z0-9.&/'’+-]*|[\u0900-\u097f]+",question)
    kept=[]
    for token in tokens:
        normalized=token.casefold().strip(".,!?;:")
        if len(normalized)<=1 or normalized in _SEARCH_FILLER:
            continue
        if normalized not in {item.casefold() for item in kept}:
            kept.append(token)
    return " ".join(kept)

_INTENT_EXPANSIONS=(
    (r"\b(clubs?|chapters?|societies|student organizations?)\b","official department student association technical clubs professional chapters student teams organizations activities coordinators"),
    (r"\b(fee|tuition|cost|charge|payment)\b","official fee structure annual tuition amount category academic year"),
    (r"\b(admission|eligibility|cap|entrance|cutoff)\b","official admission process eligibility CAP round required documents current academic year"),
    (r"\b(syllabus|curriculum|subject|course|unit|semester)\b","official department program academic year regulation course code subject units teaching scheme syllabus PDF"),
    (r"\b(hod|head of department|faculty|staff|professor|email|phone|contact)\b","official department head faculty profile designation email phone roster"),
    (r"\b(placement|internship|package|salary|recruitment)\b","official department placement report batch students placed eligible highest average salary internship"),
    (r"\b(exam|examination|timetable|assessment|calendar|schedule)\b","official examination timetable academic calendar program semester exam session published date"),
    (r"\b(scholarship|tfws|ebc|freeship|financial aid)\b","official scholarship eligibility required documents deadline financial aid"),
    (r"\b(notice|circular|announcement|deadline|last date)\b","latest official notice circular publication date deadline department"),
    (r"\b(hostel|accommodation|mess)\b","official hostel admission fees rules warden student accommodation"),
    (r"\b(address|direction|campus|visitor|transport|bus)\b","official campus address directions visitor instructions transport"),
)

def _fallback_english_query(question):
    tokens=re.findall(r"[A-Za-z0-9]+|[\u0900-\u097f]+",question)
    mapped=[]
    for token in tokens:
        lower=token.lower()
        if re.fullmatch(r"[\u0900-\u097f]+",token):
            mapped.append(token)
        elif lower in _ROMAN_HINDI_MARATHI:
            mapped.append(_ROMAN_HINDI_MARATHI[lower])
        else:
            mapped.append(token)
    return " ".join(mapped)

def _looks_multilingual(question):
    # Cover scripts beyond Devanagari (Arabic, Cyrillic, CJK, etc.) as well as
    # accented Latin languages. Romanized Hindi/Marathi is handled below.
    if any(character.isalpha() and ord(character)>127 for character in question): return True
    words={word.lower() for word in re.findall(r"[a-zA-Z]+",question)}
    return len(words&_LANGUAGE_MARKERS)>=2 or bool(words&{"kitni","kitna","kiti","kaise","kasa","madhye","aahe","aahet","sathi"})

@lru_cache(maxsize=256)
def _model_query_plan(question):
    """Plan PCCOE searches and request missing scope; never produce evidence."""
    provider=settings.active_llm_provider()
    empty_plan=((),None)
    if provider not in {"gemini","groq","ollama"}: return empty_plan
    prompt=("You are the query-planning step of a PCCOE-only knowledge assistant. "
            "Decide whether the request lacks program, department, year, or route context that materially changes the answer. "
            "If so, ask one short clarification in the user's language; otherwise set clarification to null. "
            "Also rewrite the request into up to three short, distinct searches for PCCOE's official websites. "
            "Preserve every named department, person, program, year and number exactly. "
            "Searches are retrieval hints only. Do not answer, add facts, or use non-PCCOE sources. "
            'Return JSON only: {"queries":["..."],"clarification":null}. Question: '+question)
    try:
        if provider=="gemini" and settings.gemini_api_key:
            response=httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{settings.gemini_model}:generateContent",headers={"x-goog-api-key":settings.gemini_api_key},json={"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"temperature":0}},timeout=14)
            response.raise_for_status()
            raw=response.json()["candidates"][0]["content"]["parts"][0]["text"]
        elif provider=="groq" and settings.groq_api_key:
            response=httpx.post("https://api.groq.com/openai/v1/chat/completions",headers={"Authorization":"Bearer "+settings.groq_api_key},json={"model":settings.effective_groq_model(),"messages":[{"role":"user","content":prompt}],"temperature":0,"response_format":{"type":"json_object"}},timeout=14)
            response.raise_for_status()
            raw=response.json()["choices"][0]["message"]["content"]
        elif provider=="ollama":
            response=httpx.post(settings.ollama_base_url+"/api/generate",json={"model":settings.ollama_model,"prompt":prompt,"stream":False,"format":"json","options":{"temperature":0}},timeout=14)
            response.raise_for_status()
            raw=response.json().get("response","")
        else:
            return empty_plan
        parsed=json.loads(raw)
        queries=parsed.get("queries",[]) if isinstance(parsed,dict) else []
        variants=tuple(" ".join(query.split())[:240] for query in queries[:3] if isinstance(query,str) and query.strip())
        clarification=parsed.get("clarification") if isinstance(parsed,dict) else None
        if not isinstance(clarification,str) or not clarification.strip():
            clarification=None
        elif len(clarification)>300:
            clarification=clarification[:300].rsplit(" ",1)[0].rstrip(" ,;:")
        return variants,clarification
    except (httpx.HTTPError,KeyError,IndexError,TypeError,ValueError) as exc:
        logger.warning("PCCOE query planning failed; using local query understanding: %s",exc)
        return empty_plan

def _model_query_variants(question):
    return _model_query_plan(question)[0]

def _intent_query(question):
    for pattern,expansion in _INTENT_EXPANSIONS:
        if re.search(pattern,question,re.I): return question+" "+expansion
    return question


class PCCOEKnowledgeAgent:
    """Plan a query, retrieve hybrid evidence, and expand relevant source pages."""

    def retrieve(self, question: str, limit: int = 18):
        # A natural-language query and its normalized concept query give vector
        # search two complementary phrasings. FTS inside search() adds exact
        # lexical retrieval for names, years, amounts and course codes.
        fallback_translation=_fallback_english_query(question) if _looks_multilingual(question) else question
        multilingual=_looks_multilingual(question)
        model_variants=_model_query_variants(question)
        # Search the user's full phrasing and its content words. Do not route
        # the query through a fixed question-type template: exact terms plus
        # vector similarity work for topics the assistant has never seen.
        queries=[question]
        compact=_keyword_query(fallback_translation)
        if compact and compact.casefold() not in {item.casefold() for item in queries}:
            queries.append(compact)
        expanded=_intent_query(fallback_translation)
        if expanded.casefold() not in {item.casefold() for item in queries}:
            queries.append(expanded)
        btech_admission_query=(
            re.search(r"\b(b\.?\s*tech|btech)\b",question,re.I)
            and re.search(r"\b(admission|eligibility|entrance|cut.?off|application|cap)\b",question,re.I)
        )
        if btech_admission_query:
            queries.extend((
                "PCCOE first year B.Tech CAP admission process seat acceptance ERP admission form eligibility",
                "PCCOE B.Tech institute level and against CAP admission process",
            ))
        if not multilingual:
            # Preserve the user's wording alongside expanded and compact forms.
            # For course-level syllabus questions, also search the named course
            # itself; adding generic words like "official" and "curriculum"
            # otherwise crowds out the actual course title in a large corpus.
            if re.search(r"\b(syllabus|curriculum|course structure|course content|unit|subject)\b",question,re.I):
                generic={"syllabus","syllabu","curriculum","course","structure","content","unit","units","subject","subjects","what","which","who","where","when","how","give","tell","show","me","the","a","an","of","for","in","on","is","are","do","does","pccoe","college","engineering","department","program","programme"}
                named_terms=_terms(question)-generic
                if named_terms:
                    queries.append(" ".join(sorted(named_terms)))
        queries.extend(model_variants[:1] if multilingual else model_variants[:2])
        role_query=bool(re.search(r"\bHOD\b|head\s+of\s+department|department\s+head",question,re.I))
        requested_department=resolve_department(question)
        faculty_query=bool(re.search(r"\b(faculty|staff|professors?|teachers?)\b",question,re.I))
        if faculty_query and requested_department:
            # A faculty roster is a page-level lookup: add a compact profile
            # title query so a nearby alumni/FDP/M.Tech guide result cannot
            # crowd the actual department people directory out of top-k.
            department=requested_department[1]
            queries.extend((
                f"{department} Faculty Profiles PCCOE",
                f"{department} faculty profile teachers staff roster names designations email",
            ))
        if role_query and requested_department:
            _,department=requested_department
            short=next((alias for alias,label in sorted(DEPARTMENT_HOSTS.items(),key=lambda item:len(item[0]))
                        if label==department and len(alias)<=5),department)
            # Official PCCOE rosters use variants such as "HOD-Civil", while
            # department sites often put HOD, name, and contact under a HOD
            # Desk or Contact page. Search those document terms explicitly.
            queries.extend((
                f"PCCOE Academic Council HOD-{short} {department} head name current roster",
                f"{department} department HOD Desk faculty designation head of department contact",
            ))
        # A configured model can translate/normalize any language, including
        # Latin-script languages that are not covered by word-list heuristics.
        # The planner is constrained to search rewrites and never supplies facts.
        if multilingual:
            if fallback_translation.casefold() not in {item.casefold() for item in queries}:
                queries.append(fallback_translation)
        by_id={}
        for query in queries:
            for hit in search(query,limit=limit):
                meta=hit["metadata"]
                key=(meta.get("source_id"),meta.get("chunk"),meta.get("url"))
                previous=by_id.get(key)
                if previous is None or hit.get("rank_score",0)>previous.get("rank_score",0):
                    by_id[key]=hit
        hits=sorted(by_id.values(),key=lambda row:row.get("rank_score",row.get("score",0)),reverse=True)
        weak_direct_match=not hits or max((hit.get("line_coverage",0) for hit in hits),default=0)<0.48
        missing_role_row=role_query and not any(
            _has_department_role_line(hit["text"],requested_department[1] if requested_department else None)
            and _authoritative_role_source(hit)
            for hit in hits
        )
        if not multilingual and (weak_direct_match or missing_role_row) and not model_variants:
            # Ask the LLM only when lexical/vector results lack a passage-level
            # match. It proposes search language, never facts; every candidate
            # still has to come from the PCCOE Chroma/FTS index.
            for variant in _model_query_variants(question)[:2]:
                for hit in search(variant,limit=limit):
                    meta=hit["metadata"]
                    key=(meta.get("source_id"),meta.get("chunk"),meta.get("url"))
                    previous=by_id.get(key)
                    if previous is None or hit.get("rank_score",0)>previous.get("rank_score",0):
                        by_id[key]=hit
            hits=sorted(by_id.values(),key=lambda row:row.get("rank_score",row.get("score",0)),reverse=True)
        if re.search(r"\bhod\b|head\s+of\s+department|department\s+head",question,re.I) and not hits:
            normalized=fallback_translation.lower()
            department=next((label for alias,label in sorted(DEPARTMENT_HOSTS.items(),key=lambda item:len(item[0]),reverse=True) if re.search(r"(?<!\w)"+re.escape(alias)+r"(?!\w)",normalized)),"PCCOE")
            hits=search(f"{department} HOD head of department professor faculty profile email phone contact",limit=limit)
        expansion_question=" ".join([question,*model_variants[:1]])
        return self._expand_relevant_sources(expansion_question,hits)

    def _expand_relevant_sources(self, question, hits):
        terms=_terms(question)
        # Enumerative questions need the rest of a matched source (e.g. all
        # faculty profiles), not just the one chunk the vector index returned.
        wants_complete=re.search(r"\b(list|all|every|complete|full|overview|details?|detailed|process|steps?|procedure|how\s+to\s+apply|faculty|staff|names|who\s+are|what\s+are|show\s+me|clubs?|chapters?|societies)\b",question,re.I)
        # For factual questions, retrieve the rest of the best topical page as
        # well. A single matching chunk may contain only the page heading while
        # the answer-bearing table is stored in a later chunk.
        topic_patterns={
            "Placements and careers":r"placement|recruitment|internship|package|salary|career",
            "Tuition and fees":r"fee|fees|tuition|cost|charges",
            "Syllabus and curriculum":r"syllabus|curriculum|course structure|subject|unit",
            "Academic calendars and schedules":r"calendar|timetable|schedule|exam date",
            "Faculty and official contacts":r"faculty|staff|professor|teacher|contact|\bhod\b|head of department|email|phone",
            "Admissions":r"admission|eligibility|cut.?off|entrance",
            "Scholarships and financial aid":r"scholarship|financial aid|tfws|freeship",
            "Student activities and events":r"clubs?|chapters?|societies|student organizations?|events?|teams?",
        }
        topics=[category for category,pattern in topic_patterns.items() if re.search(pattern,question,re.I)]
        if not hits:
            return hits
        role_query=bool(re.search(r"\bhod\b|head\s+of\s+department|department\s+head",question,re.I))
        faculty_query=bool(re.search(r"\b(faculty|staff|professors?|teachers?)\b",question,re.I)) or role_query
        btech_admission_query=bool(
            re.search(r"\b(b\.?\s*tech|btech)\b",question,re.I)
            and re.search(r"\b(admission|eligibility|entrance|cut.?off|application|cap)\b",question,re.I)
        )
        working_professional_query=bool(re.search(r"\bworking\s+professionals?\b|\bb\.?\s*tech\s*\(?w\.?\s*p\.?\)?",question,re.I))
        ranked_sources=[]
        for hit in hits:
            meta=hit["metadata"]
            source_key=meta.get("source_key") or meta.get("url")
            if not source_key:
                continue
            source_title_url=(str(meta.get("title",""))+" "+str(meta.get("url",""))).casefold()
            if btech_admission_query and re.search(r"\bmca\b|mca\.pccoepune",source_title_url):
                continue
            if btech_admission_query and not working_professional_query and re.search(r"working[- _]?professional|btech-wp",source_title_url):
                continue
            normalized_key=re.sub(r"^https?://(?:www\.)?","",source_key,flags=re.I).rstrip("/")
            source_identity=(normalized_key,meta.get("title",""))
            if source_identity in [item[0] for item in ranked_sources]:
                continue
            title_terms=_terms(meta.get("title","")+" "+meta.get("category",""))
            title_text=(meta.get("title","")+" "+meta.get("url","")).lower()
            category=meta.get("category","")
            activity_topic="Student activities and events" in topics
            topic_match=not topics or category in topics or any(term in title_text for term in ("placement","fee","syllabus","calendar","faculty","admission","scholarship")) or (activity_topic and any(term in title_text for term in ("club","chapter","association","student-development","student development","student-activities","extra-curricular","stdev")))
            body_lines=[line for line in hit["text"].splitlines() if line.strip() and not line.lstrip().lower().startswith("link:")]
            content_topic_match=not topics or any(re.search(pattern,line,re.I) for line in body_lines for topic in topics for pattern in [topic_patterns[topic]])
            requested=resolve_department(question)
            department=requested[1] if requested else None
            inferred_department=infer_department(meta.get("url",""),meta.get("title",""))
            source_department=inferred_department if inferred_department!="Institution-wide" else (meta.get("department") or inferred_department)
            if role_query and department and source_department not in {department,"Institution-wide",""}:
                # "Civil" also occurs in Civil (Regional Language) URLs and
                # metadata. Keep those as separate programs for role lookups.
                continue
            role_evidence=_has_department_role_line(hit["text"],department) or (role_query and source_department==department and _has_named_role(hit["text"]))
            role_source=_authoritative_role_source(hit)
            if role_query and (not role_evidence or not role_source):
                continue
            if topics and not content_topic_match:
                continue
            # Taxonomy labels and page titles are only hints. A table row on a
            # generic governance page can directly answer a department query
            # (for example, a current HOD roster), so preserve sources whose
            # retrieved passage itself has meaningful query overlap.
            if not topic_match and hit.get("line_coverage",0)<0.25:
                continue
            if faculty_query and (not role_query and not ({"faculty","staff","professor"}&title_terms) or "development" in title_text or "fdp-" in title_text):
                continue
            faculty_roster_request=bool(faculty_query and not role_query and re.search(r"\b(list|all|every|complete|full|names?|teachers?)\b",question,re.I))
            if faculty_roster_request and not re.search(r"faculty[- _]?profiles?|/faculty(?:/|$)|staff[- _]?profiles?|faculty[- _]?list|staff[- _]?list",title_text):
                continue
            ranked_sources.append((source_identity,hit,source_key,source_department))
            source_limit=3 if wants_complete else (2 if role_query else 1)
            if len(ranked_sources)>=source_limit:
                break
        expanded={}
        eligible_sources={identity for identity,_,_,_ in ranked_sources}
        for (_, _),base_hit,source_key,source_department in ranked_sources:
            meta0=base_hit["metadata"]
            try:
                stored=collection.get(where={"source_key":source_key},include=["documents","metadatas"])
            except Exception:
                continue
            for doc,meta in zip(stored.get("documents",[]) or [],stored.get("metadatas",[]) or []):
                body_terms=_terms(doc)
                coverage=len(terms&body_terms)/max(1,len(terms))
                line_coverage=max((len(terms&_terms(line))/max(1,len(terms)) for line in doc.splitlines() if line.strip() and not line.lstrip().lower().startswith("link:")),default=0)
                direct_role=_has_department_role_line(doc,department)
                if role_query and not direct_role and not (source_department==department and _has_named_role(doc)):
                    continue
                title_coverage=len(terms&_terms(meta.get("title","")))/max(1,len(terms))
                row={"text":doc,"metadata":meta,"score":base_hit.get("score",0),"coverage":coverage,"line_coverage":line_coverage,"direct_role_match":direct_role,
                     "rank_score":0.35*base_hit.get("rank_score",base_hit.get("score",0))+0.35*coverage+0.20*line_coverage+0.10*title_coverage+(0.20 if direct_role else 0),
                     "expanded_from_source":meta0.get("title")}
                key=(meta.get("source_id"),meta.get("chunk"),meta.get("url"))
                old=expanded.get(key)
                if old is None or row["rank_score"]>old["rank_score"]:
                    expanded[key]=row
        if not expanded:
            return hits
        # Preserve a few independently retrieved corroborating chunks too.
        for hit in hits[:6]:
            meta=hit["metadata"]
            source_key=meta.get("source_key") or meta.get("url")
            normalized_key=re.sub(r"^https?://(?:www\.)?","",source_key or "",flags=re.I).rstrip("/")
            if (normalized_key,meta.get("title","")) not in eligible_sources:
                continue
            inferred_department=infer_department(meta.get("url",""),meta.get("title",""))
            corroborating_department=inferred_department if inferred_department!="Institution-wide" else (meta.get("department") or inferred_department)
            if role_query and not (_has_department_role_line(hit["text"],department) or (corroborating_department==department and _has_named_role(hit["text"]))):
                continue
            key=(hit["metadata"].get("source_id"),hit["metadata"].get("chunk"),hit["metadata"].get("url"))
            expanded.setdefault(key,hit)
        return sorted(expanded.values(),key=lambda row:row.get("rank_score",0),reverse=True)[:30]


knowledge_agent=PCCOEKnowledgeAgent()
