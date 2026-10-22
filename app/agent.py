"""One retrieval-grounded PCCOE knowledge agent (no model training required)."""
import json
import re
from functools import lru_cache
import httpx
from .config import settings
from .store import collection, search, _terms
from .taxonomy import DEPARTMENT_HOSTS

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
def _model_query_variants(question):
    """Use the configured LLM only as a language/query planner, never as evidence."""
    provider=settings.active_llm_provider()
    if provider not in {"gemini","groq","ollama"}: return ()
    prompt=("Translate and rewrite this student question into up to three short English search queries for finding evidence on PCCOE's official websites. "
            "Preserve every named department, person, program, year and number exactly. Do not answer the question or add facts. "
            'Return JSON only in this form: {"queries":["..."]}. Question: '+question)
    try:
        if provider=="gemini" and settings.gemini_api_key:
            response=httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{settings.gemini_model}:generateContent",headers={"x-goog-api-key":settings.gemini_api_key},json={"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"temperature":0}},timeout=14)
            response.raise_for_status()
            raw=response.json()["candidates"][0]["content"]["parts"][0]["text"]
        elif provider=="groq" and settings.groq_api_key:
            response=httpx.post("https://api.groq.com/openai/v1/chat/completions",headers={"Authorization":"Bearer "+settings.groq_api_key},json={"model":settings.groq_model,"messages":[{"role":"user","content":prompt}],"temperature":0,"response_format":{"type":"json_object"}},timeout=14)
            response.raise_for_status()
            raw=response.json()["choices"][0]["message"]["content"]
        elif provider=="ollama":
            response=httpx.post(settings.ollama_base_url+"/api/generate",json={"model":settings.ollama_model,"prompt":prompt,"stream":False,"format":"json","options":{"temperature":0}},timeout=14)
            response.raise_for_status()
            raw=response.json().get("response","")
        else:
            return ()
        parsed=json.loads(raw)
        queries=parsed.get("queries",[]) if isinstance(parsed,dict) else []
        return tuple(" ".join(str(query).split())[:240] for query in queries[:3] if isinstance(query,str) and query.strip())
    except Exception:
        return ()

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
        concepts=" ".join(sorted(_terms(fallback_translation)))
        queries=[question,_intent_query(fallback_translation)]
        # A configured model can translate/normalize any language, including
        # Latin-script languages that are not covered by word-list heuristics.
        # The planner is constrained to search rewrites and never supplies facts.
        queries.extend(_model_query_variants(question))
        if _looks_multilingual(question):
            queries.append(fallback_translation)
        if concepts and concepts.casefold()!=question.casefold():
            queries.append(concepts)
        if re.search(r"\bhod\b|head\s+of\s+department|department\s+head",question,re.I):
            normalized=fallback_translation.lower()
            department=next((label for alias,label in sorted(DEPARTMENT_HOSTS.items(),key=lambda item:len(item[0]),reverse=True) if re.search(r"(?<!\w)"+re.escape(alias)+r"(?!\w)",normalized)),"PCCOE")
            queries.append(f"{department} HOD head of department professor faculty profile email contact")
        by_id={}
        for query in queries:
            for hit in search(query,limit=limit):
                meta=hit["metadata"]
                key=(meta.get("source_id"),meta.get("chunk"),meta.get("url"))
                previous=by_id.get(key)
                if previous is None or hit.get("rank_score",0)>previous.get("rank_score",0):
                    by_id[key]=hit
        hits=sorted(by_id.values(),key=lambda row:row.get("rank_score",row.get("score",0)),reverse=True)
        expansion_question=" ".join([question,*_model_query_variants(question)])
        return self._expand_relevant_sources(expansion_question,hits)

    def _expand_relevant_sources(self, question, hits):
        terms=_terms(question)
        # Enumerative questions need the rest of a matched source (e.g. all
        # faculty profiles), not just the one chunk the vector index returned.
        wants_complete=re.search(r"\b(list|all|every|faculty|staff|names|who\s+are|clubs?|chapters?|societies)\b",question,re.I)
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
            "Student activities and events":r"clubs?|chapters?|societies|student organizations?|events?",
        }
        topics=[category for category,pattern in topic_patterns.items() if re.search(pattern,question,re.I)]
        if not hits:
            return hits
        role_query=bool(re.search(r"\bhod\b|head\s+of\s+department|department\s+head",question,re.I))
        faculty_query=bool(re.search(r"\bfaculty\b|\bstaff\b|\bprofessors?\b",question,re.I)) or role_query
        ranked_sources=[]
        for hit in hits:
            meta=hit["metadata"]
            source_key=meta.get("source_key") or meta.get("url")
            if not source_key:
                continue
            source_identity=(source_key,meta.get("title",""))
            if source_identity in [item[0] for item in ranked_sources]:
                continue
            title_terms=_terms(meta.get("title","")+" "+meta.get("category",""))
            title_text=(meta.get("title","")+" "+meta.get("url","")).lower()
            category=meta.get("category","")
            activity_topic="Student activities and events" in topics
            topic_match=not topics or category in topics or any(term in title_text for term in ("placement","fee","syllabus","calendar","faculty","admission","scholarship")) or (activity_topic and any(term in title_text for term in ("club","chapter","association","student-development","student development","student-activities","extra-curricular","stdev")))
            if not topic_match:
                continue
            if faculty_query and (not role_query and not ({"faculty","staff","professor"}&title_terms) or "development" in title_text or "fdp-" in title_text):
                continue
            ranked_sources.append((source_identity,hit))
            source_limit=3 if wants_complete else (2 if role_query else 1)
            if len(ranked_sources)>=source_limit:
                break
        expanded={}
        for (source_key,_),base_hit in ranked_sources:
            meta0=base_hit["metadata"]
            try:
                stored=collection.get(where={"source_key":source_key},include=["documents","metadatas"])
            except Exception:
                continue
            for doc,meta in zip(stored.get("documents",[]) or [],stored.get("metadatas",[]) or []):
                row_terms=_terms(doc+" "+meta.get("title","")+" "+meta.get("category","")+" "+meta.get("department",""))
                coverage=len(terms&row_terms)/max(1,len(terms))
                row={"text":doc,"metadata":meta,"score":base_hit.get("score",0),"coverage":coverage,
                     "rank_score":0.55*base_hit.get("rank_score",base_hit.get("score",0))+0.45*coverage,
                     "expanded_from_source":meta0.get("title")}
                row["coverage"]=max(row["coverage"],base_hit.get("coverage",0))
                key=(meta.get("source_id"),meta.get("chunk"),meta.get("url"))
                old=expanded.get(key)
                if old is None or row["rank_score"]>old["rank_score"]:
                    expanded[key]=row
        if not expanded:
            return hits
        # Preserve a few independently retrieved corroborating chunks too.
        for hit in hits[:6]:
            key=(hit["metadata"].get("source_id"),hit["metadata"].get("chunk"),hit["metadata"].get("url"))
            expanded.setdefault(key,hit)
        return sorted(expanded.values(),key=lambda row:row.get("rank_score",0),reverse=True)[:30]


knowledge_agent=PCCOEKnowledgeAgent()
