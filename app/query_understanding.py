"""Lightweight, explainable intent and entity extraction for PCCOE retrieval."""
import re
from .taxonomy import DEPARTMENT_HOSTS, resolve_department, extract_academic_year

_FEE_SUBTOPICS = (
    ("hostel_and_mess", r"\b(hostel|hostel fee|hostel charges|accommodation fee|residential charges|mess)\b|वसतिगृह|हॉस्टेल|हॉस्टेलचा"),
    ("hostel", r"\b(hostel|accommodation|residential)\b|वसतिगृह|हॉस्टेल|hostel\s+ki\s+fees?"),
    ("mess", r"\bmess\b|mess\s+(?:fee|fees|charge|charges)|खानावळ|मेस शुल्क"),
    ("examination", r"\b(exam|examination)\s+(?:fee|fees|charge|charges)\b|\b(?:fee|fees|charge|charges)\s+(?:for\s+)?(?:exam|examination)\b"),
    ("admission", r"\badmission\s+(?:fee|fees|charge|charges)\b|\b(?:fee|fees|charge|charges)\s+(?:for\s+)?admission\b"),
    ("transport", r"\b(transport|bus)\s+(?:fee|fees|charge|charges)\b|\b(?:fee|fees|charge|charges)\s+(?:for\s+)?(?:transport|bus)\b"),
    ("development", r"\bdevelopment\s+(?:fee|fees|charge|charges)\b"),
    ("deposit", r"\b(refundable\s+)?deposit\b|\bsecurity\s+deposit\b"),
    ("tuition", r"\b(tuition|tuition fee|college tuition|engineering tuition)\b"),
)
_FEE_WORDS = re.compile(r"\b(fee|fees|tuition|charges?|cost|amount|शुल्क|फी)\b|हॉस्टेल|वसतिगृह", re.I)
_SYLLABUS_LEVEL = re.compile(
    r"\b(?:fy|sy|ty|first|second|third|fourth|final|[1-4](?:st|nd|rd|th))\s+year\b|"
    r"\b(?:fy|sy|ty)\b|"
    r"\b(?:semester|sem)\s*(?:[1-8](?:st|nd|rd|th)?|[ivx]+)\b|"
    r"\b[1-8](?:st|nd|rd|th)\s*(?:semester|sem)\b",
    re.I,
)

def _program_label(question: str):
    matches=list(re.finditer(r"\b(b\.?\s*tech|btech|m\.?\s*tech|mtech|mca)\b",question,re.I))
    positive=[]
    for match in matches:
        before=question[max(0,match.start()-40):match.start()]
        if not re.search(r"\bnot\b(?:\s+[\w.&/-]+){0,3}\s*$",before,re.I):
            positive.append(match)
    if positive:
        selected=positive[-1].group(0).casefold()
        if selected=="mca": return "MCA"
        return "M.Tech" if selected.startswith("m") else "B.Tech"
    return "B.Tech" if re.search(r"\bengineering\b",question,re.I) and syllabus_course_terms(question) else None

def detect_fee_subtopic(question: str):
    value=question.lower()
    for name,pattern in _FEE_SUBTOPICS:
        if re.search(pattern,value,re.I):
            if name=="hostel_and_mess":
                # Hostel costs and mess charges are separate rows in the source.
                if re.search(r"\bhostel\b|वसतिगृह|हॉस्टेल",value,re.I) and re.search(r"\bmess\b|मेस",value,re.I):
                    return "hostel_and_mess"
                return "hostel" if re.search(r"\bhostel\b|वसतिगृह|हॉस्टेल",value,re.I) else "mess"
            return name
    if _FEE_WORDS.search(value): return "generic_fee"
    return None

def analyze_query(question: str):
    """Produce an auditable query plan; it never supplies factual answers."""
    department=resolve_department(question)
    subtopic=detect_fee_subtopic(question)
    fee_related=subtopic is not None
    topic="fees" if fee_related else None
    if not topic:
        if re.search(r"\b(admission|eligibility|entrance|cut.?off|application|cap round)\b",question,re.I): topic="admissions"
        elif re.search(r"\b(syllabus|curriculum|course structure|course content)\b",question,re.I): topic="syllabus"
        elif re.search(r"\b(hod|head of department|faculty|staff|professor)\b",question,re.I): topic="faculty"
        elif re.search(r"\b(placement|placements|package|recruitment|internship)\b",question,re.I): topic="placements"
        elif re.search(r"\b(exam|examination|timetable|calendar|schedule)\b",question,re.I): topic="exams"
        elif re.search(r"\b(clubs?|chapters?|societies|student organizations?)\b",question,re.I): topic="activities"
    requested=[]
    if fee_related: requested=["amount","fee_breakdown"]
    elif re.search(r"\b(list|all|every|who are|names)\b",question,re.I): requested=["complete_list"]
    elif topic: requested=["direct_answer"]
    program=_program_label(question)
    entities=["PCCOE"] if re.search(r"pccoe|pccoepune|college",question,re.I) else []
    if department: entities.append(department[1])
    if program: entities.append(program)
    if subtopic and subtopic not in {"generic_fee","tuition"}: entities.append(subtopic)
    return {
        "normalized_question":" ".join(question.split()),
        "topic":topic,
        "subtopic":subtopic,
        "entities":list(dict.fromkeys(entities)),
        "department":department[1] if department else None,
        "program":program,
        "academic_year":extract_academic_year(question),
        "requested_information":requested,
        "required_evidence_type":"official_fee_source" if fee_related else ("official_pccoe_source" if topic else "general_or_clarify"),
    }

def syllabus_course_terms(question: str):
    """Return the named course phrase, excluding degree and department scope."""
    named_course=re.search(
        r"\b(?:syllabus|curriculum|course structure|course content)\s+(?:of|for|about)\s+"
        r"(.+?)(?=\s+(?:from|in|at|during|academic year|semester|department|program|programme|for)\b|[?.,;]|$)",
        question,re.I,
    )
    if named_course:
        phrase=named_course.group(1)
        phrase=re.sub(r"\b(?:b\.?\s*tech|btech|m\.?\s*tech|mtech|mca)\b"," ",phrase,flags=re.I)
        department=resolve_department(question)
        if department:
            for alias,label in DEPARTMENT_HOSTS.items():
                if label==department[1]:
                    phrase=re.sub(r"(?<!\w)"+re.escape(alias)+r"(?!\w)"," ",phrase,flags=re.I)
        terms=set(re.findall(r"[a-z0-9]+",phrase.casefold()))
        terms-={"pccoe","college","engineering","department","program","programme","course","syllabus","curriculum","structure","content","subject","unit","units","year","semester","sem"}
    else:
        stop_words={
            "a","an","the","is","are","of","for","from","in","at","about","during","and","or",
            "what","which","give","tell","show","me","pccoe","college","syllabus","syllabu","curriculum",
            "course","structure","content","units","unit","subject","subjects","program","programme",
            "department","semester","year","academic","b","btech","tech","m","mtech","mca","cse",
            "scope","clarification","not","want","specifically","first","second","third","fourth","final",
            "fy","sy","ty","sem",
        }
        terms=set(re.findall(r"[a-z0-9]+",question.casefold()))-stop_words
        department=resolve_department(question)
        if department:
            department_terms=set(re.findall(r"[a-z0-9]+",department[1].casefold()))
            for alias,label in DEPARTMENT_HOSTS.items():
                if label==department[1]:
                    department_terms.update(re.findall(r"[a-z0-9]+",alias.casefold()))
            terms-=department_terms
            if department[1]=="Artificial Intelligence and Machine Learning":
                terms-={"ai","ml"}
    return terms

def clarify_if_underspecified(question: str, contextual_question: str = ""):
    """Ask for missing scope only when the current turn/context cannot resolve it."""
    value=" ".join((contextual_question or question).lower().split())
    intent=analyze_query(value)
    if (intent.get("topic")=="admissions" and not intent.get("program")
            and not re.search(r"\b(international|working professionals?|direct second year|lateral entry)\b",value,re.I)):
        return "Which PCCOE admission route do you mean—B.Tech first year through CAP, direct second year/against CAP, B.Tech for working professionals, M.Tech, MCA, or international admissions?"
    # A request for a named program's annual fee has enough scope to search the
    # latest official fee schedule. The answerer labels the academic year and
    # approval status, so asking the user to choose a category first is needless.
    annual_program_fee=(intent.get("program") is not None
                        and re.search(r"\b(per\s+year|yearly|annual(?:ly)?|each\s+year)\b",value,re.I))
    if intent["subtopic"]=="generic_fee" and not annual_program_fee:
        return "Which fee do you mean—tuition, hostel, mess, admission, examination, transport, or a refundable deposit? If you mean tuition, tell me the program and academic year too."
    department=intent.get("department")
    if intent.get("topic")=="faculty" and not department and re.search(r"\b(hod|head of department|faculty|staff|professor|teacher)\b",value,re.I):
        return "Which PCCOE department do you mean? For example, Computer, Mechanical, Civil, or IT."
    if intent.get("topic")=="placements" and not department and re.search(r"\b(placement|placements|package|recruitment|internship)\b",value,re.I):
        return "Which department and placement batch/year should I look up?"
    syllabus_level=bool(_SYLLABUS_LEVEL.search(value))
    broad_syllabus=bool(
        intent.get("topic")=="syllabus"
        and not syllabus_course_terms(value)
        and not re.search(r"\b(?:all|every|complete|entire)\s+(?:years?|program(?:me)?|curriculum)\b",value,re.I)
    )
    if broad_syllabus:
        missing=[]
        if not department: missing.append("department")
        if not intent.get("program"): missing.append("program (B.Tech, M.Tech, or MCA)")
        if not syllabus_level: missing.append("year/semester")
        if missing:
            if len(missing)==3:
                return "Which PCCOE department, program (B.Tech, M.Tech, or MCA), and year/semester should I use for the syllabus?"
            if len(missing)==2:
                return f"Which {missing[0]} and {missing[1]} should I use for the syllabus?"
            return f"Which {missing[0]} should I use for the syllabus?"
    if intent.get("topic")=="activities" and not department:
        return "Which department’s student clubs or chapters do you mean?"
    if re.fullmatch(r"(?:pccoe\s+)?(?:syllabus|curriculum|course structure)\??",value):
        return "Which department or program, year, and academic regulation should I use for the syllabus?"
    if re.fullmatch(r"(?:pccoe\s+)?(?:placements?|packages?)\??",value):
        return "Which department and placement batch/year do you mean?"
    if re.fullmatch(r"(?:pccoe\s+)?(?:faculty|staff)\??",value):
        return "Which PCCOE department’s faculty or staff list do you need?"
    if re.fullmatch(r"(?:pccoe\s+)?(?:hod|head of department)\??",value):
        return "Which PCCOE department’s HOD are you asking about?"
    if re.fullmatch(r"(?:pccoe\s+)?(?:exam dates?|exam timetable|when are exams)\??",value):
        return "Which program/year and semester should I check for exam dates?"
    return None

def scope_clarification_followup(question: str, history):
    """Attach a short program/department reply to the question that requested it."""
    if len(question.split()) > 8 or not history:
        return None
    user_turns=[row.get("content","") for row in history if row.get("role")=="user" and row.get("content")]
    assistant_turns=[row.get("content","") for row in history if row.get("role")=="assistant" and row.get("content")]
    if not user_turns or not assistant_turns:
        return None
    adds_scope=bool(
        resolve_department(question)
        or re.search(r"\b(?:b\.?\s*tech|m\.?\s*tech|mca|b\.?\s*voc|first year|fy|direct second year|lateral entry)\b",question,re.I)
        or re.search(r"\b20\d{2}\s*[-–/]\s*(?:\d{2}|20\d{2})\b",question)
        or _SYLLABUS_LEVEL.search(question)
    )
    invited_scope=bool(re.search(
        r"(?:which\s+(?:pccoe\s+)?(?:department|program|programme|year|semester|course|(?:admission\s+)?route)|"
        r"if\s+you(?:\s+are|'re)?\s+interested\s+in\s+(?:a\s+)?(?:program|programme)|"
        r"(?:tell|share|provide|specify|request).{0,80}(?:program|programme|department|year|semester|course))",
        assistant_turns[-1],re.I,
    ))
    rejected_scope=bool(re.search(
        r"\bnot\b.{0,40}\b(?:b\.?\s*tech|m\.?\s*tech|mca)\b",
        question,re.I,
    ))
    corrects_prior_answer=bool(
        rejected_scope
        and analyze_query(user_turns[-1]).get("topic")
        and re.search(r"\b(?:b\.?\s*tech|m\.?\s*tech|mca)\b",assistant_turns[-1],re.I)
    )
    if not adds_scope or not (invited_scope or corrects_prior_answer):
        return None
    if invited_scope:
        prompt_pattern=re.compile(
            r"(?:which|what)\b.{0,140}\b(?:department|program|programme|year|semester|batch|course|fee)",
            re.I,
        )
        first_prompt=None
        for index in range(len(history)-1,-1,-1):
            row=history[index]
            if row.get("role")!="assistant":
                continue
            if prompt_pattern.search(row.get("content","")):
                first_prompt=index
                continue
            break
        if first_prompt is not None:
            root_index=next(
                (index for index in range(first_prompt-1,-1,-1)
                 if history[index].get("role")=="user" and history[index].get("content")),
                None,
            )
            if root_index is not None:
                contextual=history[root_index]["content"]
                for row in history[root_index+1:]:
                    if row.get("role")=="user" and row.get("content"):
                        contextual+="\nScope clarification: "+row["content"]
                return contextual+"\nScope clarification: "+question
    previous_user=next(
        (row.get("content","") for row in reversed(history)
         if row.get("role")=="user" and row.get("content")),
        "",
    )
    if previous_user:
        return previous_user+"\nScope clarification: "+question
    return None
