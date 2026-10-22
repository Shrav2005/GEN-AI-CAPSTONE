"""Lightweight, explainable intent and entity extraction for PCCOE retrieval."""
import re
from .taxonomy import resolve_department, extract_academic_year

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
        if re.search(r"\b(syllabus|curriculum|course structure|course content)\b",question,re.I): topic="syllabus"
        elif re.search(r"\b(hod|head of department|faculty|staff|professor)\b",question,re.I): topic="faculty"
        elif re.search(r"\b(placement|placements|package|recruitment|internship)\b",question,re.I): topic="placements"
        elif re.search(r"\b(exam|examination|timetable|calendar|schedule)\b",question,re.I): topic="exams"
        elif re.search(r"\b(clubs?|chapters?|societies|student organizations?)\b",question,re.I): topic="activities"
    requested=[]
    if fee_related: requested=["amount","fee_breakdown"]
    elif re.search(r"\b(list|all|every|who are|names)\b",question,re.I): requested=["complete_list"]
    elif topic: requested=["direct_answer"]
    program=next((label for pattern,label in ((r"\b(b\.?tech|btech|engineering)\b","B.Tech"),(r"\b(m\.?tech|mtech)\b","M.Tech"),(r"\bmca\b","MCA")) if re.search(pattern,question,re.I)),None)
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

def clarify_if_underspecified(question: str, contextual_question: str = ""):
    """Ask for missing scope only when the current turn/context cannot resolve it."""
    value=" ".join((contextual_question or question).lower().split())
    intent=analyze_query(value)
    if intent["subtopic"]=="generic_fee":
        return "Which fee do you mean—tuition, hostel, mess, admission, examination, transport, or a refundable deposit? If you mean tuition, tell me the program and academic year too."
    department=intent.get("department")
    if intent.get("topic")=="faculty" and not department and re.search(r"\b(hod|head of department|faculty|staff|professor|teacher)\b",value,re.I):
        return "Which PCCOE department do you mean? For example, Computer, Mechanical, Civil, or IT."
    if intent.get("topic")=="placements" and not department and re.search(r"\b(placement|placements|package|recruitment|internship)\b",value,re.I):
        return "Which department and placement batch/year should I look up?"
    if intent.get("topic")=="syllabus" and not department and not intent.get("program") and re.search(r"\b(syllabus|curriculum|course structure)\b",value,re.I):
        return "Which department or program, year, and academic regulation should I use for the syllabus?"
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
