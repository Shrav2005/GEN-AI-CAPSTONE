import re
from difflib import SequenceMatcher
from urllib.parse import urlsplit

CATEGORY_RULES = [
    ("Admissions", ("admission", "cap-round", "eligibility", "mht-cet", "entrance exam", "erp registration", "branch change")),
    ("Tuition and fees", ("fee", "fees", "refund", "payment", "receipt", "tuition")),
    ("Scholarships and financial aid", ("scholarship", "tfws", "ebc", "freeship", "financial aid", "reserved category")),
    ("Syllabus and curriculum", ("syllabus", "curriculum", "course outcome", "course-structure", "practical manual")),
    ("Academic calendars and schedules", ("calendar", "timetable", "schedule", "holiday", "orientation")),
    ("Examinations and assessment", ("exam", "examination", "revaluation", "assessment", "result", "backlog")),
    ("Academic regulations and attendance", ("attendance", "regulation", "grading", "semester progression", "academic discipline")),
    ("Hostel and accommodation", ("hostel", "mess", "accommodation", "warden")),
    ("Placements and careers", ("placement", "recruitment", "internship", "training", "career", "employer")),
    ("Higher studies and research guidance", ("gate", "gre", "cat", "gmat", "ielts", "toefl", "higher studies")),
    ("Faculty and official contacts", ("faculty", "hod", "head of department", "office", "contact", "dean")),
    ("Student welfare and safety", ("grievance", "anti-ragging", "safety", "welfare", "internal complaint", "counselling")),
    ("Certificates and administration", ("certificate", "transcript", "id card", "document correction", "application")),
    ("ERP and student services", ("erp", "student portal", "login", "online portal", "student service")),
    ("Research, projects and innovation", ("research", "project", "incubation", "entrepreneurship", "patent", "intellectual property")),
    ("International relations", ("international", "exchange program", "overseas", "global collaboration")),
    ("Alumni", ("alumni", "alumnus", "mentoring", "homecoming")),
    ("Transportation and campus directions", ("transport", "bus route", "directions", "visitor", "navigation", "address")),
    ("Notices and circulars", ("notice", "circular", "announcement", "deadline", "registration")),
    ("Student activities and events", ("club", "chapter", "event", "workshop", "hackathon", "competition", "sports", "nss")),
    ("Campus facilities", ("facility", "facilities", "library", "laboratory", "wifi", "cafeteria", "medical", "infrastructure")),
    ("Programs and departments", ("department", "program", "course", "intake", "specialization", "b.voc", "mca", "ph.d", "m.tech")),
    ("College profile and campus", ("about", "accreditation", "approval", "leadership", "profile", "campus", "organization chart")),
    ("Parent and guardian information", ("parent", "guardian", "parent meeting")),
    ("Graduation and completion procedures", ("graduation", "convocation", "completion", "degree certificate", "final year")),
    ("Unanswered questions and escalation", ("unanswered", "human support", "escalation", "help desk")),
]

DEPARTMENT_HOSTS = {
    "computer": "Computer Engineering",
    "compregional": "Computer Engineering (Regional Language)",
    "computer regional": "Computer Engineering (Regional Language)",
    "computer (regional language)": "Computer Engineering (Regional Language)",
    "computer engineering regional language": "Computer Engineering (Regional Language)",
    "it": "Information Technology",
    "information technology": "Information Technology",
    "entc": "Electronics and Telecommunication",
    "e&tc": "Electronics and Telecommunication",
    "electronics and telecommunication": "Electronics and Telecommunication",
    "civil": "Civil Engineering",
    "civil engineering": "Civil Engineering",
    "civil regional": "Civil Engineering (Regional Language)",
    "civil (regional)": "Civil Engineering (Regional Language)",
    "civil (regional language)": "Civil Engineering (Regional Language)",
    "civil engineering regional": "Civil Engineering (Regional Language)",
    "civil engineering regional language": "Civil Engineering (Regional Language)",
    "civil engineering (regional language)": "Civil Engineering (Regional Language)",
    "mech": "Mechanical Engineering",
    "mechanical": "Mechanical Engineering",
    "mechanical engineering": "Mechanical Engineering",
    "ai": "Artificial Intelligence",
    "aiml": "Artificial Intelligence and Machine Learning",
    "ai ml": "Artificial Intelligence and Machine Learning",
    "ai/ml": "Artificial Intelligence and Machine Learning",
    "ai-ml": "Artificial Intelligence and Machine Learning",
    # The college's official full name contains the generic word "Computer".
    # Keep the complete AIML department aliases ahead of the short "computer"
    # alias so they are never misclassified as Computer Engineering.
    "computer science & engineering (aiml)": "Artificial Intelligence and Machine Learning",
    "computer science and engineering (aiml)": "Artificial Intelligence and Machine Learning",
    "computer science & engineering aiml": "Artificial Intelligence and Machine Learning",
    "computer science and engineering aiml": "Artificial Intelligence and Machine Learning",
    "cse (aiml)": "Artificial Intelligence and Machine Learning",
    "cse aiml": "Artificial Intelligence and Machine Learning",
    "artificial intelligence and machine learning": "Artificial Intelligence and Machine Learning",
    "vlsi": "Electronics Engineering (VLSI Design and Technology)",
    "fe": "First Year Engineering",
    "mca": "MCA",
    "b.voc": "B. Voc.",
    "applied science and humanities": "Applied Science and Humanities",
    "multidisciplinary studies": "Multidisciplinary Studies",
}

def resolve_department(value: str):
    """Resolve exact department aliases first, then tolerate a clear typo."""
    aliases=sorted(DEPARTMENT_HOSTS.items(),key=lambda item:len(item[0]),reverse=True)
    for alias,label in aliases:
        if re.search(r"(?<!\w)"+re.escape(alias)+r"(?!\w)",value,re.I):
            return alias,label
    words=re.findall(r"[a-zA-Z&]+",value.lower())
    candidates=[]
    for word in words:
        if len(word)<5: continue
        for alias,label in aliases:
            if len(alias)<5 or " " in alias or "/" in alias: continue
            ratio=SequenceMatcher(None,word,alias).ratio()
            if ratio>=0.84:
                candidates.append((ratio,alias,label))
    if not candidates: return None
    candidates.sort(reverse=True)
    best=candidates[0]
    # Ambiguous near-ties should not select an arbitrary department.
    if len(candidates)>1 and candidates[1][0]>=best[0]-0.025 and candidates[1][2]!=best[2]:
        return None
    return best[1],best[2]

def classify_category(url: str, title: str, content: str = "") -> str:
    # Prefer the page identity over incidental words in the global navigation.
    page_identity = f"{url} {title}".lower()
    for category, terms in CATEGORY_RULES:
        if any(term in page_identity for term in terms):
            return category
    value = content[:5000].lower()
    for category, terms in CATEGORY_RULES:
        if any(term in value for term in terms):
            return category
    return "College profile and campus"

def infer_department(url: str, title: str = "") -> str:
    host = (urlsplit(url).hostname or "").lower()
    path = urlsplit(url).path.lower()
    parts=host.split(".")
    first=parts[1] if len(parts)>1 and parts[0]=="www" else parts[0]
    if first in DEPARTMENT_HOSTS:
        return DEPARTMENT_HOSTS[first]
    # Some PCCOE sub-sites are path-mounted on the main domain instead of
    # using their own hostname. Resolve these before title keywords so the
    # regional-language program cannot be mislabeled as the main department.
    if re.search(r"(?:^|/)civilregional(?:/|$)",path):
        return "Civil Engineering (Regional Language)"
    if re.search(r"(?:^|/)(?:computerregional|compregional)(?:/|$)",path):
        return "Computer Engineering (Regional Language)"
    value = (path+" "+title).lower()
    for key, label in (("computer", "Computer Engineering"), ("information technology", "Information Technology"), ("e&tc", "Electronics and Telecommunication"), ("civil", "Civil Engineering"), ("mechanical", "Mechanical Engineering"), ("mech", "Mechanical Engineering"), ("artificial intelligence", "Artificial Intelligence")):
        if re.search(r"(?<![a-z])"+re.escape(key)+r"(?![a-z])",value):
            return label
    return "Institution-wide"

def extract_academic_year(value: str):
    matches=re.findall(r"\b(20\d{2})\s*[-–/]\s*(\d{2}|20\d{2})\b",value or "")
    if not matches: return None
    normalized=[]
    for start,end in matches:
        end=start[:2]+end if len(end)==2 else end
        normalized.append((int(start),int(end),f"{start}-{end}"))
    # Pages often mention an older policy first and put the current academic
    # year near a later notice/table. Keep the newest explicit range.
    return max(normalized,key=lambda item:(item[0],item[1]))[2]
