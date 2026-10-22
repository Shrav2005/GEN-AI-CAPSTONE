from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    # Automatically use a configured provider; retain the local extractive
    # answerer when the user has not added an API key.
    llm_provider: str = "auto"
    gemini_api_key: str = ""
    groq_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    groq_model: str = "llama-3.3-70b-versatile"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    chroma_path: str = "./data/chroma"
    data_path: str = "./data"
    college_url: str = "https://www.pccoepune.com/"
    allowed_domains: str = "pccoepune.com"
    seed_urls: str = ",".join((
        "https://www.pccoepune.com/",
        "https://www.pccoepune.com/heads-and-deans.php",
        "https://www.pccoepune.com/fee-structure.php",
        "https://www.pccoepune.com/pdf/fee-structure-2026-27/First-Year-ENGG-CAP-AY-2026-27.pdf",
        "https://www.pccoepune.com/pdf/fee-structure-2026-27/FY-Engineering-Fees-Structure-(A-CAP)-AY-2026-27.pdf",
        "https://www.pccoepune.com/pdf/fee-structure-2026-27/First-Year-IL-AY-2026-27.pdf",
        "https://www.pccoepune.com/pdf/fee-structure-2026-27/First-Year-NRI-AY-2026-27.pdf",
        "https://www.pccoepune.com/pdf/fee-structure-2026-27/First-Year-CIWGC-OCI-PIO-FNS-AY-2026-27.pdf",
        "https://www.pccoepune.com/pdf/fee-structure-2026-27/First-Year-SAARC-AY-2026-27.pdf",
        "https://www.pccoepune.com/pdf/fee-structure-2026-27/Direct-Second-Year-ENGG-CAP-AY-2026-27.pdf",
        "https://www.pccoepune.com/pdf/fee-structure-2026-27/First-Year-MTech-CAP-AY-2026-27.PDF",
        "https://www.pccoepune.com/pdf/fee-structure-2026-27/First-Year-MCA-CAP-AY-2026-27.pdf",
        "https://www.pccoepune.com/pdf/fee-structure-2026-27/Mtech-ACAP-Fees-Structure-For-AY-2026-2027.pdf",
        "https://www.pccoepune.com/pdf/fee-structure-2026-27/MCA-ACAP-Fees-Structure-For-AY-2026-2027.pdf",
        "https://www.pccoepune.com/academic-calendar.php",
        "https://www.pccoepune.com/timetable-examinations.php",
        "https://www.pccoepune.com/notice-circular.php",
        "https://www.pccoepune.com/nba-records.php",
        "https://www.pccoepune.com/pdf/nba-records/Faculty-List-2022-to-2026-Comp.pdf",
        "https://www.computer.pccoepune.com/", "https://computer.pccoepune.com/",
        "https://it.pccoepune.com/", "https://entc.pccoepune.com/",
        "https://compregional.pccoepune.com/", "https://civil.pccoepune.com/",
        "https://mechanical.pccoepune.com/", "https://aiml.pccoepune.com/",
        "https://aiml.pccoepune.com/faculty-profile",
        "https://fe.pccoepune.com/", "https://mca.pccoepune.com/",
        "https://www.pccoepune.com/bvoc/",
        "https://computer.pccoepune.com/faculty-profile",
        "https://computer.pccoepune.com/computer-hod",
        "https://computer.pccoepune.com/m-tech-faculty",
        "https://computer.pccoepune.com/professor-of-practice",
        "https://mechanical.pccoepune.com/pdf/ATAL-FDP-BROCHURE-AND-SCHEDULE.pdf",
        "https://computer.pccoepune.com/assets/pdf/TY%20Syllabus%202025-26%20%28Regulations%202023%29.pdf",
        "https://it.pccoepune.com/assets/pdf/BE_IT_2019_Course_Syllabus.pdf",
        "https://entc.pccoepune.com/syllabi.php",
        "https://compregional.pccoepune.com/syllabus.php",
        "https://fe.pccoepune.com/Academics/course.php",
        "https://mca.pccoepune.com/syllabus.php",
        "https://mca.pccoepune.com/fymca-calendar.php",
        "https://mca.pccoepune.com/symca-calendar.php",
        "https://mca.pccoepune.com/assets/pdf/syllabus/FYMCA%20Syllabus%20%28V1.0%29%20for%202024-2025%20%28Cycle-3%202024-2025%20Pattern%29.pdf",
        "https://mca.pccoepune.com/assets/pdf/syllabus/SYMCA%20Syllabus%20for%202025-2026%20%28Cycle-3%20Regulation%202024%20Batch24-26%29.pdf",
    ))
    # Keep update/admin routes disabled until the operator sets a private token.
    admin_token: str = ""
    confidence_threshold: float = 0.50
    max_crawl_pages: int = 5000
    crawl_interval_hours: int = 24
    frontend_origin: str = "http://localhost:5173"

    def active_llm_provider(self):
        configured=self.llm_provider.lower().strip()
        if configured not in {"", "auto"}:
            return configured
        if self.gemini_api_key.strip(): return "gemini"
        if self.groq_api_key.strip(): return "groq"
        return "mock"

settings = Settings()
