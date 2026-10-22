import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from app.store import _chunks, _terms
from app.main import _PCCOE_SCOPE, _compact_evidence, _format_clubs_answer, _format_faculty_answer, _format_fee_answer, _format_hod_contact_answer, _format_placement_answer, _syllabus_course_rows, _translate_grounded_answer, source_references
from app.agent import _looks_multilingual
from app.config import settings
from app.taxonomy import classify_category, infer_department, extract_academic_year, resolve_department
from app.query_understanding import clarify_if_underspecified, detect_fee_subtopic
from app.conversations import ConversationStore


class KnowledgePipelineTests(unittest.TestCase):
    def test_multilingual_detector_handles_multiple_writing_systems(self):
        self.assertTrue(_looks_multilingual("कंप्यूटर विभाग के क्लब कौन से हैं"))
        self.assertTrue(_looks_multilingual("ما هي أندية قسم الحاسوب؟"))
        self.assertTrue(_looks_multilingual("¿Cuáles son los clubes de informática?"))

    def test_configured_llm_localizes_without_becoming_an_evidence_source(self):
        class FakeResponse:
            def raise_for_status(self): pass
            def json(self): return {"choices":[{"message":{"content":"إجابة مترجمة [1]"}}]}
        with patch.object(settings,"llm_provider","groq"), patch.object(settings,"groq_api_key","unit-test-key"), patch("httpx.post",return_value=FakeResponse()) as request:
            answer=_translate_grounded_answer("ما هي أندية قسم الحاسوب؟","Computer clubs are ACM and OWASP [1].")
        self.assertEqual(answer,"إجابة مترجمة [1]")
        prompt=request.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("Do not add, remove, correct, or infer facts",prompt)

    def test_club_answer_uses_department_chapter_and_activity_pages(self):
        pages={
            "https://computer.pccoepune.com/sdw-chapters":("Professional Student Chapters | PCCOE Computer Engineering","Joining professional student chapters such as ACM, ACM-W, OWASP, GDGC, and LFDT provides our students with professional networks.\nLink: Visit Website | https://iiris.vercel.app/"),
            "https://computer.pccoepune.com/sdw-coordinators-desk":("Student Development & Welfare Coordinator Desk | PCCOE Computer Engineering","Developer Clubs Coordinator (Coding Club, GDSC, Open-Source Club, Software Development Club)\nSecurity & Trust Club Coordinator (OWASP, LFDT, Trust Lab IIT Bombay)\nIoT, Intelligence & Robotics Innovation Society coordinator\nStudent Association (CESA), Chapters, Clubs and coordination of all co-curricular activities"),
            "https://computer.pccoepune.com/sdw-student-activities":("Student Activities | PCCOE Computer Engineering","Table: S.N. | Organization / Chapter | Event Name\n1 | Codechef Student Chapter | Hackathon\n2 | PCCoE IEEE Student Branch | Event"),
        }
        def fake_get(where=None,include=None,**kwargs):
            url=(where or {}).get("url")
            if url not in pages: return {"ids":[],"documents":[],"metadatas":[]}
            title,text=pages[url]
            return {"ids":["fixture"],"documents":[text],"metadatas":[{"title":title,"url":url,"type":"HTML","category":"Student activities and events","department":"Computer Engineering"}]}
        with patch("app.main.collection.get",side_effect=fake_get):
            result=_format_clubs_answer("List clubs in Computer department",[])
        self.assertIsNotNone(result)
        self.assertIn("ACM-W",result[0])
        self.assertIn("Coding Club",result[0])
        self.assertIn("Security & Trust Club",result[0])
        self.assertIn("Codechef Student Chapter",result[0])
        self.assertNotIn("and LFDT",result[0])
        self.assertEqual(len(source_references(result[1],max_sources=3)),3)
    def test_fee_queries_keep_fee_categories_distinct_and_clarify_generic(self):
        self.assertEqual(detect_fee_subtopic("hostel fees"),"hostel")
        self.assertEqual(detect_fee_subtopic("mess charges"),"mess")
        self.assertEqual(detect_fee_subtopic("tuition fee for engineering"),"tuition")
        self.assertEqual(detect_fee_subtopic("fees"),"generic_fee")
        self.assertIn("Which fee",clarify_if_underspecified("fees"))

    def test_generic_department_query_asks_for_scope(self):
        self.assertIn("Which PCCOE department",clarify_if_underspecified("Who is the HOD?"))
        self.assertIn("Which department",clarify_if_underspecified("show placements"))

    def test_conversations_survive_store_recreation(self):
        data_dir=Path(__file__).resolve().parents[1]/"data"
        data_dir.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=data_dir) as directory:
            store=ConversationStore(directory)
            session=store.create()
            store.append_turn(session,"Who is the Mechanical HOD?",{"answer":"Dr. Example","sources":[{"title":"Roster","url":"https://example.org"}],"fallback":False})
            reopened=ConversationStore(directory)
            self.assertEqual(reopened.history(session),[{"role":"user","content":"Who is the Mechanical HOD?"},{"role":"assistant","content":"Dr. Example"}])
            self.assertEqual(reopened.messages(session)[-1]["sources"][0]["title"],"Roster")
    def test_chunks_preserve_table_rows_and_stay_bounded(self):
        text="Detailed Syllabus\nUnit I | Distributed Systems | 07 hours\n" + ("A course outcome about distributed systems. " * 80)
        chunks=_chunks(text,size=320,overlap=40)
        self.assertGreater(len(chunks),1)
        self.assertTrue(any("Unit I | Distributed Systems | 07 hours" in chunk for chunk in chunks))
        self.assertTrue(all(len(chunk)<=320 for chunk in chunks))

    def test_query_terms_keep_abbreviations_and_normalize_variants(self):
        self.assertIn("ai",_terms("AI syllabus"))
        self.assertIn("it",_terms("IT syllabus"))
        self.assertIn("final",_terms("fourth year"))
        self.assertIn("system",_terms("Distributed Systems"))

    def test_syllabus_pdf_table_rows_join_wrapped_course_names(self):
        text="""Semester V
BCE25PC01 Theory of 2 - 1 3 2 - 1 1 20 20 60 - - - 100
Computation
BCE25PE01 Program Elective 4 - 4 2 4 - 1 10 10 30 50 - 50 150
-03 Course I"""
        rows=_syllabus_course_rows(text)
        self.assertIn(("BCE25PC01","Theory of Computation"),rows)
        self.assertIn(("BCE25PE01-03","Program Elective Course I"),rows)

    def test_taxonomy_and_department_metadata(self):
        self.assertEqual(classify_category("https://computer.pccoepune.com/syllabus","Computer Syllabus"),"Syllabus and curriculum")
        self.assertEqual(classify_category("https://www.computer.pccoepune.com/syllabus","Computer Engineering Syllabus","Admissions portal navigation"),"Syllabus and curriculum")
        self.assertEqual(infer_department("https://www.computer.pccoepune.com/syllabus"),"Computer Engineering")
        self.assertEqual(extract_academic_year("Syllabus 2025-26"),"2025-2026")

    def test_question_bank_has_26_categories_and_language_variants(self):
        path=Path(__file__).resolve().parents[1]/"data"/"question_bank.json"
        questions=json.loads(path.read_text(encoding="utf-8"))["questions"]
        self.assertEqual(len(questions),26)
        self.assertTrue(all({"hi","mr","hinglish"}.issubset(question["variants"]) for question in questions))
        self.assertTrue(all(question["test_status"]=="not_run" for question in questions))

    def test_hod_does_not_join_general_college_contacts_to_a_person(self):
        url="https://www.pccoepune.com/heads-and-deans.php"
        roster={"text":"Head of the Department / Program Heads\nMechanical Engineering | Dr. P.R. Kale | Professor Mechanical\nEmail: pccoeadmin@gmail.com\nPhone: 8087174347",
                "metadata":{"title":"PCCOE Heads and Deans","url":url,"source_key":url,"department":"Institution-wide","type":"HTML"},
                "score":0.9,"rank_score":0.9}
        with patch("app.main.search",return_value=[]):
            result=_format_hod_contact_answer("Who is HOD of Mechanical Engineering? Give email and phone",[roster])
        self.assertIsNotNone(result)
        answer,used=result
        self.assertIn("Dr. P.R. Kale",answer)
        self.assertNotIn("pccoeadmin@gmail.com",answer)
        self.assertNotIn("8087174347",answer)
        self.assertEqual(len(source_references(used,max_sources=2)),1)

    def test_hod_does_not_use_another_departments_roster_entry(self):
        url="https://www.pccoepune.com/heads-and-deans.php"
        computer={"text":"Head of the Department / Program Heads\nComputer Engineering | Dr. Computer Person | Professor Computer","metadata":{"title":"PCCOE Heads and Deans","url":url,"source_key":url,"department":"Institution-wide","type":"HTML"},"score":0.9,"rank_score":0.9}
        with patch("app.main.search",return_value=[]):
            result=_format_hod_contact_answer("Who is the AI ML department HOD?",[computer])
        self.assertIsNone(result)

    def test_hod_query_tolerates_mechanical_department_typo(self):
        url="https://www.pccoepune.com/heads-and-deans.php"
        roster={"text":"Head of the Department / Program Heads\nMechanical Engineering | Dr. P.R. Kale | Professor Mechanical","metadata":{"title":"PCCOE Heads and Deans","url":url,"source_key":url,"department":"Institution-wide","type":"HTML"},"score":0.9,"rank_score":0.9}
        self.assertEqual(resolve_department("Who is HOD of mechnical department?"),("mechanical","Mechanical Engineering"))
        with patch("app.main.search",return_value=[]):
            result=_format_hod_contact_answer("Who is HOD of mechnical department?",[roster])
        self.assertIsNotNone(result)
        self.assertIn("Mechanical Engineering",result[0])
        self.assertIn("Dr. P.R. Kale",result[0])

    def test_hod_parser_ignores_mechanical_dean_rows_before_hod_table(self):
        url="https://www.pccoepune.com/heads-and-deans.php"
        roster={"text":"Associate Dean Administration - I | Dr. N. Vivekanandan | Mechanical\nHead of the Department / Program Heads\nMechanical Engineering | Dr. P.R. Kale | Professor Mechanical","metadata":{"title":"PCCOE Heads and Deans","url":url,"source_key":url,"department":"Institution-wide","type":"HTML"},"score":0.9,"rank_score":0.9}
        with patch("app.main.search",return_value=[]):
            result=_format_hod_contact_answer("Who is HOD of Mechanical department?",[roster])
        self.assertIsNotNone(result)
        self.assertIn("Dr. P.R. Kale",result[0])
        self.assertNotIn("Dr. N. Vivekanandan",result[0])

    def test_regular_answer_evidence_is_limited_to_top_distinct_pages(self):
        hits=[]
        for index in range(7):
            hits.append({"text":f"Evidence {index}","metadata":{"url":f"https://www.pccoepune.com/page-{index}.php","page":None,"type":"HTML"},"rank_score":1-index*0.1})
        hits.append({"text":"duplicate page lower ranked","metadata":{"url":"https://www.pccoepune.com/page-0.php","page":None,"type":"HTML"},"rank_score":0.1})
        compact=_compact_evidence(hits,max_sources=3)
        self.assertEqual(len(compact),3)
        self.assertEqual(len(source_references(compact)),3)
        self.assertEqual(compact[0]["text"],"Evidence 0")

    def test_department_placement_question_enters_pccoe_retrieval(self):
        self.assertIsNotNone(_PCCOE_SCOPE.search("What were the previous year Computer Engineering placements?"))

    def test_table_first_placement_page_returns_previous_batch_summary(self):
        hit={"text":"2025 Batch Placement Data\nTable:\nTotal Number of students placed till date | 214\nHighest Salary | 60.34 LPA\nAverage Salary | 6.78 LPA\nMedian Salary | 5 LPA\nTable:\nTotal Students | 298\nEligible Students | 286\nPlaced Students | 218\nHighest Salary till date | 22 LPA\nAverage Salary | 6.54 LPA\nMedian Salary | 5.75 LPA",
             "metadata":{"title":"Computer Engineering Placements","url":"https://computer.pccoepune.com/placements","source_key":"https://computer.pccoepune.com/placements","category":"Placements and careers","department":"Computer Engineering"},"rank_score":0.9}
        result=_format_placement_answer("What were the previous year Computer Engineering placements?",[hit])
        self.assertIsNotNone(result)
        self.assertIn("2025 Batch Placement Summary",result[0])
        self.assertIn("Students placed: 214",result[0])
        self.assertIn("Highest salary: 60.34 LPA",result[0])

    def test_faculty_profile_does_not_present_a_teaching_assistant_as_faculty_list(self):
        hit={"text":"FACULTY PROFILE\nNot Found\nMr. Rahul Salunkhe\nDesignation | Teaching Assistant","metadata":{"title":"Computer Engineering Faculty Profiles","url":"https://computer.pccoepune.com/faculty-profile","source_key":"https://computer.pccoepune.com/faculty-profile","category":"Faculty and official contacts","department":"Computer Engineering"},"rank_score":0.9}
        result=_format_faculty_answer("List Computer Engineering department faculty",[hit])
        self.assertIsNotNone(result)
        self.assertIn("does not currently provide a faculty roster",result[0])
        self.assertNotIn("Mr. Rahul Salunkhe",result[0])

    def test_fee_formatter_separates_approved_fee_from_proposal(self):
        url="https://www.pccoepune.com/pdf/B.Tech-FRA-Fee-Proposal-2026-27.pdf"
        doc="Academic Year Fee Status Tuition Fee Development Fee Total Fee\nFee for Academic Year 2025-26 Approved 153044 22956 176000\nFee Proposed by College for AY 2026-27 Proposal Status Y and Proposed fee for 2026-27 Rs. 194717"
        meta={"title":"B.Tech-FRA-Fee-Proposal-2026-27.pdf","url":url,"source_key":url,"category":"Tuition and fees","type":"PDF","department":"Institution-wide","page":1}
        hit={"text":doc,"metadata":meta,"rank_score":0.9,"score":0.9}
        with patch("app.main.search",return_value=[hit]), patch("app.main.collection.get",return_value={"documents":[doc],"metadatas":[meta]}):
            result=_format_fee_answer("What is the annual engineering fee at PCCOE?",[hit])
        self.assertIsNotNone(result)
        self.assertIn("2025-2026: ₹176,000",result[0])
        self.assertIn("2026-27: ₹194,717",result[0])
        self.assertIn("approval is not established",result[0])


if __name__=="__main__":
    unittest.main()
