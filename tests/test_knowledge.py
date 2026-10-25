import json
import sqlite3
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from app.store import _chunks, _terms
from app.main import _PCCOE_SCOPE, _admission_process_chunks, _compact_evidence, _coverage_metadatas, _format_clubs_answer, _format_faculty_answer, _format_fee_answer, _format_hod_contact_answer, _format_placement_answer, _indexed_metadatas, _syllabus_course_rows, _translate_grounded_answer, answer, generate_extractively, source_references
from app.crawler import _retryable_urls, _prioritize_unseen_seeds, _seed_urls
from app.agent import PCCOEKnowledgeAgent, _looks_multilingual, _model_query_plan
from app.config import settings
from app.taxonomy import classify_category, infer_department, extract_academic_year, resolve_department
from app.query_understanding import clarify_if_underspecified, detect_fee_subtopic, scope_clarification_followup, analyze_query
from app.conversations import ConversationStore


class KnowledgePipelineTests(unittest.TestCase):
    def setUp(self):
        self.provider_patch=patch.object(settings,"llm_provider","mock")
        self.provider_patch.start()

    def tearDown(self):
        self.provider_patch.stop()

    def test_unprocessed_configured_seed_moves_ahead_of_saved_backlog(self):
        queue=_prioritize_unseen_seeds(
            ["old-page", "official-mechanical-roster", "later-page"],
            ["home", "official-mechanical-roster"],
            {"home"},
            ["stale-pdf-retry"],
        )
        self.assertEqual(list(queue),[
            "official-mechanical-roster",
            "stale-pdf-retry",
            "old-page",
            "later-page",
        ])

    def test_btech_program_clarification_keeps_the_admission_question_context(self):
        history=[
            {"role":"user","content":"What is the admission process of PCCOE?"},
            {"role":"assistant","content":"Which PCCOE admission route do you mean—B.Tech first year through CAP, direct second year/against CAP, MCA, or M.Tech?"},
        ]
        contextual=scope_clarification_followup("want specifically for b tech",history)
        self.assertEqual(contextual,"What is the admission process of PCCOE?\nScope clarification: want specifically for b tech")
        plan=analyze_query(contextual)
        self.assertEqual(plan["topic"],"admissions")
        self.assertEqual(plan["program"],"B.Tech")

    def test_followup_answer_uses_prior_admission_question_and_formats_steps(self):
        history=[
            {"role":"user","content":"What is the admission process of PCCOE?"},
            {"role":"assistant","content":"Which PCCOE admission route do you mean—B.Tech first year through CAP, direct second year/against CAP, MCA, or M.Tech?"},
        ]
        hit={
            "text":"First Year B.Tech CAP Admission\nStep 1: Complete seat acceptance through the CET login.\nStep 2: Fill out the ERP Admission Form and print it before reporting to PCCOE.",
            "metadata":{"title":"B.Tech Admissions PCCOE","url":"https://www.pccoepune.com/admission-home.php","source_key":"https://www.pccoepune.com/admission-home.php","category":"Admissions","type":"HTML"},
            "score":0.9,"rank_score":0.9,"coverage":0.9,"line_coverage":0.9,
        }
        with patch.object(settings,"llm_provider","mock"), patch("app.main.knowledge_agent.retrieve",return_value=[hit]) as retrieve, patch("app.main._refresh_retrieved_sources"), patch("app.main._live_web_answer",return_value=None):
            result=answer("want specifically for b tech",history)
        self.assertIn("Step 1: Complete seat acceptance",result["answer"])
        self.assertIn("Step 2: Fill out the ERP Admission Form",result["answer"])
        self.assertIn("admission process",retrieve.call_args.args[0].lower())

    def test_broad_admission_question_asks_for_program_route(self):
        clarification=clarify_if_underspecified("What is the admission process of PCCOE?")
        self.assertIn("Which PCCOE admission route",clarification)

    def test_model_query_plan_returns_search_terms_and_clarification(self):
        class FakeResponse:
            def raise_for_status(self): pass
            def json(self):
                return {"choices":[{"message":{"content":json.dumps({
                    "queries":["PCCOE B.Tech CAP admission steps"],
                    "clarification":"Which B.Tech admission route do you mean?",
                })}}]}
        with patch.object(settings,"llm_provider","groq"), patch.object(settings,"groq_api_key","unit-test-key"), patch("app.agent.httpx.post",return_value=FakeResponse()):
            _model_query_plan.cache_clear()
            queries,clarification=_model_query_plan("What is the admission process of PCCOE?")
        self.assertEqual(queries,("PCCOE B.Tech CAP admission steps",))
        self.assertEqual(clarification,"Which B.Tech admission route do you mean?")
        _model_query_plan.cache_clear()

    def test_admission_retrieval_expands_to_official_process_terms(self):
        queries=[]
        def fake_search(query,limit=5):
            queries.append(query)
            return []
        with patch.object(settings,"llm_provider","mock"), patch("app.agent.search",side_effect=fake_search):
            PCCOEKnowledgeAgent().retrieve("What is the PCCOE admission process specifically for B Tech?",limit=5)
        self.assertTrue(any("eligibility CAP round required documents" in query for query in queries))
        self.assertIn("PCCOE first year B.Tech CAP admission process seat acceptance ERP admission form eligibility",queries)

    def test_off_topic_questions_stay_inside_the_pccoe_assistant_scope(self):
        result=answer("How do I bake sourdough bread?")
        self.assertEqual(result["generation"],"out-of-scope")
        self.assertIn("focused on PCCOE",result["answer"])
        self.assertEqual(result["sources"],[])

    def test_core_btech_admission_pages_are_added_even_with_custom_seeds(self):
        with patch.object(settings,"college_url","https://www.pccoepune.com/"), patch.object(settings,"seed_urls","https://computer.pccoepune.com/"):
            seeds=_seed_urls()
        self.assertIn("https://www.pccoepune.com/admission-home.php",seeds)
        self.assertIn("https://www.pccoepune.com/against-cap-admissions.php",seeds)
        self.assertIn("https://www.pccoepune.com/institute-level-admissions.php",seeds)
        self.assertIn("https://www.pccoepune.com/cut-off.php",seeds)
        self.assertIn("https://computer.pccoepune.com/",seeds)

    def test_index_inventory_paginates_chroma_metadata(self):
        def fake_get(limit,offset,include):
            return {"metadatas":[{"url":str(index)} for index in range(offset,min(offset+limit,5))]}
        with patch("app.main.collection.count",return_value=5), patch("app.main.collection.get",side_effect=fake_get) as get:
            rows=list(_indexed_metadatas(batch_size=2))
        self.assertEqual(len(rows),5)
        self.assertEqual([call.kwargs["offset"] for call in get.call_args_list],[0,2,4])

    def test_coverage_inventory_reads_chroma_metadata_read_only(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()/"data") as directory:
            path=Path(directory)/"chroma.sqlite3"
            with sqlite3.connect(path) as connection:
                connection.executescript("""
                    CREATE TABLE collections(id TEXT,name TEXT);
                    CREATE TABLE segments(id TEXT,type TEXT,scope TEXT,collection TEXT);
                    CREATE TABLE embeddings(id INTEGER,segment_id TEXT);
                    CREATE TABLE embedding_metadata(id INTEGER,key TEXT,string_value TEXT,int_value INTEGER);
                    INSERT INTO collections VALUES('collection-id','pccoe_sources');
                    INSERT INTO segments VALUES('metadata-segment','metadata','METADATA','collection-id');
                    INSERT INTO embeddings VALUES(1,'metadata-segment');
                    INSERT INTO embedding_metadata VALUES(1,'url','https://www.pccoepune.com/faculty.php',NULL);
                    INSERT INTO embedding_metadata VALUES(1,'source_key','https://www.pccoepune.com/faculty.php',NULL);
                    INSERT INTO embedding_metadata VALUES(1,'title','Faculty',NULL);
                    INSERT INTO embedding_metadata VALUES(1,'category','People',NULL);
                    INSERT INTO embedding_metadata VALUES(1,'department','Institution-wide',NULL);
                    INSERT INTO embedding_metadata VALUES(1,'type','HTML',NULL);
                    INSERT INTO embedding_metadata VALUES(1,'page',NULL,2);
                """)
            connection.close()
            with patch.object(settings,"chroma_path",directory):
                rows=list(_coverage_metadatas())
            self.assertEqual(rows,[{"url":"https://www.pccoepune.com/faculty.php","source_key":"https://www.pccoepune.com/faculty.php","title":"Faculty","category":"People","department":"Institution-wide","type":"HTML","page":2}])

    def test_crawler_retries_transient_failures_but_not_permanent_http_errors(self):
        rows=[{"url":"https://x/timeout","error":""},
              {"url":"https://x/reset","error":"ReadTimeout: timed out"},
              {"url":"https://x/server","error":"HTTP 503"},
              {"url":"https://x/missing","error":"HTTP 404"}]
        self.assertEqual(_retryable_urls(rows),["https://x/timeout","https://x/reset","https://x/server"])

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
        evidence=[{"text":text,"metadata":{"title":title,"url":url,"type":"HTML","category":"Student activities and events","department":"Computer Engineering"},"rank_score":0.9}
                  for url,(title,text) in pages.items()]
        with patch("app.main.collection.get",side_effect=fake_get):
            result=_format_clubs_answer("List clubs in Computer department",evidence)
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
        self.assertIn("Dr. P. R. Kale",answer)
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
        self.assertIn("Dr. P. R. Kale",result[0])

    def test_hod_parser_ignores_mechanical_dean_rows_before_hod_table(self):
        url="https://www.pccoepune.com/heads-and-deans.php"
        roster={"text":"Associate Dean Administration - I | Dr. N. Vivekanandan | Mechanical\nHead of the Department / Program Heads\nMechanical Engineering | Dr. P.R. Kale | Professor Mechanical","metadata":{"title":"PCCOE Heads and Deans","url":url,"source_key":url,"department":"Institution-wide","type":"HTML"},"score":0.9,"rank_score":0.9}
        with patch("app.main.search",return_value=[]):
            result=_format_hod_contact_answer("Who is HOD of Mechanical department?",[roster])
        self.assertIsNotNone(result)
        self.assertIn("Dr. P. R. Kale",result[0])
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

    def test_multi_step_answers_keep_multiple_chunks_from_a_source_page(self):
        hits=[
            {"text":"Page overview","metadata":{"url":"https://www.pccoepune.com/admission.php"},"rank_score":0.9},
            {"text":"Admission step one","metadata":{"url":"https://www.pccoepune.com/admission.php"},"rank_score":0.8},
            {"text":"Admission step two","metadata":{"url":"https://www.pccoepune.com/admission.php"},"rank_score":0.7},
            {"text":"Another official admission source","metadata":{"url":"https://www.pccoepune.com/cap.php"},"rank_score":0.6},
        ]
        compact=_compact_evidence(hits,max_sources=2,max_chunks_per_source=2)
        self.assertEqual([hit["text"] for hit in compact],[
            "Page overview","Admission step one","Another official admission source",
        ])

    def test_admission_process_selection_keeps_step_and_form_instructions(self):
        hits=[
            {"text":"Step 1: Complete seat acceptance through the CET login."},
            {"text":"Login to ERP Pre-Admission Portal and enter Application ID."},
            {"text":"Admission enquiry contact and homepage navigation."},
        ]
        self.assertEqual(_admission_process_chunks(hits),hits[:2])

    def test_extractive_admission_answer_includes_supported_steps(self):
        evidence=[{
            "text":"Page description: PCCOE B.Tech admission process.\nStep 1: Complete seat acceptance (Freeze / Betterment) through your CET login.\nStep 2: Fill out the ERP Admission Form and print it before reporting to the institute.",
            "metadata":{"title":"B.Tech Admissions PCCOE","url":"https://www.pccoepune.com/admission-home.php","category":"Admissions"},
            "rank_score":0.5,
        }]
        result=generate_extractively("What is PCCOE B Tech admission process?",evidence)
        self.assertIn("Step 1: Complete seat acceptance",result)
        self.assertIn("Step 2: Fill out the ERP Admission Form",result)

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
        with patch("app.main.collection.get",return_value={"ids":["fixture"],"documents":[hit["text"]],"metadatas":[hit["metadata"]]}):
            result=_format_faculty_answer("List Computer Engineering department faculty",[hit])
        self.assertIsNotNone(result)
        self.assertIn("no parseable teacher cards",result[0])
        self.assertNotIn("Mr. Rahul Salunkhe",result[0])

    def test_faculty_refresh_reloads_full_source_when_fetch_adds_zero_chunks(self):
        url="https://computer.pccoepune.com/faculty-profile"
        meta={"title":"Computer Engineering Faculty Profiles","url":url,"source_key":url,"category":"Faculty and official contacts","department":"Computer Engineering","type":"HTML"}
        placeholder="FACULTY PROFILE\nNot Found"
        card="Dr. Ada Test\nDesignation | Assistant Professor\nEmail Id | ada.test@pccoepune.org"
        hit={"text":placeholder,"metadata":meta,"score":0.8,"rank_score":0.8}
        stored_placeholder={"documents":[placeholder],"metadatas":[meta]}
        stored_complete={"documents":[placeholder,card],"metadatas":[meta,meta]}
        with patch("app.main.knowledge_agent.retrieve",side_effect=[[hit],[hit]]), patch("app.main._refresh_retrieved_sources",return_value=False), patch("app.main.crawler.refresh_live_source",return_value=0), patch("app.main.collection.get",side_effect=[stored_placeholder,stored_complete]):
            result=answer("List names and contact of Computer Engineering teachers")
        self.assertEqual(result["generation"],"structured-faculty-roster")
        self.assertIn("Dr. Ada Test",result["answer"])
        self.assertIn("ada.test@pccoepune.org",result["answer"])

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
