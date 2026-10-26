import json
import asyncio
import sqlite3
import unittest
import tempfile
import httpx
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, Mock, call, patch

from pypdf import PdfWriter

from app.store import _chunks, _terms
from app.main import _PCCOE_SCOPE, _admission_process_chunks, _compact_evidence, _coverage_metadatas, _format_clubs_answer, _format_faculty_answer, _format_fee_answer, _format_hod_contact_answer, _format_placement_answer, _format_syllabus_answer, _indexed_metadatas, _syllabus_course_rows, _translate_grounded_answer, answer, generate_extractively, source_references
from app.ingest import extract
from app.crawler import _crawlable_url, _public_host, _retryable_urls, _prioritize_unseen_seeds, _retry_unindexed_faculty_rosters, _seed_urls, load_robots, request_stop
from app.agent import PCCOEKnowledgeAgent, _looks_multilingual, _model_query_plan
from app.config import settings
from app.taxonomy import classify_category, infer_department, extract_academic_year, resolve_department
from app.query_understanding import clarify_if_underspecified, detect_fee_subtopic, scope_clarification_followup, analyze_query, syllabus_course_terms
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

    def test_syllabus_program_correction_keeps_original_question_context(self):
        history=[
            {"role":"user","content":"give me syllabus of machine learning from computer science and engineering(AI and ML)"},
            {"role":"assistant","content":"M.Tech Computer Engineering Machine Learning syllabus"},
        ]
        contextual=scope_clarification_followup("not m tech i want for b tech",history)
        self.assertEqual(contextual,"give me syllabus of machine learning from computer science and engineering(AI and ML)\nScope clarification: not m tech i want for b tech")
        plan=analyze_query(contextual)
        self.assertEqual(plan["topic"],"syllabus")
        self.assertEqual(plan["program"],"B.Tech")
        self.assertEqual(plan["department"],"Artificial Intelligence and Machine Learning")

    def test_syllabus_program_detection_does_not_mistake_mtech_for_btech(self):
        self.assertEqual(analyze_query("M.Tech Machine Learning syllabus for Computer Engineering")["program"],"M.Tech")
        corrected="M.Tech Computer Engineering syllabus; not M.Tech, I want B.Tech"
        self.assertEqual(analyze_query(corrected)["program"],"B.Tech")

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

    def test_broad_syllabus_question_requests_missing_scope_before_search(self):
        clarification=clarify_if_underspecified("What is the syllabus of PCCOE?")
        self.assertIn("department",clarification)
        self.assertIn("program",clarification)
        self.assertIn("year/semester",clarification)
        self.assertIn("year/semester",clarify_if_underspecified("Civil B.Tech syllabus"))
        self.assertIn("program",clarify_if_underspecified("Civil third year syllabus"))
        self.assertIn("department",clarify_if_underspecified("B.Tech third year syllabus"))
        self.assertIsNone(clarify_if_underspecified("Civil B.Tech third year syllabus"))

    def test_model_planner_can_choose_search_instead_of_local_clarification(self):
        history=[
            {"role":"user","content":f"Previous question {index}"}
            for index in range(12)
        ]
        with patch.object(settings,"llm_provider","groq"), \
             patch.object(settings,"groq_api_key","unit-test-key"), \
             patch("app.main._model_query_plan",return_value=(("Civil Engineering syllabus third year",),None,True)) as planner, \
             patch("app.main.knowledge_agent.retrieve",return_value=[]) as retrieve, \
             patch("app.main._live_web_answer",return_value=None):
            answer("Civil third year syllabus",history)
        planner.assert_called_once()
        self.assertIn("user: Previous question 11",planner.call_args.args[1])
        retrieve.assert_called_once()

    def test_model_search_queries_do_not_override_user_intent(self):
        question="What is the PCCOE attendance requirement, and what happens if I fall short?"
        with patch.object(settings,"llm_provider","mock"), \
             patch("app.main._model_query_plan",return_value=(("PCCOE exam timetable 2026-27",),None,True)), \
             patch("app.main.knowledge_agent.retrieve",return_value=[]) as retrieve, \
             patch("app.main._refresh_retrieved_sources"), \
             patch("app.main._schedule_answer") as schedule:
            answer(question)
        retrieve.assert_called_once()
        self.assertEqual(retrieve.call_args.args[0],question)
        schedule.assert_not_called()

    def test_extractive_fallback_keeps_relevant_chunks_from_same_policy_page(self):
        question="What is the PCCOE attendance requirement, and what happens if I fall short?"
        url="https://www.pccoepune.com/pdf/RR/B.Tech-R&R-V2.4.pdf"
        hits=[
            {"text":"6.3 Attendance Requirement: The Institute expects all B.Tech students to attend 100% lectures. A shortfall of up to 25% may be condoned for valid reasons with appropriate evidence. A minimum attendance of 75% in each course is required.","metadata":{"title":"B.Tech-R&R-V2.4.pdf","url":url,"page":12,"category":"Admissions","chunk":0},"rank_score":0.6,"score":0.6,"coverage":0.4,"line_coverage":0.3},
            {"text":"iii. A student who has not attended the minimum 75% of conducted classes shall be declared Detained and shall not be permitted to attend the SA.","metadata":{"title":"B.Tech-R&R-V2.4.pdf","url":url,"page":12,"category":"Admissions","chunk":1},"rank_score":0.2,"score":0.2,"coverage":0.01,"line_coverage":0.01},
        ]
        with patch.object(settings,"llm_provider","mock"), \
             patch("app.main._model_query_plan",return_value=((),None,False)), \
             patch("app.main.knowledge_agent.retrieve",return_value=hits), \
             patch("app.main._refresh_retrieved_sources"):
            result=answer(question)
        self.assertIn("75%",result["answer"])
        self.assertIn("detained",result["answer"].lower())
        self.assertIn("condoned",result["answer"].lower())
        self.assertEqual(result["generation"],"extractive-fallback")
        self.assertTrue(result["fallback"])

    def test_department_name_does_not_imply_btech_program(self):
        question="give me syllabus of computer department Computer Engineering"
        self.assertIsNone(analyze_query(question)["program"])
        clarification=clarify_if_underspecified(question)
        self.assertIn("program",clarification)
        self.assertIn("year/semester",clarification)

    def test_syllabus_context_is_collected_across_multiple_followups(self):
        first_history=[
            {"role":"user","content":"What is the syllabus of PCCOE?"},
            {"role":"assistant","content":"Which PCCOE department, program (B.Tech, M.Tech, or MCA), and year/semester should I use for the syllabus?"},
        ]
        second_context=scope_clarification_followup("Civil Engineering",first_history)
        self.assertIsNotNone(second_context)
        self.assertIn("Civil Engineering",second_context)
        second_history=first_history+[
            {"role":"user","content":"Civil Engineering"},
            {"role":"assistant","content":"Which program (B.Tech, M.Tech, or MCA) and year/semester should I use for the syllabus?"},
        ]
        final_context=scope_clarification_followup("B.Tech third year",second_history)
        self.assertIsNotNone(final_context)
        self.assertIn("Civil Engineering",final_context)
        self.assertIsNone(clarify_if_underspecified(final_context))

    def test_year_and_semester_followup_preserves_syllabus_context(self):
        original="give me syllabus of computer department"
        history=[
            {"role":"user","content":original},
            {"role":"assistant","content":"Which year/semester should I use for the syllabus?"},
        ]
        contextual=scope_clarification_followup("third year 6th sem",history)
        self.assertIsNotNone(contextual)
        self.assertIn(original,contextual)
        clarification=clarify_if_underspecified("third year 6th sem",contextual)
        self.assertIsNotNone(clarification)
        self.assertIn("program",clarification)
        with patch.object(settings,"llm_provider","mock"), patch("app.main.knowledge_agent.retrieve") as retrieve:
            result=answer("third year 6th sem",history)
        self.assertEqual(result["generation"],"clarification")
        self.assertIn("program",result["answer"])
        retrieve.assert_not_called()

    def test_multiple_syllabus_followups_accumulate_scope_until_complete(self):
        history=[
            {"role":"user","content":"What is the syllabus of PCCOE?"},
            {"role":"assistant","content":"Which PCCOE department, program (B.Tech, M.Tech, or MCA), and year/semester should I use for the syllabus?"},
            {"role":"user","content":"Civil Engineering"},
            {"role":"assistant","content":"Which program (B.Tech, M.Tech, or MCA) and year/semester should I use for the syllabus?"},
            {"role":"user","content":"B.Tech"},
            {"role":"assistant","content":"Which year/semester should I use for the syllabus?"},
        ]
        contextual=scope_clarification_followup("third year 6th sem",history)
        self.assertIsNotNone(contextual)
        self.assertIn("What is the syllabus of PCCOE?",contextual)
        self.assertIn("Civil Engineering",contextual)
        self.assertIn("B.Tech",contextual)
        self.assertEqual(analyze_query(contextual)["department"],"Civil Engineering")
        self.assertEqual(analyze_query(contextual)["program"],"B.Tech")
        self.assertIsNone(clarify_if_underspecified(contextual))

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
            queries,clarification,planned=_model_query_plan("What is the admission process of PCCOE?")
        self.assertEqual(queries,("PCCOE B.Tech CAP admission steps",))
        self.assertEqual(clarification,"Which B.Tech admission route do you mean?")
        self.assertTrue(planned)
        _model_query_plan.cache_clear()

    def test_model_query_plan_uses_recent_conversation_and_open_ended_reasoning(self):
        class FakeResponse:
            def raise_for_status(self): pass
            def json(self):
                return {"candidates":[{"content":{"parts":[{"text":json.dumps({
                    "queries":["PCCOE attendance minimum percentage regulation","PCCOE short attendance detained ETE consequences"],
                    "clarification":None,
                })}]}}]}
        context="user: What are the PCCOE attendance requirements?\nassistant: I will check the official rules."
        with patch.object(settings,"llm_provider","gemini"), patch.object(settings,"gemini_api_key","unit-test-key"), patch("app.agent.httpx.post",return_value=FakeResponse()) as request:
            _model_query_plan.cache_clear()
            queries,clarification,planned=_model_query_plan("What if I don't meet them?",context)
        prompt=request.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]
        self.assertIn("What are the PCCOE attendance requirements?",prompt)
        self.assertIn("What if I don't meet them?",prompt)
        self.assertIn("Do not follow a fixed checklist",prompt)
        self.assertEqual(queries,("PCCOE attendance minimum percentage regulation","PCCOE short attendance detained ETE consequences"))
        self.assertIsNone(clarification)
        self.assertTrue(planned)
        _model_query_plan.cache_clear()

    def test_model_query_plan_fails_over_from_rate_limited_gemini_to_groq(self):
        class RateLimitedResponse:
            def raise_for_status(self):
                raise httpx.ConnectError("rate limited")
        class GroqResponse:
            def raise_for_status(self): pass
            def json(self):
                return {"choices":[{"message":{"content":json.dumps({
                    "queries":["PCCOE B.Tech attendance regulation threshold"],
                    "clarification":None,
                })}}]}
        with patch.object(settings,"llm_provider","gemini"), \
             patch.object(settings,"gemini_api_key","unit-test-gemini-key"), \
             patch.object(settings,"groq_api_key","unit-test-groq-key"), \
             patch("app.agent.httpx.post",side_effect=[RateLimitedResponse(),GroqResponse()]) as request:
            _model_query_plan.cache_clear()
            queries,clarification,planned=_model_query_plan("What are the PCCOE attendance rules?")
        self.assertEqual(request.call_count,2)
        self.assertEqual(queries,("PCCOE B.Tech attendance regulation threshold",))
        self.assertIsNone(clarification)
        self.assertTrue(planned)
        _model_query_plan.cache_clear()

    def test_attendance_search_does_not_assume_a_fixed_policy_threshold(self):
        queries=[]
        def fake_search(query,limit=5):
            queries.append(query)
            return []
        question="What are the PCCOE attendance requirements and consequences if I miss them?"
        with patch.object(settings,"llm_provider","mock"), \
             patch("app.agent._model_query_variants",return_value=()), \
             patch("app.agent.search",side_effect=fake_search):
            PCCOEKnowledgeAgent().retrieve(question,limit=5)
        self.assertIn(question,queries)
        self.assertFalse(any("minimum 75 percent" in query or "detained not permitted" in query for query in queries))

    def test_search_rewrites_cannot_outrank_matching_user_query_evidence(self):
        question="What is the PCCOE attendance requirement?"
        btech={
            "text":"The B.Tech rules specify an attendance threshold.",
            "metadata":{"title":"B.Tech Rules and Regulations","url":"https://www.pccoepune.com/btech-rules.pdf","source_id":"btech","chunk":0},
            "rank_score":0.44,"score":0.44,"coverage":0.5,"line_coverage":0.5,
        }
        bvoc={
            "text":"The B.Voc rules specify a different attendance threshold.",
            "metadata":{"title":"B.Voc Rules and Regulations","url":"https://www.pccoepune.com/bvoc-rules.pdf","source_id":"bvoc","chunk":0},
            "rank_score":0.80,"score":0.80,"coverage":0.8,"line_coverage":0.8,
        }
        def fake_search(query,limit=5):
            if query==question: return [dict(btech)]
            if "B.Voc attendance threshold" in query: return [dict(bvoc)]
            return []
        with patch.object(settings,"llm_provider","mock"), \
             patch("app.agent._model_query_variants",return_value=("B.Voc attendance threshold",)), \
             patch("app.agent.search",side_effect=fake_search), \
             patch.object(PCCOEKnowledgeAgent,"_expand_relevant_sources",side_effect=lambda _question,hits:hits):
            hits=PCCOEKnowledgeAgent().retrieve(question,limit=5)
        self.assertEqual(hits[0]["metadata"]["title"],"B.Tech Rules and Regulations")

    def test_policy_retrieval_expands_to_relevant_sections_in_same_regulation(self):
        question="What is the PCCOE attendance requirement, and what happens if I fall short?"
        url="https://www.pccoepune.com/pdf/RR/B.Tech-R&R-V2.4.pdf"
        metadata={"title":"B.Tech-R&R-V2.4.pdf","url":url,"source_key":url,"source_id":"btech-rules","category":"Admissions","type":"PDF","chunk":0}
        attendance={"text":"6.3 Attendance Requirement: Minimum attendance is 75% in each course.","metadata":metadata,"rank_score":0.6,"score":0.6,"line_coverage":0.6}
        detention="6.7 Detention: A student below 75% attendance shall be detained and not permitted to attend the SA and Practical/Oral Exams."
        detention_metadata={**metadata,"chunk":1}
        stored={"ids":["attendance","detention"],"documents":[attendance["text"],detention],"metadatas":[metadata,detention_metadata]}
        with patch("app.agent.collection.get",return_value=stored):
            expanded=PCCOEKnowledgeAgent()._expand_relevant_sources(question,[attendance])
        self.assertTrue(any("detained" in hit["text"].lower() for hit in expanded))

    def test_admission_retrieval_expands_to_official_process_terms(self):
        queries=[]
        def fake_search(query,limit=5):
            queries.append(query)
            return []
        with patch.object(settings,"llm_provider","mock"), patch("app.agent.search",side_effect=fake_search):
            PCCOEKnowledgeAgent().retrieve("What is the PCCOE admission process specifically for B Tech?",limit=5)
        self.assertTrue(any("eligibility CAP round required documents" in query for query in queries))
        self.assertIn("PCCOE first year B.Tech CAP admission process seat acceptance ERP admission form eligibility",queries)

    def test_course_syllabus_retrieval_adds_program_and_department_search(self):
        queries=[]
        def fake_search(query,limit=5):
            queries.append(query)
            return []
        question="give me syllabus of machine learning from computer science and engineering(AI and ML) B.Tech"
        with patch.object(settings,"llm_provider","mock"), patch("app.agent.search",side_effect=fake_search):
            PCCOEKnowledgeAgent().retrieve(question,limit=5)
        self.assertIn("AI ML B.Tech machine learning course syllabus PCCOE",queries)

    def test_department_year_syllabus_is_not_filtered_as_a_named_course(self):
        question="what is syllabus of b tech Electronics and Telecommunication department third year"
        self.assertEqual(syllabus_course_terms(question),set())
        self.assertEqual(
            syllabus_course_terms("give me syllabus of machine learning from computer science and engineering(AI and ML) B.Tech"),
            {"machine","learning"},
        )
        url="https://entc.pccoepune.com/pdf/UG-ETC-TY-2024-25.pdf"
        text="Course Syllabus\nT.Y. B.Tech. E&TC Engineering\nSemester-V\nCourse Code | Course Name"
        meta={"title":"UG-ETC-TY-2024-25.pdf","url":url,"source_key":url,
              "source_id":url+"::page:19","chunk":0,"department":"Electronics and Telecommunication",
              "category":"Syllabus and curriculum","type":"PDF","page":19}
        hit={"text":text,"metadata":meta,"score":0.8,"rank_score":0.8,
             "coverage":0.8,"line_coverage":0.8}
        with patch.object(settings,"llm_provider","mock"), \
             patch("app.agent.search",return_value=[hit]), \
             patch("app.agent.collection.get",return_value={"documents":[text],"metadatas":[meta]}):
            hits=PCCOEKnowledgeAgent().retrieve(question,limit=5)
        self.assertTrue(hits)
        self.assertTrue(any(row["metadata"]["url"]==url for row in hits))

    def test_department_year_syllabus_search_targets_semester_structure(self):
        queries=[]
        def fake_search(query,limit=5):
            queries.append(query)
            return []
        question="what is syllabus of b tech Electronics and Telecommunication department third year"
        with patch.object(settings,"llm_provider","mock"), patch("app.agent.search",side_effect=fake_search), patch("app.agent.datetime") as mocked_datetime:
            mocked_datetime.now.return_value.year=2026
            PCCOEKnowledgeAgent().retrieve(question,limit=5)
        self.assertIn("entc B.Tech syllabus semester V VI curriculum structure",queries)
        self.assertIn("entc B.Tech syllabus semester V VI 2026-27 curriculum structure",queries)

    def test_department_year_syllabus_prefers_latest_indexed_academic_year(self):
        question="what is syllabus of b tech Electronics and Telecommunication department third year"
        def make_hit(url,title,year,score,page):
            meta={"title":title,"url":url,"source_key":url,"source_id":f"{url}::page:{page}",
                  "chunk":0,"department":"Electronics and Telecommunication",
                  "category":"Syllabus and curriculum","type":"PDF","page":page,
                  "academic_year":year}
            return {"text":"Third Year B.Tech E&TC Engineering Semester V curriculum structure",
                    "metadata":meta,"score":score,"rank_score":score,"coverage":0.8,"line_coverage":0.8}
        old_hit=make_hit("https://entc.pccoepune.com/pdf/TY2526.pdf","TY2526.pdf","2025-2026",0.78,6)
        latest_hit=make_hit("https://entc.pccoepune.com/pdf/UG-E&TC-TY-2026-27.pdf","UG-E&TC-TY-2026-27.pdf","2026-2027",0.75,6)
        with patch.object(settings,"llm_provider","mock"), \
             patch("app.agent.search",return_value=[old_hit,latest_hit]), \
             patch.object(PCCOEKnowledgeAgent,"_expand_relevant_sources",side_effect=lambda question,hits:hits), \
             patch("app.agent.datetime") as mocked_datetime:
            mocked_datetime.now.return_value.year=2026
            hits=PCCOEKnowledgeAgent().retrieve(question,limit=5)
        self.assertEqual(hits[0]["metadata"]["url"],latest_hit["metadata"]["url"])

    def test_syllabus_parser_joins_wrapped_civil_course_names_and_electives(self):
        text="""Semester V
BCI25PC23 PCC Hydrology &
Water Resources 3 - - 3 3 - - 1 4 20 20 60 - - - 100
BCI25PC13 PCC Hydrology &
Water Resources - 1 - 1 - 2 - - 2 - - - 25 - 25 50
BCI25PE01/03/05/07 PEC PEC 1 3 - - 3 3 - - 1 4 20 20 60 - - - 100
BCI25PE02/04/06/08 PEC PEC 1 Lab - 1 - 1 2 - - 2 - - - 50 - - 50"""
        rows=_syllabus_course_rows(text)
        self.assertIn(("BCI25PC23","PCC Hydrology & Water Resources"),rows)
        self.assertIn(("BCI25PC13","PCC Hydrology & Water Resources"),rows)
        self.assertIn(("BCI25PE01/03/05/07","Program Elective 1"),rows)
        self.assertIn(("BCI25PE02/04/06/08","Program Elective 1 Lab"),rows)

    def test_syllabus_parser_joins_multi_line_civil_structural_subject_names(self):
        text="""Semester VI
BCI26PC26 PCC Design of
Reinforced 3 - - 3 3 - - 1 4 20 20 60 - - - 100
Concrete Structures
BCI26PC21 PCC Design of
Reinforced - 1 - 1 - 2 - - 2 - - - 25 - 25 50
Concrete Structures Lab"""
        rows=_syllabus_course_rows(text)
        self.assertIn(("BCI26PC26","PCC Design of Reinforced Concrete Structures"),rows)
        self.assertIn(("BCI26PC21","PCC Design of Reinforced Concrete Structures Lab"),rows)

    def test_syllabus_parser_reads_variable_elective_course_codes(self):
        text="""Semester V
BET25PExx PEC PEC1 2 _ _ 2 2 _ _ 1 10 10 30 _ _ _ 50
BET25PExx PEC PEC1 Lab _ 1 _ 1 _ 2 _ _ _ _ _ 50 _ _ 50
BET25MDxx MDM MDM3 3 _ _ 3 3 _ _ 1 20 20 60 _ _ _ 100"""
        rows=_syllabus_course_rows(text)
        self.assertIn(("BET25PExx","Program Elective 1"),rows)
        self.assertIn(("BET25PExx","Program Elective 1 Lab"),rows)
        self.assertIn(("BET25MDxx","MDM MDM3"),rows)

    def test_department_year_syllabus_is_not_rejected_as_course_level_lookup(self):
        question="what is syllabus of b tech Electronics and Telecommunication department third year"
        url="https://entc.pccoepune.com/pdf/UG-ETC-TY-2024-25.pdf"
        hit={"text":"Course Syllabus T.Y. B.Tech E&TC Engineering Semester-V",
             "metadata":{"title":"UG-ETC-TY-2024-25.pdf","url":url,"source_key":url,
                         "source_id":url+"::page:19","chunk":0,"department":"Electronics and Telecommunication",
                         "category":"Syllabus and curriculum","type":"PDF","page":19},
             "score":0.9,"rank_score":0.9,"coverage":0.9,"line_coverage":0.9}
        with patch.object(settings,"llm_provider","mock"), \
             patch("app.main.knowledge_agent.retrieve",return_value=[hit]), \
             patch("app.main._refresh_retrieved_sources"), \
             patch("app.main._format_syllabus_answer",return_value=("Third-year syllabus.",[hit])), \
             patch("app.main._compose_grounded_answer",return_value=("Syllabus available.",[], "mock")):
            result=answer(question)
        self.assertEqual(result["answer"],"Syllabus available.")

    def test_course_syllabus_search_targets_ai_ml_btech_source(self):
        queries=[]
        def fake_search(query,limit=5):
            queries.append(query)
            return []
        question="give me syllabus of machine learning from computer science and engineering(AI and ML)"
        with patch.object(settings,"llm_provider","mock"), patch("app.agent.search",side_effect=fake_search):
            PCCOEKnowledgeAgent().retrieve(question,limit=5)
        self.assertIn("AI ML B.Tech machine learning course syllabus PCCOE",queries)

    def test_syllabus_program_correction_searches_for_requested_btech(self):
        queries=[]
        def fake_search(query,limit=5):
            queries.append(query)
            return []
        question=("give me syllabus of machine learning from computer science and engineering(AI and ML)\n"
                  "Scope clarification: not m tech i want for b tech")
        with patch.object(settings,"llm_provider","mock"), patch("app.agent.search",side_effect=fake_search):
            PCCOEKnowledgeAgent().retrieve(question,limit=5)
        self.assertIn("AI ML B.Tech machine learning course syllabus PCCOE",queries)
        self.assertFalse(any("M.Tech machine learning course syllabus" in query for query in queries))

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
              {"url":"https://x/quota","error":"OCR failed (Gemini ClientError; Groq RateLimitError)"},
              {"url":"https://x/cooldown","error":"OCR RateLimitError","retry_after":"2099-01-01T00:00:00+00:00"},
              {"url":"https://x/cooldown-expired","error":"HTTP 429","retry_after":"2000-01-01T00:00:00+00:00"},
              {"url":"http://mechanical.pccoepune.com/page","error":"RuntimeError: robots.txt returned HTTP 301 for http://mechanical.pccoepune.com"},
              {"url":"https://x/missing","error":"HTTP 404"},
              {"url":"https://x/denied","error":"HTTP 403"}]
        self.assertEqual(_retryable_urls(rows),[
            "https://x/timeout","https://x/reset","https://x/server","https://x/quota",
            "https://x/cooldown-expired","http://mechanical.pccoepune.com/page",
        ])

    def test_partial_crawl_restarts_configured_seeds_after_retryable_errors(self):
        class DenyAllParser:
            def can_fetch(self, agent, url): return False
        class Client:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return None

        urls=[]
        async def fake_load_robots(url, client, cache):
            urls.append(url)
            return DenyAllParser(), []

        seeds=["https://www.pccoepune.com/seed-a","https://www.pccoepune.com/seed-b"]
        retry=["https://www.pccoepune.com/previous-timeout"]
        with tempfile.TemporaryDirectory() as directory, \
             patch("app.crawler.running",False), \
             patch("app.crawler.last_sync",{"status":"partial","errors":[{"url":retry[0],"error":"ReadTimeout"}]}), \
             patch("app.crawler._read_json",return_value=None), \
             patch("app.crawler._write_json"), \
             patch("app.crawler._checkpoint_path",Path(directory)/"checkpoint.json"), \
             patch("app.crawler._status_path",Path(directory)/"status.json"), \
             patch("app.crawler._seed_urls",return_value=seeds), \
             patch("app.crawler._retryable_urls",return_value=retry), \
             patch("app.crawler.load_robots",new=AsyncMock(side_effect=fake_load_robots)), \
             patch("app.crawler.add_sitemap",new=AsyncMock()), \
             patch("app.crawler.httpx.AsyncClient",return_value=Client()), \
             patch.object(settings,"max_crawl_pages",100):
            result=asyncio.run(__import__("app.crawler",fromlist=["crawl"]).crawl())

        self.assertEqual(urls[1:],[*retry,*seeds])
        self.assertEqual(result["status"],"complete")

    def test_crawler_follows_only_in_scope_redirects_for_robots_and_sitemaps(self):
        class Response:
            def __init__(self,status,headers=None,body=b""):
                self.status_code=status
                self.headers=headers or {}
                self.is_redirect=status in {301,302,303,307,308}
                self.body=body
            async def __aenter__(self): return self
            async def __aexit__(self,*args): return None
            async def aiter_bytes(self):
                yield self.body
        class Client:
            def __init__(self,responses): self.responses=iter(responses); self.urls=[]
            def stream(self,method,url):
                self.urls.append(url)
                return next(self.responses)
        robots_body=b"User-agent: *\nDisallow: /private/\nSitemap: https://www.pccoepune.com/sitemap.xml\n"
        client=Client([
            Response(302,{"location":"https://www.pccoepune.com/robots.txt"}),
            Response(200,body=robots_body),
        ])
        cache={}
        with patch("app.crawler.safe_url",return_value=True):
            parser,sitemaps=asyncio.run(load_robots("http://pccoepune.com/",client,cache))
        self.assertEqual(client.urls,[
            "http://pccoepune.com/robots.txt","https://www.pccoepune.com/robots.txt",
        ])
        self.assertFalse(parser.can_fetch("PCCOEStudentAssistant","https://www.pccoepune.com/private/file.pdf"))
        self.assertEqual(sitemaps,["https://www.pccoepune.com/sitemap.xml"])

        out_of_scope=Client([Response(302,{"location":"https://example.org/robots.txt"})])
        with patch("app.crawler.safe_url",side_effect=lambda url:"pccoepune.com" in url):
            with self.assertRaisesRegex(RuntimeError,"out-of-scope redirect"):
                asyncio.run(load_robots("https://pccoepune.com/",out_of_scope,{}))

    def test_crawler_stop_requests_graceful_checkpointed_shutdown(self):
        with patch("app.crawler.running",True), patch("app.crawler._stop_requested",False):
            result=request_stop()
            self.assertEqual(result,{"status":"stopping","running":True})
            self.assertTrue(__import__("app.crawler",fromlist=["_stop_requested"])._stop_requested)
        with patch("app.crawler.running",False):
            self.assertEqual(request_stop(),{"status":"not_running","running":False})

    def test_crawler_skips_static_assets_but_keeps_pages_and_documents(self):
        for url in (
            "https://www.pccoepune.com/images/faculty.jpg",
            "https://www.pccoepune.com/assets/app.js",
            "https://www.pccoepune.com/styles/site.css",
            "https://www.pccoepune.com/icons/site.svg",
            "https://www.pccoepune.com/assets/images/fav/manifest.json",
        ):
            self.assertFalse(_crawlable_url(url),url)
        for url in (
            "https://www.pccoepune.com/faculty-profile",
            "https://www.pccoepune.com/admission.php?program=btech",
            "https://www.pccoepune.com/pdf/mechanical-faculty.pdf",
            "https://www.pccoepune.com/data/faculty.json",
        ):
            self.assertTrue(_crawlable_url(url),url)

    def test_dns_failures_are_rechecked_so_temporary_resolution_errors_recover(self):
        host="temporary-dns-failure.pccoepune.com"
        clock=[100]
        with patch.dict("app.crawler._dns_cache",{},clear=True), patch("app.crawler.socket.getaddrinfo",side_effect=[
            OSError("temporary DNS failure"),
            [(None,None,None,None,("8.8.8.8",0))],
        ]) as resolve, patch("app.crawler.time.monotonic",side_effect=lambda:clock[0]):
            self.assertFalse(_public_host(host))
            self.assertFalse(_public_host(host))
            clock[0]=116
            self.assertTrue(_public_host(host))
            self.assertTrue(_public_host(host))
        self.assertEqual(resolve.call_count,2)

    def test_crawler_requeues_a_seeded_faculty_pdf_missing_from_legacy_index(self):
        url="https://www.pccoepune.com/pdf/nba-records/Faculty-List-2023-to-2026-Mech.pdf"
        seen={url}
        with patch("app.crawler.collection.get",return_value={"ids":[]}):
            retry=_retry_unindexed_faculty_rosters([url],seen)
        queue=_prioritize_unseen_seeds(["https://www.pccoepune.com/old-page"],[url],seen,retry)
        seen.difference_update(retry)
        self.assertEqual(retry,[url])
        self.assertEqual(queue[0],url)
        self.assertNotIn(url,seen)

    def test_scanned_faculty_roster_ocr_uses_small_bounded_page_batches(self):
        writer=PdfWriter()
        for _ in range(5):
            writer.add_blank_page(width=612,height=792)
        data=BytesIO()
        writer.write(data)
        responses=[
            type("Response",(),{"text":"Ada Example | Professor | PhD | Thermal systems | Yes\n"*3})(),
            type("Response",(),{"text":"Sam Example | Associate Professor | PhD | Design | Yes\n"*3})(),
        ]
        url="https://www.pccoepune.com/pdf/nba-records/Faculty-List-2023-to-2026-Mech.pdf"
        with patch.object(settings,"gemini_api_key","test-key"), patch.object(settings,"gemini_model","test-model"), patch("google.genai.Client") as client_factory:
            client=client_factory.return_value
            client.models.generate_content.side_effect=responses
            docs=extract("Faculty-List-2023-to-2026-Mech.pdf",data.getvalue(),url)
        self.assertEqual(client.models.generate_content.call_count,2)
        self.assertIn("Pages 1-4",docs[0][0])
        self.assertIn("Pages 5-5",docs[0][0])
        self.assertIn("Sam Example",docs[0][0])
        self.assertEqual(docs[0][1]["extraction"],"gemini-ocr")
        indexed={"ids":["ocr-roster"],"documents":[docs[0][0]],"metadatas":[docs[0][1]]}
        with patch("app.main.collection.get",return_value=indexed):
            roster=_format_faculty_answer(
                "List of teachers of Mechanical department",
                [{"text":docs[0][0],"metadata":docs[0][1],"rank_score":1.0}],
            )
        self.assertIn("Ada Example",roster[0])
        self.assertIn("Sam Example",roster[0])
        self.assertEqual(client_factory.call_count,2)
        self.assertEqual(client_factory.call_args_list,[
            call(
                api_key="test-key",
                http_options={"timeout":60_000,"retry_options":{"attempts":1}},
            ),
            call(
                api_key="test-key",
                http_options={"timeout":60_000,"retry_options":{"attempts":1}},
            ),
        ])
        self.assertEqual(client.close.call_count,2)

    def test_scanned_faculty_roster_ocr_failure_is_reported_without_partial_indexing(self):
        writer=PdfWriter()
        for _ in range(5):
            writer.add_blank_page(width=612,height=792)
        data=BytesIO()
        writer.write(data)
        url="https://www.pccoepune.com/pdf/nba-records/Faculty-List-2023-to-2026-Mech.pdf"
        with patch.object(settings,"gemini_api_key","test-key"), patch.object(settings,"groq_api_key",""), patch("google.genai.Client") as client_factory:
            client=client_factory.return_value
            client.models.generate_content.side_effect=[
                type("Response",(),{"text":"A valid OCR batch with enough readable text."})(),
                TimeoutError("timed out"),
            ]
            with self.assertRaisesRegex(RuntimeError,"pages 5-5: TimeoutError"):
                extract("Faculty-List-2023-to-2026-Mech.pdf",data.getvalue(),url)
        self.assertEqual(client.close.call_count,2)

    def test_scanned_roster_falls_back_to_configured_vision_model_when_gemini_is_unavailable(self):
        writer=PdfWriter()
        for _ in range(5):
            writer.add_blank_page(width=612,height=792)
        data=BytesIO()
        writer.write(data)
        url="https://www.pccoepune.com/pdf/nba-records/Faculty-List-2023-to-2026-Mech.pdf"
        ocr_rows="Ada Example | Professor | PhD | Thermal systems | Yes\n"
        with patch.object(settings,"gemini_api_key","test-gemini-key"), patch.object(settings,"groq_api_key","test-groq-key"), patch.object(settings,"groq_ocr_model","unit-test-vision"), patch("google.genai.Client") as gemini_factory, patch("app.ingest._groq_ocr_pages",return_value=ocr_rows) as groq_ocr:
            gemini=gemini_factory.return_value
            gemini.models.generate_content.side_effect=RuntimeError("provider unavailable")
            docs=extract("Faculty-List-2023-to-2026-Mech.pdf",data.getvalue(),url)
        self.assertEqual(groq_ocr.call_count,3)
        self.assertEqual(groq_ocr.call_args_list[0].args[1],[0,1])
        self.assertEqual(groq_ocr.call_args_list[1].args[1],[2,3])
        self.assertEqual(groq_ocr.call_args.args[1],[4])
        self.assertEqual(docs[0][1]["extraction"],"groq-vision-ocr")
        self.assertEqual(docs[0][0].count("Ada Example"),3)
        gemini.close.assert_called_once()

    def test_groq_vision_retries_rate_limit_using_provider_cooldown(self):
        import httpx
        from groq import RateLimitError
        from app.ingest import _groq_completion_with_retry

        request=httpx.Request("POST","https://api.groq.com/openai/v1/chat/completions")
        response=httpx.Response(429,headers={"retry-after":"3"},request=request)
        error=RateLimitError("rate limited",response=response,body=None)
        expected=object()
        create=Mock(side_effect=[error,expected])
        with patch("app.ingest.time.sleep") as sleep:
            result=_groq_completion_with_retry(create,model="vision-test")
        self.assertIs(result,expected)
        self.assertEqual(create.call_count,2)
        sleep.assert_called_once_with(3.0)

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

    def test_english_clarification_is_not_rewritten_by_translation_model(self):
        clarification="Which program (B.Tech, M.Tech, or MCA) and year/semester should I use?"
        with patch.object(settings,"llm_provider","gemini"), patch.object(settings,"gemini_api_key","unit-test-key"), patch("httpx.post") as request:
            translated=_translate_grounded_answer("give me syllabus of computer department",clarification)
        self.assertEqual(translated,clarification)
        request.assert_not_called()

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

    def test_conversation_context_keeps_last_twelve_question_answer_pairs(self):
        data_dir=Path(__file__).resolve().parents[1]/"data"
        data_dir.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=data_dir) as directory:
            store=ConversationStore(directory)
            session=store.create()
            for index in range(15):
                store.append_turn(session,f"Question {index}",{"answer":f"Answer {index}","sources":[],"fallback":False})
            history=store.history(session)
        self.assertEqual(len(history),24)
        self.assertEqual(history[0]["content"],"Question 3")
        self.assertEqual(history[-1]["content"],"Answer 14")
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

    def test_syllabus_formatter_prefers_requested_btech_ai_ml_course_over_mtech(self):
        question="give me syllabus of machine learning from computer science and engineering(AI and ML) B.Tech"
        btech_url="https://aiml.pccoepune.com/assets/docs/Syllabus/SY-B.Tech-CSE-(AI-ML).pdf"
        mtech_url="https://computer.pccoepune.com/assets/pdf/M.Tech-Computer-Engg-syllabus-2022-23.pdf"
        btech_text=("Department of Computer Science & Engineering (AI & ML)\n"
                    "Program: B. Tech. CSE (AI & ML) Semester : IV Course : Machine Learning Code : BCS4409\n"
                    "Detailed Syllabus\nUnit I: Introduction to Machine Learning\nSupervised learning, regression, classification.\n"
                    "Unit II: Regression Analysis\nLinear regression, logistic regression.")
        continuation_text=("Unsupervised learning:\nIV Introduction to clustering, hierarchical and k-means clustering.\n"
                           "V Introduction to ANN, perceptron, multilayer perceptron and backpropagation.\n"
                           "VI Reinforcement Learning & Ensemble Method: bagging, boosting and stacking.\nTotal 45")
        mtech_text=("Program: M.Tech Computer Engineering Semester: II Course: Machine Learning Code: MCE2407\n"
                    "Unit I: Introduction to Advanced Machine Learning")
        btech_meta={"title":"SY-B.Tech-CSE-(AI-ML).pdf","url":btech_url,"source_key":btech_url,"source_id":btech_url+"::page:38","category":"Syllabus and curriculum","type":"PDF","department":"Artificial Intelligence and Machine Learning","page":38,"academic_year":"2024-2025","chunk":0}
        continuation_meta={**btech_meta,"page":39,"source_id":btech_url+"::page:39","chunk":0}
        mtech_meta={"title":"M.Tech-Computer-Engg-syllabus-2022-23.pdf","url":mtech_url,"source_key":mtech_url,"category":"Syllabus and curriculum","type":"PDF","department":"Computer Engineering","page":45,"chunk":0}
        hits=[
            {"text":mtech_text,"metadata":mtech_meta,"score":0.95,"rank_score":0.95},
            {"text":btech_text,"metadata":btech_meta,"score":0.65,"rank_score":0.65},
        ]
        with patch("app.main.collection.get",return_value={"documents":[btech_text,continuation_text],"metadatas":[btech_meta,continuation_meta]}):
            result=_format_syllabus_answer(question,hits)
        self.assertIsNotNone(result)
        self.assertIn("BCS4409",result[0])
        self.assertIn("Unit I: Introduction to Machine Learning",result[0])
        self.assertIn("IV Introduction to clustering",result[0])
        self.assertIn("V Introduction to ANN",result[0])
        self.assertIn("VI Reinforcement Learning",result[0])
        self.assertNotIn("MCE2407",result[0])
        self.assertEqual({hit["metadata"]["page"] for hit in result[1]},{38,39})

    def test_syllabus_formatter_reads_third_year_semester_tables(self):
        question="what is syllabus of b tech Electronics and Telecommunication department third year"
        url="https://entc.pccoepune.com/pdf/UG-ETC-TY-2024-25.pdf"
        text=("STRUCTURE FOR THIRD YEAR B. TECH (E&TC ENGINEERING)\n"
              "SEM-V\nB.TECH Semester-V\nCourse Course Course Name Teaching Evaluation Scheme\n"
              "BET5414 PCC Control System 2 - 1 3 3 20 30 50 - - - 100\n")
        meta={"title":"UG-ETC-TY-2024-25.pdf","url":url,"source_key":url,
              "source_id":url+"::page:12","chunk":0,"department":"Electronics and Telecommunication",
              "category":"Syllabus and curriculum","type":"PDF","page":12}
        hit={"text":text,"metadata":meta,"score":0.8,"rank_score":0.8}
        with patch("app.main.collection.get",return_value={"documents":[text],"metadatas":[meta]}):
            result=_format_syllabus_answer(question,[hit])
        self.assertIsNotNone(result)
        self.assertIn("Semester V",result[0])
        self.assertIn("BET5414",result[0])
        self.assertIn("Control System",result[0])

    def test_syllabus_formatter_reads_unicode_semester_and_underscore_marks(self):
        question="what is syllabus of b tech Electronics and Telecommunication department third year"
        url="https://entc.pccoepune.com/pdf/UG-E&TC-TY-2026-27.pdf"
        text=("CURRICULUM STRUCTURE\nThird Year B.Tech. (E&TC Engineering) Semester � V\n"
              "Course Course Course Name L P T Total\nCode Type\n"
              "BET25PC01 PCC Communication 2 _ _ 2 2 _ _ 1 10 10 30 _ _ _ 50\nsystems\n")
        meta={"title":"UG-E&TC-TY-2026-27.pdf","url":url,"source_key":url,
              "source_id":url+"::page:10","chunk":0,"department":"Electronics and Telecommunication",
              "category":"Syllabus and curriculum","type":"PDF","page":10,"academic_year":"2026-2027"}
        hit={"text":text,"metadata":meta,"score":0.8,"rank_score":0.8}
        with patch("app.main.collection.get",return_value={"documents":[text],"metadatas":[meta]}):
            result=_format_syllabus_answer(question,[hit])
        self.assertIsNotNone(result)
        self.assertIn("Semester V",result[0])
        self.assertIn("BET25PC01 - PCC Communication systems",result[0])

    def test_syllabus_formatter_prefers_latest_academic_year_when_unspecified(self):
        question="what is syllabus of b tech Electronics and Telecommunication department third year"
        old_url="https://entc.pccoepune.com/pdf/UG-ETC-TY-2024-25.pdf"
        latest_url="https://entc.pccoepune.com/pdf/UG-E&TC-TY-2026-27.pdf"
        old_text=("Curriculum Structure\nThird Year B.Tech E&TC Engineering Semester-V\n"
                  "Course Code Course Name\nBET24PC01 PCC Old Course\n")
        latest_text=("Curriculum Structure\nThird Year B.Tech E&TC Engineering Semester � V\n"
                     "Course Code Course Name\nBET26PC01 PCC Current Course\n")
        def make_hit(url,title,year,text,page):
            meta={"title":title,"url":url,"source_key":url,"source_id":f"{url}::page:{page}",
                  "chunk":0,"department":"Electronics and Telecommunication",
                  "category":"Syllabus and curriculum","type":"PDF","page":page,
                  "academic_year":year}
            return {"text":text,"metadata":meta,"score":0.8,"rank_score":0.8}
        old_hit=make_hit(old_url,"UG-ETC-TY-2024-25.pdf","2024-2025",old_text,10)
        latest_hit=make_hit(latest_url,"UG-E&TC-TY-2026-27.pdf","2026-2027",latest_text,10)
        documents={old_url:(old_text,old_hit["metadata"]),latest_url:(latest_text,latest_hit["metadata"])}
        def get_source(*,where,include):
            text,meta=documents[where["source_key"]]
            return {"documents":[text],"metadatas":[meta]}
        with patch("app.main.collection.get",side_effect=get_source):
            result=_format_syllabus_answer(question,[old_hit,latest_hit])
        self.assertIsNotNone(result)
        self.assertIn("Current Course",result[0])
        self.assertNotIn("Old Course",result[0])
        self.assertEqual({hit["metadata"]["url"] for hit in result[1]},{latest_url})

    def test_btech_syllabus_correction_returns_requested_degree_and_units(self):
        question="give me syllabus of machine learning from computer science and engineering(AI and ML)"
        btech_url="https://aiml.pccoepune.com/assets/docs/Syllabus/SY-B.Tech-CSE-(AI-ML).pdf"
        mtech_url="https://computer.pccoepune.com/assets/pdf/M.Tech-Computer-Engg-syllabus-2022-23.pdf"
        btech_text=("Department of Computer Science & Engineering (AI & ML)\n"
                    "Program: B. Tech. CSE (AI & ML) Semester : IV Course : Machine Learning Code : BCS4409\n"
                    "Detailed Syllabus:\nUnit I: Basics of machine learning\nSupervised, unsupervised and reinforcement learning.\n"
                    "Unit II: Regression techniques\nLinear, ridge and lasso regression.")
        mtech_text="Program: M.Tech Computer Engineering Course: Machine Learning Code: MCE2407"
        btech_meta={"title":"SY-B.Tech-CSE-(AI-ML).pdf","url":btech_url,"source_key":btech_url,"category":"Syllabus and curriculum","type":"PDF","department":"Artificial Intelligence and Machine Learning","page":38,"chunk":0}
        mtech_meta={"title":"M.Tech-Computer-Engg-syllabus-2022-23.pdf","url":mtech_url,"source_key":mtech_url,"category":"Syllabus and curriculum","type":"PDF","department":"Computer Engineering","page":45,"chunk":0}
        history=[
            {"role":"user","content":question},
            {"role":"assistant","content":"M.Tech Computer Engineering Machine Learning syllabus"},
        ]
        with patch("app.main.knowledge_agent.retrieve",return_value=[
            {"text":mtech_text,"metadata":mtech_meta,"score":0.95,"rank_score":0.95,"coverage":0.9},
            {"text":btech_text,"metadata":btech_meta,"score":0.65,"rank_score":0.65,"coverage":0.7},
        ]), patch("app.main.collection.get",return_value={"documents":[btech_text],"metadatas":[btech_meta]}), patch("app.main._refresh_retrieved_sources"):
            result=answer("not m tech i want for b tech",history)
        self.assertIn("BCS4409",result["answer"])
        self.assertIn("Unit I: Basics of machine learning",result["answer"])
        self.assertNotIn("MCE2407",result["answer"])
        self.assertEqual(result["sources"][0]["url"],btech_url+"#page=38")
        self.assertEqual(syllabus_course_terms(question),{"machine","learning"})

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

    def test_faculty_roster_deduplicates_honorific_and_initial_variants(self):
        url="https://www.pccoepune.com/pdf/nba-records/Faculty-List-2023-to-2026-Mech.pdf"
        text=("Name | Designation | Qualification | Specialization | Currently Associated\n"
              "Dr. A. B. Example | Professor | PhD | Design | Yes\n"
              "Mr. A.B Example | Associate Professor | PhD | Design | Yes\n")
        meta={"title":"Faculty-List-2023-to-2026-Mech.pdf","url":url,"source_key":url,"type":"PDF","department":"Mechanical Engineering"}
        hit={"text":text,"metadata":meta,"rank_score":1.0}
        with patch("app.main.collection.get",return_value={"ids":["fixture"],"documents":[text],"metadatas":[meta]}):
            result=_format_faculty_answer("List of teachers of Mechanical department",[hit])
        self.assertIsNotNone(result)
        self.assertEqual(result[0].count("A. B. Example"),1)
        self.assertIn("Source: official PCCOE faculty records.",result[0])

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
