"""Run a transparent retrieval-coverage audit against the question bank.

This measures retrieval evidence only; it never treats a generated answer as proof.
Run with: python -m app.evaluate
"""
import json
import fnmatch
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .agent import knowledge_agent
from .store import search, collection

ROOT=Path(__file__).resolve().parent.parent
BANK=ROOT/"data"/"question_bank.json"
REPORT=ROOT/"data"/"evaluation_report.json"

def evaluate():
    bank=json.loads(BANK.read_text(encoding="utf-8"))
    results=[]
    stats=defaultdict(lambda:{"total":0,"retrieval_hits":0,"legacy_pattern_hits":0,"with_citation":0})
    language_stats=defaultdict(lambda:{"total":0,"retrieval_hits":0})
    for item in bank["questions"]:
        samples={"en":item["example_question"],**item.get("variants",{})}
        item_results=[]
        for language,query in samples.items():
            # Exercise the same rewrite, language normalization, hybrid
            # retrieval, and source expansion used by the live chatbot.
            hits=knowledge_agent.retrieve(query,limit=12)
            best=max((hit.get("coverage",0) for hit in hits),default=0)
            citations=sorted({(hit["metadata"].get("url",""),hit["metadata"].get("page")) for hit in hits if hit["metadata"].get("url") and hit.get("coverage",0)>0})
            expected_pattern=item.get("expected_source_pattern", "")
            pattern_match=any(fnmatch.fnmatch(url,expected_pattern) for url,_ in citations) if expected_pattern else bool(citations)
            expected_category=item.get("category","")
            category_match=any(hit["metadata"].get("category")==expected_category for hit in hits)
            source_domain_match=any(urlsplit(url).hostname and (urlsplit(url).hostname=="pccoepune.com" or urlsplit(url).hostname.endswith(".pccoepune.com")) for url,_ in citations)
            # Legacy threshold and URL glob remain visible for comparison.
            # Main hit requires meaningful term coverage and an official PCCOE
            # citation, allowing canonical pages outside stale URL globs.
            supported=bool(hits and best>=0.5 and source_domain_match and (category_match or pattern_match))
            legacy_supported=bool(hits and best>=0.6 and citations and pattern_match)
            item_results.append({"language":language,"query":query,"retrieval_hit":supported,"legacy_pattern_hit":legacy_supported,"best_passage_coverage":round(best,3),"expected_source_pattern_match":pattern_match,"expected_category_match":category_match,"official_pccoe_source":source_domain_match,"citations":[{"url":url,"page":page} for url,page in citations[:5]]})
            stats[item["category"]]["total"]+=1
            stats[item["category"]]["retrieval_hits"]+=int(supported)
            stats[item["category"]]["legacy_pattern_hits"]+=int(legacy_supported)
            stats[item["category"]]["with_citation"]+=int(bool(citations))
            language_stats[language]["total"]+=1
            language_stats[language]["retrieval_hits"]+=int(supported)
        results.append({"id":item["id"],"category":item["category"],"queries":item_results})
    summary={category:{**values,"retrieval_success_rate":round(values["retrieval_hits"]/values["total"],3) if values["total"] else 0} for category,values in sorted(stats.items())}
    by_language={language:{**values,"retrieval_success_rate":round(values["retrieval_hits"]/values["total"],3) if values["total"] else 0} for language,values in sorted(language_stats.items())}
    report={"created_at":datetime.now(timezone.utc).isoformat(),"index_chunks":collection.count(),"evaluation_scope":"live agent retrieval, term coverage, category relevance and official PCCOE citation; answer correctness is not implied","total_questions":sum(x["total"] for x in stats.values()),"total_retrieval_hits":sum(x["retrieval_hits"] for x in stats.values()),"legacy_pattern_hits":sum(x["legacy_pattern_hits"] for x in stats.values()),"by_category":summary,"by_language":by_language,"results":results}
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="results"},ensure_ascii=False,indent=2))
    print(f"Full report saved to {REPORT}")

if __name__=="__main__": evaluate()
