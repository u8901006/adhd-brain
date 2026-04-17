#!/usr/bin/env python3
"""
Fetch latest ADHD research papers from PubMed E-utilities API.
Targets top ADHD-related journals and covers major ADHD research topics.
Keywords and journals based on ADHD_journals_keywords_pubmed_templates.md.
"""

import json
import sys
import os
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import quote_plus

PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

JOURNALS = [
    "Journal of Attention Disorders",
    "ADHD Attention Deficit and Hyperactivity Disorders",
    "Journal of Child Psychology and Psychiatry",
    "Journal of the American Academy of Child and Adolescent Psychiatry",
    "European Child and Adolescent Psychiatry",
    "Journal of Clinical Child and Adolescent Psychology",
    "Child and Adolescent Psychiatry and Mental Health",
    "Child Psychiatry and Human Development",
    "Journal of Child and Adolescent Psychopharmacology",
    "Research on Child and Adolescent Psychopathology",
    "Development and Psychopathology",
    "Developmental Cognitive Neuroscience",
    "Biological Psychiatry",
    "Biological Psychiatry Cognitive Neuroscience and Neuroimaging",
    "NeuroImage Clinical",
    "Journal of Psychiatric Research",
    "Progress in Neuro-Psychopharmacology and Biological Psychiatry",
    "Cortex",
    "Neuroscience and Biobehavioral Reviews",
    "Psychological Medicine",
]

TOPICS = [
    "ADHD",
    "attention-deficit/hyperactivity disorder",
    "executive function",
    "inhibitory control",
    "working memory",
    "sustained attention",
    "emotion dysregulation",
    "methylphenidate",
    "atomoxetine",
    "guanfacine",
    "lisdexamfetamine",
    "neuroimaging",
    "fMRI",
    "cognitive behavioral therapy",
    "parent training",
    "adult ADHD",
    "comorbidity",
    "autism",
    "sleep",
    "digital therapeutics",
]

HEADERS = {"User-Agent": "ADHDBrainBot/1.0 (research aggregator)"}


def build_query(days: int = 7, max_journals: int = 12) -> str:
    adhd_core = '("Attention Deficit Disorder with Hyperactivity"[Mesh] OR ADHD[tiab] OR "attention-deficit/hyperactivity disorder"[tiab])'
    journal_part = " OR ".join([f'"{j}"[Journal]' for j in JOURNALS[:max_journals]])
    lookback = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y/%m/%d")
    date_part = f'"{lookback}"[Date - Publication] : "3000"[Date - Publication]'
    return f"{adhd_core} AND ({journal_part}) AND {date_part}"


def search_papers(query: str, retmax: int = 50) -> list[str]:
    params = (
        f"?db=pubmed&term={quote_plus(query)}&retmax={retmax}&sort=date&retmode=json"
    )
    url = PUBMED_SEARCH + params
    try:
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        print(f"[ERROR] PubMed search failed: {e}", file=sys.stderr)
        return []


def fetch_details(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    ids = ",".join(pmids)
    params = f"?db=pubmed&id={ids}&retmode=xml"
    url = PUBMED_FETCH + params
    try:
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=60) as resp:
            xml_data = resp.read().decode()
    except Exception as e:
        print(f"[ERROR] PubMed fetch failed: {e}", file=sys.stderr)
        return []

    papers = []
    try:
        root = ET.fromstring(xml_data)
        for article in root.findall(".//PubmedArticle"):
            medline = article.find(".//MedlineCitation")
            art = medline.find(".//Article") if medline else None
            if art is None:
                continue

            title_el = art.find(".//ArticleTitle")
            title = (
                (title_el.text or "").strip()
                if title_el is not None and title_el.text
                else ""
            )

            abstract_parts = []
            for abs_el in art.findall(".//Abstract/AbstractText"):
                label = abs_el.get("Label", "")
                text = "".join(abs_el.itertext()).strip()
                if label and text:
                    abstract_parts.append(f"{label}: {text}")
                elif text:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts)[:2000]

            journal_el = art.find(".//Journal/Title")
            journal = (
                (journal_el.text or "").strip()
                if journal_el is not None and journal_el.text
                else ""
            )

            pub_date = art.find(".//PubDate")
            date_str = ""
            if pub_date is not None:
                year = pub_date.findtext("Year", "")
                month = pub_date.findtext("Month", "")
                day = pub_date.findtext("Day", "")
                parts = [p for p in [year, month, day] if p]
                date_str = " ".join(parts)

            pmid_el = medline.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None else ""
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

            keywords = []
            for kw in medline.findall(".//KeywordList/Keyword"):
                if kw.text:
                    keywords.append(kw.text.strip())

            papers.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "journal": journal,
                    "date": date_str,
                    "abstract": abstract,
                    "url": link,
                    "keywords": keywords,
                }
            )
    except ET.ParseError as e:
        print(f"[ERROR] XML parse failed: {e}", file=sys.stderr)

    return papers


def load_published_pmids(tracking_file: str) -> set[str]:
    if not os.path.exists(tracking_file):
        return set()
    try:
        with open(tracking_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        cutoff = (datetime.now(timezone(timedelta(hours=8))) - timedelta(days=7)).strftime("%Y-%m-%d")
        pmids = set()
        for pmid, date_str in data.get("published", {}).items():
            if date_str >= cutoff:
                pmids.add(pmid)
        print(f"[INFO] Loaded {len(pmids)} published PMIDs (last 7 days) from {tracking_file}", file=sys.stderr)
        return pmids
    except Exception as e:
        print(f"[WARN] Failed to load published PMIDs: {e}", file=sys.stderr)
        return set()


def save_published_pmids(tracking_file: str, new_pmids: list[str], date_str: str):
    data = {"published": {}}
    if os.path.exists(tracking_file):
        try:
            with open(tracking_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass
    cutoff = (datetime.now(timezone(timedelta(hours=8))) - timedelta(days=7)).strftime("%Y-%m-%d")
    data["published"] = {
        k: v for k, v in data.get("published", {}).items() if v >= cutoff
    }
    for pmid in new_pmids:
        data["published"][pmid] = date_str
    os.makedirs(os.path.dirname(tracking_file) or ".", exist_ok=True)
    with open(tracking_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved {len(new_pmids)} new PMIDs to {tracking_file} (total tracked: {len(data['published'])})", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Fetch ADHD papers from PubMed")
    parser.add_argument("--days", type=int, default=7, help="Lookback days")
    parser.add_argument(
        "--max-papers", type=int, default=40, help="Max papers to fetch"
    )
    parser.add_argument("--output", default="-", help="Output file (- for stdout)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--tracking", default="docs/published_pmids.json", help="Published PMIDs tracking file")
    args = parser.parse_args()

    query = build_query(days=args.days)
    print(
        f"[INFO] Searching PubMed for ADHD papers from last {args.days} days...",
        file=sys.stderr,
    )

    pmids = search_papers(query, retmax=args.max_papers)
    print(f"[INFO] Found {len(pmids)} papers from PubMed", file=sys.stderr)

    published = load_published_pmids(args.tracking)
    if published:
        before = len(pmids)
        pmids = [p for p in pmids if p not in published]
        print(f"[INFO] Deduplicated: {before} -> {len(pmids)} (removed {before - len(pmids)} already published)", file=sys.stderr)

    if not pmids:
        print("NO_CONTENT", file=sys.stderr)
        if args.json:
            print(
                json.dumps(
                    {
                        "date": datetime.now(timezone(timedelta(hours=8))).strftime(
                            "%Y-%m-%d"
                        ),
                        "count": 0,
                        "papers": [],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return

    papers = fetch_details(pmids)
    print(f"[INFO] Fetched details for {len(papers)} papers", file=sys.stderr)

    today_str = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")
    save_published_pmids(args.tracking, [p["pmid"] for p in papers if p.get("pmid")], today_str)

    output_data = {
        "date": today_str,
        "count": len(papers),
        "papers": papers,
    }

    out_str = json.dumps(output_data, ensure_ascii=False, indent=2)

    if args.output == "-":
        print(out_str)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_str)
        print(f"[INFO] Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
