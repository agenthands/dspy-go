#!/usr/bin/env python3
"""
BrowseComp-Plus Benchmark Results Analyzer.

Usage (from project root):
    python3 pkg/benchmark/browsecomp/scripts/analyze_results.py \\
        --input datasets/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl \\
        --runs  runs/gemini_flash_lite

Produces a summary report of benchmark accuracy, tool usage, and example outputs.
"""

import argparse
import json
import os
import glob
import sys
import textwrap


def truncate(text, length=100):
    if not text:
        return ""
    text = text.replace("\n", " ")
    if len(text) > length:
        return text[:length] + "..."
    return text


def load_ground_truth(input_file):
    """Load ground truth answers from the dataset JSONL file."""
    gt = {}
    with open(input_file, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                qid = str(item.get("query_id", ""))
                gt[qid] = {
                    "query": item.get("query", ""),
                    "answer": item.get("answer", "").strip().lower(),
                }
            except json.JSONDecodeError:
                continue
    return gt


def load_results(runs_dir):
    """Load benchmark results from a runs directory."""
    results = []
    for f in sorted(glob.glob(os.path.join(runs_dir, "run_*.json"))):
        try:
            with open(f, "r") as fh:
                results.append(json.load(fh))
        except Exception as e:
            print(f"Error reading {f}: {e}", file=sys.stderr)
    return results


def analyze(ground_truth, results):
    """Analyze results against ground truth and return metrics."""
    total = len(results)
    exact = 0
    partial = 0
    wrong = 0
    no_answer = 0
    search_exact = 0
    search_total = 0
    nosearch_exact = 0
    nosearch_total = 0
    total_searches = 0
    total_getdocs = 0
    examples_correct = []
    examples_wrong = []

    for r in results:
        qid = str(r["query_id"])
        pred = ""
        if r.get("result") and r["result"][0].get("output"):
            pred = r["result"][0]["output"].strip()
        truth = ground_truth.get(qid, {}).get("answer", "")
        tc = r.get("tool_call_counts", {})
        used_search = tc.get("Search", 0) > 0
        total_searches += tc.get("Search", 0)
        total_getdocs += tc.get("GetDocument", 0)

        if used_search:
            search_total += 1
        else:
            nosearch_total += 1

        if not pred or pred == "answer" or r["status"] == "failed":
            no_answer += 1
            continue

        pred_l = pred.lower()
        if truth and truth in pred_l:
            exact += 1
            if used_search:
                search_exact += 1
            else:
                nosearch_exact += 1
            if len(examples_correct) < 3:
                examples_correct.append((qid, truth, pred[:120]))
        elif truth and any(w in pred_l for w in truth.split() if len(w) > 3):
            partial += 1
        else:
            wrong += 1
            if len(examples_wrong) < 3:
                examples_wrong.append((qid, truth, pred[:120]))

    answered = total - no_answer
    return {
        "total": total,
        "exact": exact,
        "partial": partial,
        "wrong": wrong,
        "no_answer": no_answer,
        "answered": answered,
        "search_exact": search_exact,
        "search_total": search_total,
        "nosearch_exact": nosearch_exact,
        "nosearch_total": nosearch_total,
        "total_searches": total_searches,
        "total_getdocs": total_getdocs,
        "examples_correct": examples_correct,
        "examples_wrong": examples_wrong,
    }


def print_report(metrics):
    """Print a formatted report from metrics."""
    m = metrics
    total = m["total"]
    answered = m["answered"]

    print(f"=== PERFORMANCE REPORT ({total} queries processed) ===")
    print()
    print(f"  Exact match:    {m['exact']:>4}/{total} ({m['exact']/total*100:5.1f}%)")
    print(f"  Partial match:  {m['partial']:>4}/{total} ({m['partial']/total*100:5.1f}%)")
    print(f"  Wrong answer:   {m['wrong']:>4}/{total} ({m['wrong']/total*100:5.1f}%)")
    print(f"  No answer:      {m['no_answer']:>4}/{total} ({m['no_answer']/total*100:5.1f}%)")
    print()
    print(
        f"  Effective accuracy (exact):   {m['exact']}/{answered} = {m['exact']/max(answered,1)*100:.1f}%"
    )
    print(
        f"  Effective w/ partial:         {m['exact']+m['partial']}/{answered} = {(m['exact']+m['partial'])/max(answered,1)*100:.1f}%"
    )
    print()
    print("--- TOOL USAGE ---")
    print(
        f"  Queries w/ Search: {m['search_total']}/{total} ({m['search_total']/total*100:.0f}%)"
    )
    print(
        f"  Total Search:      {m['total_searches']}  ({m['total_searches']/total:.1f}/query)"
    )
    print(
        f"  Total GetDocument: {m['total_getdocs']}  ({m['total_getdocs']/total:.1f}/query)"
    )
    if m["search_total"] > 0:
        print(
            f"  Accuracy WITH tools:    {m['search_exact']}/{m['search_total']} = {m['search_exact']/m['search_total']*100:.1f}%"
        )
    if m["nosearch_total"] > 0:
        print(
            f"  Accuracy WITHOUT tools: {m['nosearch_exact']}/{m['nosearch_total']} = {m['nosearch_exact']/m['nosearch_total']*100:.1f}%"
        )
    print()
    print("--- CORRECT EXAMPLES ---")
    for qid, truth, pred in m["examples_correct"]:
        print(f"  Q{qid}: expected={truth!r}")
        print(f"         got={truncate(pred)!r}")
    print()
    print("--- WRONG EXAMPLES ---")
    for qid, truth, pred in m["examples_wrong"]:
        print(f"  Q{qid}: expected={truth!r}")
        print(f"         got={truncate(pred)!r}")


def main():
    parser = argparse.ArgumentParser(description="Analyze BrowseComp benchmark results")
    parser.add_argument(
        "--input",
        default="datasets/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl",
        help="Path to dataset JSONL",
    )
    parser.add_argument(
        "--runs", default="runs/go_rlm", help="Path to benchmark runs directory"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: dataset file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.runs):
        print(f"Error: runs directory not found: {args.runs}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading ground truth from {args.input}...", file=sys.stderr)
    gt = load_ground_truth(args.input)
    print(f"  {len(gt)} queries loaded.", file=sys.stderr)

    print(f"Loading results from {args.runs}...", file=sys.stderr)
    results = load_results(args.runs)
    print(f"  {len(results)} result files loaded.", file=sys.stderr)

    metrics = analyze(gt, results)
    print_report(metrics)


if __name__ == "__main__":
    main()
