"""Evaluate an embedding model: semantic relevance vs keyword overlap.

Runs the whole comparison several times and checks that the ranking is stable.
Writes results.json, which build_pdf.py consumes.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from semantic_search import search
from sentences import CONCEPT, RELEVANT, TRAPS, corpus, ids
from similarity import EMBEDDING_MODEL, cosine_between, cosine_matrix, embed_texts, keyword_similarity

RUNS = 3
OUTPUT = Path(__file__).parent / "results.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("evaluate")


def one_run(run: int) -> Dict:
    """Embed concept + sentences once and compute every similarity of interest."""
    doc_ids = ids()
    texts = [corpus()[i] for i in doc_ids]
    logger.info("run=%d start | model=%s | texts=%d", run, EMBEDDING_MODEL, len(texts) + 1)
    vectors = embed_texts([CONCEPT] + texts)
    semantic = cosine_between(vectors[0], vectors[1:])
    lexical = keyword_similarity(CONCEPT, texts)
    matrix = cosine_matrix(vectors[1:])
    ranking = [doc_ids[i] for i in np.argsort(semantic)[::-1]]
    loo = leave_one_out(doc_ids, matrix)
    logger.info("run=%d done | semantic ranking=%s", run, ranking)
    for query_id, ranked in loo.items():
        logger.info("run=%d leave-one-out query=%s -> %s", run, query_id, ranked)
    return {
        "run": run,
        "semantic": dict(zip(doc_ids, map(float, semantic))),
        "lexical": dict(zip(doc_ids, map(float, lexical))),
        "matrix": matrix.tolist(),
        "dims": int(vectors.shape[1]),
        "ranking": ranking,
        "leave_one_out": loo,
    }


def leave_one_out(doc_ids: List[str], matrix: np.ndarray) -> Dict[str, List[str]]:
    """Use each relevant sentence as the query against all the other sentences.

    A sentence-length query carries far more context than the 3-word concept
    title, so this is the fairer test of semantic vs lexical discrimination.
    """
    result = {}
    for q in RELEVANT:
        qi = doc_ids.index(q)
        others = [i for i in range(len(doc_ids)) if i != qi]
        ordered = sorted(others, key=lambda i: matrix[qi][i], reverse=True)
        result[q] = [doc_ids[i] for i in ordered]
    return result


def check_invariants(runs: List[Dict]) -> Dict[str, bool]:
    """Verify the properties the exercise is about, on every run."""
    checks = {
        "ranking_identical_across_runs": len({tuple(r["ranking"]) for r in runs}) == 1,
        "short_query_all_relevant_above_T1": all(
            min(r["semantic"][k] for k in RELEVANT) > r["semantic"]["T1"] for r in runs),
        "short_query_all_relevant_above_T2": all(
            min(r["semantic"][k] for k in RELEVANT) > r["semantic"]["T2"] for r in runs),
        "leave_one_out_relevant_before_traps": all(
            all(set(ranked[:len(RELEVANT) - 1]) <= set(RELEVANT)
                for ranked in r["leave_one_out"].values()) for r in runs),
        "keyword_baseline_ranks_a_trap_first": all(
            max(r["lexical"], key=r["lexical"].get) in TRAPS for r in runs),
    }
    for name, ok in checks.items():
        logger.info("check %-40s %s", name, "PASS" if ok else "FAIL")
    return checks


def main() -> None:
    started = datetime.now().isoformat(timespec="seconds")
    runs = [one_run(n) for n in range(1, RUNS + 1)]
    checks = check_invariants(runs)

    logger.info("semantic_search demo | query=%r", CONCEPT)
    top = search(CONCEPT, corpus(), top_k=3)
    for doc_id, score in top:
        logger.info("  top result %s score=%.4f", doc_id, score)

    OUTPUT.write_text(json.dumps({
        "model": EMBEDDING_MODEL, "concept": CONCEPT, "started": started,
        "relevant": RELEVANT, "traps": TRAPS, "runs": runs, "checks": checks,
        "search_top3": top,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("results written to %s", OUTPUT.name)


if __name__ == "__main__":
    main()
