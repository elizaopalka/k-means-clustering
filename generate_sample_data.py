"""
generate_sample_data.py

Generates a synthetic sample dataset that mirrors the *shape* of the
production pipeline (keyword/context matches joined to per-document
exposure-run statistics) without using any real database, company, or
earnings-call data.

Six synthetic topic categories stand in for the kind of thematic
structure K-Means is expected to recover: cost pressure, guidance
uncertainty, macro/rates, supply chain, legal/regulatory, and
restructuring language. Each category has a small bank of template
phrases; sampling from them with noise produces a corpus with genuine
(if simple) cluster structure to validate the pipeline against.

Output:
    data/sample_keyword_matches.csv
    data/sample_exposure_runs.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 11
N_DOCUMENTS = 400
MATCHES_PER_DOC_RANGE = (3, 14)

TOPIC_BANK = {
    "cost_pressure": {
        "keywords": ["input costs", "margin pressure", "cost inflation", "pricing pressure"],
        "contexts": [
            "we are seeing continued input cost inflation across the segment",
            "margin pressure from higher raw material costs persisted this quarter",
            "we've had to absorb some of the cost increases rather than pass them through",
            "pricing actions have partially offset cost pressure but not fully",
        ],
    },
    "guidance_uncertainty": {
        "keywords": ["limited visibility", "cautious outlook", "wait and see", "hard to predict"],
        "contexts": [
            "visibility into the back half of the year remains limited",
            "we're taking a cautious approach to guidance given the current environment",
            "it's a bit of a wait and see situation heading into next quarter",
            "demand patterns have been hard to predict over the last few months",
        ],
    },
    "macro_rates": {
        "keywords": ["rate environment", "macro headwinds", "fx pressure", "economic uncertainty"],
        "contexts": [
            "the rate environment continues to weigh on financing costs",
            "we are navigating macro headwinds across most of our end markets",
            "fx pressure had a modest negative impact on reported revenue",
            "broader economic uncertainty is causing some customers to delay orders",
        ],
    },
    "supply_chain": {
        "keywords": ["supply constraints", "lead times", "logistics costs", "component shortages"],
        "contexts": [
            "supply constraints have eased somewhat compared to last year",
            "lead times on key components remain elevated versus historical levels",
            "logistics costs came down as ocean freight rates normalized",
            "component shortages affected a subset of our product lines this quarter",
        ],
    },
    "legal_regulatory": {
        "keywords": ["regulatory review", "pending litigation", "compliance costs", "investigation"],
        "contexts": [
            "we remain subject to an ongoing regulatory review in that jurisdiction",
            "pending litigation related to the matter disclosed last year continues",
            "compliance costs increased as we implemented the new reporting requirements",
            "we can't comment further given the active investigation",
        ],
    },
    "restructuring": {
        "keywords": ["headcount reduction", "facility consolidation", "restructuring charge", "cost savings program"],
        "contexts": [
            "the headcount reduction announced last quarter is substantially complete",
            "facility consolidation is expected to generate savings starting next year",
            "we recorded a restructuring charge associated with the program",
            "the cost savings program remains on track against our original targets",
        ],
    },
}

MATCH_TYPES = ["direct", "cosine"]


def simulate(seed: int = RNG_SEED):
    rng = np.random.default_rng(seed)
    topics = list(TOPIC_BANK.keys())

    keyword_rows = []
    exposure_rows = []
    row_id = 1

    for doc_id in range(1, N_DOCUMENTS + 1):
        # each document leans toward 1-2 dominant topics, with some noise
        n_dominant = rng.choice([1, 2], p=[0.7, 0.3])
        dominant_topics = rng.choice(topics, size=n_dominant, replace=False)

        n_matches = rng.integers(*MATCHES_PER_DOC_RANGE)
        total_sentences = int(rng.integers(80, 400))

        direct_count = 0
        cosine_count = 0

        for _ in range(n_matches):
            if rng.random() < 0.75 and len(dominant_topics) > 0:
                topic = rng.choice(dominant_topics)
            else:
                topic = rng.choice(topics)

            bank = TOPIC_BANK[topic]
            keyword = rng.choice(bank["keywords"])
            context = rng.choice(bank["contexts"])
            match_type = rng.choice(MATCH_TYPES, p=[0.6, 0.4])
            if match_type == "direct":
                direct_count += 1
            else:
                cosine_count += 1

            keyword_rows.append({
                "id": row_id,
                "exposure_run_id": doc_id,
                "keyword": keyword,
                "match_type": match_type,
                "context": context,
            })
            row_id += 1

        sentences_with_match = min(total_sentences, int(n_matches * rng.uniform(0.8, 1.3)))
        exposure_rows.append({
            "id": doc_id,
            "document_id": doc_id,
            "total_direct_matches": direct_count,
            "total_cosine_matches": cosine_count,
            "total_sentences_in_document": total_sentences,
            "total_sentences_with_match": sentences_with_match,
            "pct_sentences_with_match": round(sentences_with_match / total_sentences, 4),
        })

    return pd.DataFrame(keyword_rows), pd.DataFrame(exposure_rows)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[1] / "data"
    out_dir.mkdir(exist_ok=True)

    kw_df, exp_df = simulate()
    kw_df.to_csv(out_dir / "sample_keyword_matches.csv", index=False)
    exp_df.to_csv(out_dir / "sample_exposure_runs.csv", index=False)

    print(f"Wrote {len(kw_df)} keyword matches across {len(exp_df)} documents -> {out_dir}/")
