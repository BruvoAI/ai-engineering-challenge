"""Build the corpus + QA splits from SQuAD v1.1 (CC BY-SA 4.0).

Called by the notebook's setup cell (and optionally by `scripts/prepare_data.py`)
to build `corpus`, `dev_qa`, and `test_qa` (the held-out set the locked harness
scores against). The split is deterministic given the same `seed`, so every
participant's local files come out byte-identical without any of it ever being
committed to the repo.
"""

import random


def build_splits(
    n_titles: int = 20, dev_n: int = 40, test_n: int = 80, seed: int = 42
) -> tuple[list[dict], list[dict], list[dict]]:
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad", split="validation")

    # Group paragraphs (deduped, order preserved) and QA pairs by article title.
    titles_order: list[str] = []
    paragraphs_by_title: dict[str, list[str]] = {}
    qa_by_title: dict[str, list[dict]] = {}

    for row in ds:
        title = row["title"]
        if title not in paragraphs_by_title:
            paragraphs_by_title[title] = []
            qa_by_title[title] = []
            titles_order.append(title)
        if row["context"] not in paragraphs_by_title[title]:
            paragraphs_by_title[title].append(row["context"])
        answers = sorted(set(row["answers"]["text"]))
        if not answers:
            continue
        qa_by_title[title].append(
            {
                "id": row["id"],
                "question": row["question"],
                "answers": answers,
                "doc_id": title,
            }
        )

    rng = random.Random(seed)
    chosen_titles = titles_order[:]
    rng.shuffle(chosen_titles)
    chosen_titles = chosen_titles[:n_titles]

    corpus = [
        {
            "doc_id": title,
            "title": title,
            "text": "\n\n".join(paragraphs_by_title[title]),
        }
        for title in chosen_titles
    ]

    all_qa = [qa for title in chosen_titles for qa in qa_by_title[title]]
    rng.shuffle(all_qa)

    if dev_n + test_n > len(all_qa):
        raise ValueError(
            f"Only {len(all_qa)} QA pairs available for {n_titles} titles; "
            f"reduce dev_n/test_n or increase n_titles."
        )

    dev_qa = all_qa[:dev_n]
    test_qa = all_qa[dev_n : dev_n + test_n]
    return corpus, dev_qa, test_qa
