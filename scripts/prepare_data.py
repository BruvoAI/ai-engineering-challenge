"""Pre-warm the local dataset cache from SQuAD v1.1 (CC BY-SA 4.0), without opening
the notebook.

Optional — the notebook's setup cell does this automatically on first run. Use this
script if you'd rather warm the cache ahead of time (e.g. on a flaky connection before
a workshop), or after deliberately changing the split parameters below.

    uv run python scripts/prepare_data.py

Output (all gitignored — never pushed to the repo):
  data/corpus.json    - full "articles" (paragraphs concatenated per title)
  data/dev_qa.json    - questions + gold answers, for self-checking
  eval/test_qa.json   - questions + gold answers, read by the locked harness

The split is deterministic (fixed seed), so every participant's local files come out
byte-identical without ever being committed. If you change --n-titles/--dev-n/--test-n/
--seed from the defaults, everyone needs to delete their local cache and rebuild it so
their corpus/dev_qa/test_qa stay in sync with each other.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.dataset import build_splits


def build(n_titles: int, dev_n: int, test_n: int, seed: int) -> None:
    corpus, dev_qa, test_qa = build_splits(n_titles, dev_n, test_n, seed)

    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "eval").mkdir(exist_ok=True)

    (ROOT / "data" / "corpus.json").write_text(json.dumps(corpus, indent=2))
    (ROOT / "data" / "dev_qa.json").write_text(json.dumps(dev_qa, indent=2))
    (ROOT / "eval" / "test_qa.json").write_text(json.dumps(test_qa, indent=2))

    print(f"corpus: {len(corpus)} articles, {sum(len(c['text']) for c in corpus):,} chars total -> data/corpus.json")
    print(f"dev_qa: {len(dev_qa)} questions -> data/dev_qa.json")
    print(f"test_qa: {len(test_qa)} questions -> eval/test_qa.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-titles", type=int, default=20)
    parser.add_argument("--dev-n", type=int, default=40)
    parser.add_argument("--test-n", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.n_titles, args.dev_n, args.test_n, args.seed)
