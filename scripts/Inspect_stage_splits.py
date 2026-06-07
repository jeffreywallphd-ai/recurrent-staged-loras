import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from datasets import load_dataset

from data.dataset import _extract_reasoning_and_answer, _build_staged_text


def print_block(title: str, text: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(text[:3000])


def main() -> None:
    ds = load_dataset("meta-math/MetaMathQA", split="train[:100]")

    for i, row in enumerate(ds):
        problem = str(row.get("query", "")).strip()
        response = str(row.get("response", "")).strip()

        reasoning, answer = _extract_reasoning_and_answer(response)
        staged_text, stage1_span, stage2_span, stage3_span, answer_span = _build_staged_text(
            problem,
            reasoning,
            answer,
        )

        s1 = staged_text[stage1_span[0]:stage1_span[1]]
        s2 = staged_text[stage2_span[0]:stage2_span[1]]
        s3 = staged_text[stage3_span[0]:stage3_span[1]]

        print("\n\n" + "#" * 100)
        print(f"EXAMPLE {i}")
        print("#" * 100)

        print_block("RAW RESPONSE", response)
        print_block("STAGE 1", s1)
        print_block("STAGE 2", s2)
        print_block("STAGE 3", s3)


if __name__ == "__main__":
    main()