from typing import Any

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_recall, faithfulness


def run_ragas_eval(samples: list[dict[str, Any]]) -> dict[str, float]:
    """
    samples format:
    [
      {
        "question": "...",
        "answer": "...",
        "contexts": ["..."],
        "ground_truth": "..."
      }
    ]
    """
    if not samples:
        return {}
    df = pd.DataFrame(samples)
    dataset = Dataset.from_pandas(df)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
    )
    return {k: float(v) for k, v in result.items()}

