# Evaluation

## Dataset

A JSON Lines file, one question per line:

```json
{"id": "q1", "question": "حافظ در کدام شهر زاده شد؟", "relevant_ids": ["hafez.txt"], "reference_answer": "شیراز"}
```

`relevant_ids` holds document ids (the default) or chunk ids. `relevant` and `answer` are accepted as aliases.

## Retrieval metrics

```python
from persian_rag import evaluate_retrieval, load_eval_examples

examples = load_eval_examples("eval.jsonl")
report = evaluate_retrieval(retriever, examples, k_values=(1, 3, 5, 10), mode="hybrid")
print(report.format_table())
report.to_dict(include_queries=True)
```

| Metric | Meaning |
| --- | --- |
| `hit_rate@k` | Share of questions with at least one relevant result in the top `k` |
| `mrr@k` | Mean reciprocal rank of the first relevant result |
| `recall@k` | Share of relevant ids found in the top `k` |
| `ndcg@k` | Normalized discounted cumulative gain with binary relevance |

With `match="document"` several chunks from one relevant document count once. Use `match="chunk"` when labels are chunk ids. Passing a `RAGPipeline` evaluates retrieval after reranking.

## Answer faithfulness

```python
from persian_rag import LLMFaithfulnessJudge, TokenOverlapFaithfulness, evaluate_answers

report = evaluate_answers(pipeline, examples, scorer=TokenOverlapFaithfulness(), k=5)
report.mean_faithfulness, report.mean_reference_f1
```

A faithfulness scorer is any object with `score(question, answer, contexts) -> FaithfulnessResult`:

- `TokenOverlapFaithfulness` measures the share of the answer's content words found in the contexts, after Persian normalization and stopword removal. It is deterministic and free, which suits regression tracking.
- `LLMFaithfulnessJudge(llm)` asks a model for a JSON score between 0 and 1 and a short reason. The reply parser tolerates extra prose.

When an example has `reference_answer`, token F1 against it is reported too. Pipelines without an LLM produce empty answers, which are not scored.

## CLI

```bash
persian-rag eval eval.jsonl --index .persian-rag --k 1 3 5 10
persian-rag eval eval.jsonl --index .persian-rag --rerank cross-encoder --json
OPENAI_API_KEY=... persian-rag eval eval.jsonl --llm openai --answers --judge llm
```
