# ارزیابی

## داده ارزیابی

یک فایل JSON Lines که هر خط آن یک پرسش است:

```json
{"id": "q1", "question": "حافظ در کدام شهر زاده شد؟", "relevant_ids": ["hafez.txt"], "reference_answer": "شیراز"}
```

فیلد `relevant_ids` شناسه سندهای مرتبط را نگه می دارد. اگر برچسب ها در سطح تکه باشند، از `match="chunk"` استفاده کنید.

## سنجه های بازیابی

```python
from persian_rag import evaluate_retrieval, load_eval_examples

examples = load_eval_examples("eval.jsonl")
report = evaluate_retrieval(retriever, examples, k_values=(1, 3, 5, 10))
print(report.format_table())
```

| سنجه | معنا |
| --- | --- |
| `hit_rate@k` | سهم پرسش هایی که دست کم یک نتیجه مرتبط در `k` نتیجه نخست دارند |
| `mrr@k` | میانگین معکوس رتبه نخستین نتیجه مرتبط |
| `recall@k` | سهم شناسه های مرتبطی که پیدا شده اند |
| `ndcg@k` | سود تجمعی کاهشی نرمال شده |

اگر به جای بازیاب یک `RAGPipeline` بدهید، بازیابی پس از رتبه بندی دوباره سنجیده می شود.

## وفاداری پاسخ

```python
from persian_rag import LLMFaithfulnessJudge, TokenOverlapFaithfulness, evaluate_answers

report = evaluate_answers(pipeline, examples, scorer=TokenOverlapFaithfulness(), k=5)
```

هر ارزیاب وفاداری متدی به شکل `score(question, answer, contexts)` دارد:

- کلاس `TokenOverlapFaithfulness` سهم واژه های محتوایی پاسخ را که در متن های بازیابی شده آمده اند می سنجد. یکدست سازی فارسی و حذف واژه های پرتکرار پیش از مقایسه انجام می شود. این روش قطعی و رایگان است و برای پیگیری تغییرات مناسب است.
- کلاس `LLMFaithfulnessJudge` از یک مدل زبانی امتیازی بین صفر و یک همراه با دلیل کوتاه می خواهد.

اگر پرسش پاسخ مرجع داشته باشد، امتیاز F1 در سطح واژه هم گزارش می شود.

## خط فرمان

```bash
persian-rag eval eval.jsonl --index .persian-rag --k 1 3 5 10
persian-rag eval eval.jsonl --index .persian-rag --rerank cross-encoder --json
```
