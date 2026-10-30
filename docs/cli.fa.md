# خط فرمان

```bash
persian-rag --help
persian-rag <command> --help
```

## فرمان ingest

```bash
persian-rag ingest ./library --index .persian-rag
```

فایل های متنی، مارک داون، PDF و Word را می خواند و یک نمایه JSON با تکه های یکدست شده و بردارها می سازد. کلید API هرگز ذخیره نمی شود.

| گزینه | پیش فرض | کاربرد |
| --- | --- | --- |
| `--embedder` | `hashing` | انتخاب بردارساز |
| `--strategy` | `recursive` | روش تکه کردن |
| `--chunk-size` و `--overlap` | ۸۰۰ و ۱۲۰ | اندازه تکه و هم پوشانی |
| `--rtl-fix` | `auto` | اصلاح ترتیب خواندن PDF |
| `--split-pages` | خاموش | هر صفحه PDF یک سند جدا |
| `--embedding-cache` | خاموش | استفاده دوباره از بردارها با SQLite |

## فرمان query

```bash
persian-rag query "حافظ اهل کجا بود؟" --index .persian-rag --k 5 --json
```

## فرمان eval

```bash
persian-rag eval eval.jsonl --index .persian-rag --k 1 3 5 10
```

با `--answers` همراه با `--llm openai` وفاداری پاسخ ها هم سنجیده می شود.

## فرمان serve

```bash
persian-rag serve --index .persian-rag --port 8000
```

## گزینه های مشترک

فرمان های query و eval و serve گزینه های `--index`، `--llm`، `--llm-model`، `--base-url`، `--prompt-lang`، `--rerank`، `--rerank-model` و `--rerank-candidates` را می پذیرند.
