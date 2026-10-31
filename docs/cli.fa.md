# خط فرمان

```bash
persian-rag --help
persian-rag <command> --help
```

## فرمان ingest

```bash
persian-rag ingest ./library --index .persian-rag
persian-rag ingest ./library https://example.com/post --index .persian-rag
persian-rag ingest ./library --index .persian-rag --update
```

هر ترکیبی از فایل، پوشه و نشانی `http(s)` را می پذیرد. فایل های متنی، مارک داون، HTML، PDF و Word را می خواند و یک نمایه JSON با تکه های یکدست شده و بردارها می سازد. کلید API هرگز ذخیره نمی شود.

| گزینه | پیش فرض | کاربرد |
| --- | --- | --- |
| `--embedder` | `hashing` | انتخاب بردارساز |
| `--model` | `intfloat/multilingual-e5-small` | مدل بردارساز (پیش فرض برای `sentence-transformers`) |
| `--prefix` | `auto` | پیشوندهای پرسش و سند: `auto` (فقط برای مدل های E5)، `e5` یا `none` ([جزئیات](getting-started.md)) |
| `--query-prefix` و `--passage-prefix` | | پیشوندهای دلخواه به جای `--prefix` |
| `--strategy` | `recursive` | روش تکه کردن |
| `--chunk-size` و `--overlap` | ۸۰۰ و ۱۲۰ | اندازه تکه و هم پوشانی |
| `--rtl-fix` | `auto` | اصلاح ترتیب خواندن PDF |
| `--split-pages` | خاموش | هر صفحه PDF یک سند جدا |
| `--embedding-cache` | خاموش | استفاده دوباره از بردارها با SQLite |
| `--split-sections` | خاموش | هر عنوان مارک داون یک سند جدا |
| `--html-extract` | `main` | `main` فقط متن اصلی و `all` همه بلوک های قابل دیدن |
| `--update` | خاموش | به روزرسانی ایندکس موجود و بردارسازی فقط سندهای تازه یا تغییر کرده ([جزئیات](incremental.md)) |
| `--keep-missing` | خاموش | همراه `--update`، سندهایی که در ورودی نیستند حذف نمی شوند |

## فرمان query

```bash
persian-rag query "حافظ اهل کجا بود؟" --index .persian-rag --k 5 --json
persian-rag query "اطاق پنج دری" --expand rules
persian-rag query "پرسش" --llm ollama --citations
```

با `--citations` پس از پاسخ فهرست شماره دار منبع ها همراه بخش نقل شده چاپ می شود.

## فرمان chat

```bash
persian-rag chat --index .persian-rag --llm ollama --citations --max-turns 6
```

گفت و گوی تعاملی با حافظه. گزینه `--condense` با مقدارهای `auto`، `llm`، `heuristic` و `none` تعیین می کند پرسش های دنباله دار چگونه به پرسش مستقل تبدیل شوند ([جزئیات](citations-chat.md)). با `/reset` حافظه پاک می شود و با `/exit` خارج می شوید.

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

فرمان های query و chat و eval و serve گزینه های `--index`، `--llm` (`none`، `openai` یا `ollama`)، `--llm-model`، `--llm-max-tokens`، `--base-url`، `--ollama-url`، `--prompt-lang`، `--citations`، `--expand`، `--rerank`، `--rerank-model` و `--rerank-candidates` را می پذیرند. صفحه های [اتصال به مدل های زبانی](llm.md) و [بازنویسی پرسش](query-rewriting.md) را ببینید.
