# ایندکس سازی تدریجی

از نسخه 0.3.0 اجرای دوباره ایندکس سازی روی پوشه ای که تغییر کرده فقط بخش های تغییر کرده را به روز می کند.

```python
from persian_rag import IncrementalIndexer, load_documents

indexer = IncrementalIndexer(retriever, chunking=chunking)
report = indexer.sync(load_documents("./library"))
print(report.summary())
```

* برای هر سند یک چکیده SHA-256 با نام `content_hash` از متن یکدست شده، فراداده، تنظیمات یکدست سازی و تنظیمات تکه بندی ساخته می شود. تغییر هر کدام سند را به روز شده به حساب می آورد.
* سندهای به روز شده دوباره تکه بندی می شوند. تکه هایی که متنشان تغییر نکرده بردار قبلی خود را نگه می دارند، پس فقط متن تازه بردار می گیرد.
* `sync(documents)` سندهایی را که دیگر وجود ندارند حذف می کند. برای فقط افزودن و جایگزینی از `delete_missing=False` یا `upsert(documents)` استفاده کنید.
* `delete(["old.md"])` سندها را با شناسه حذف می کند.
* `indexer.manifest` از نوع `IndexManifest` شناسه هر سند را به چکیده و شناسه تکه هایش نگاشت می کند. این داده با `to_dict()` و `from_dict()` به JSON ساده تبدیل می شود و `save_index(..., manifest=...)` آن را در `index.json` ذخیره می کند.

حذف به پایگاهی با متد `delete` نیاز دارد. پایگاه های حافظه، Qdrant، pgvector، Elasticsearch و OpenSearch این متد را دارند و FAISS ندارد.

## خط فرمان

```bash
persian-rag ingest ./library --index .persian-rag
persian-rag ingest ./library --index .persian-rag --update
persian-rag ingest ./new-notes --index .persian-rag --update --keep-missing
```

با `--update` ایندکس موجود خوانده می شود، چکیده ها مقایسه می شوند و خلاصه تغییرات چاپ می شود. با `--keep-missing` سندهایی که در مسیرهای داده شده نیستند حذف نمی شوند. تنظیمات مدل بردارساز و تکه بندی از ایندکس خوانده می شوند، پس کافی است مسیرها را بدهید. ایندکس های نسخه 0.2.0 فهرست سندها را ندارند. نخستین اجرای `--update` همه سندها را به روز شده در نظر می گیرد (بردار تکه های بدون تغییر دوباره به کار می رود) و این فهرست را می سازد.
