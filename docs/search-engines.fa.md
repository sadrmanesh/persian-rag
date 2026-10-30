# Elasticsearch و OpenSearch

از نسخه 0.3.0 دو پایگاه `ElasticsearchStore` و `OpenSearchStore` بردارها، متن و فراداده را در یک ایندکس نگه می دارند و هر دو بخش جست و جوی ترکیبی را انجام می دهند: جست و جوی k-NN برای بخش برداری و BM25 با تحلیلگر داخلی `persian` برای بخش واژگانی. این پایگاه ها مستقیما با REST API کار می کنند و به بسته جداگانه ای نیاز ندارند.

```python
from persian_rag import HashingEmbedder, HybridRetriever, build_retriever
from persian_rag.vectorstores import ElasticsearchStore, OpenSearchStore, store_from_url

store = ElasticsearchStore("http://localhost:9200", "persian_docs")
# store = OpenSearchStore("https://search.example.com", "persian_docs", username="admin", password="...")
# store = store_from_url("opensearch+https://search.example.com", "persian_docs")

embedder = HashingEmbedder()
retriever = build_retriever(texts, embedder, document_ids=ids, store=store)

# بعدا در فرایندی دیگر، همه چیز در ایندکس هست:
retriever = HybridRetriever(embedder, store=store, sparse_backend=store)
results = retriever.retrieve("دیوان حافظ", k=5)
```

با `sparse_backend=store` ایندکس BM25 درون برنامه به کار نمی رود، پس فرایند تازه پیش از جست و جو لازم نیست هیچ تکه ای را بارگذاری کند.

## تنظیمات

| پارامتر | پیش فرض | کاربرد |
| --- | --- | --- |
| `url` | `ELASTICSEARCH_URL` یا `OPENSEARCH_URL` و سپس `http://localhost:9200` | نشانی خوشه |
| `index` | `persian_rag` | نام ایندکس با حروف کوچک |
| `dimension` | از نگاشت ایندکس خوانده می شود | فقط برای ساخت ایندکس تازه پیش از نخستین `add` لازم است |
| `api_key` | `ELASTICSEARCH_API_KEY` یا `OPENSEARCH_API_KEY` | به شکل `Authorization: ApiKey ...` فرستاده می شود |
| `username` و `password` | گذرواژه از `..._PASSWORD` | احراز هویت پایه |
| `analyzer` | `persian` | تحلیلگر فیلد `text` |
| `refresh` | `wait_for` | سیاست تازه سازی پس از نوشتن: `true`، `false` یا `wait_for` |
| `num_candidates` | 100 | تعداد نامزدها در هر شارد برای k-NN در Elasticsearch |
| `verify_tls`، `ca_file`، `timeout` | | تنظیمات اتصال |
| `transport` | urllib | جایگزینی تابع HTTP برای آزمون یا کلاینت دیگر |

ایندکس در نخستین نوشتن ساخته می شود: در Elasticsearch با فیلد `dense_vector` (شباهت کسینوسی و HNSW) و در OpenSearch با فیلد `knn_vector` (HNSW کتابخانه Lucene و کسینوس). نوشتن با `_bulk` انجام می شود، `add` شناسه تکراری را نمی پذیرد، `upsert` آن را جایگزین می کند و `delete(ids)` حذف می کند. `drop_index()` کل ایندکس را پاک می کند.

## به روزرسانی تدریجی

این پایگاه ها حذف را پشتیبانی می کنند، پس `HybridRetriever.delete_document`، `delete_chunks` و [ایندکس سازی تدریجی](incremental.md) با آنها کار می کنند.

## سرور محلی

```bash
docker run -p 9200:9200 -e discovery.type=single-node -e xpack.security.enabled=false \
  docker.elastic.co/elasticsearch/elasticsearch:8.15.3
docker run -p 9201:9200 -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true \
  -e DISABLE_INSTALL_DEMO_CONFIG=true opensearchproject/opensearch:2.17.1
```

آزمون های یکپارچگی در CI روی هر دو اجرا می شوند:

```bash
PERSIAN_RAG_ES_URL=http://localhost:9200 PERSIAN_RAG_OPENSEARCH_URL=http://localhost:9201 pytest -m search
```

خط فرمان همچنان از ایندکس JSON استفاده می کند و این پایگاه ها از راه کتابخانه در دسترس هستند.
