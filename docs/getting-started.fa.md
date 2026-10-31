# شروع کار

## نصب

```bash
pip install persian-rag
```

برای هر قابلیت، افزونه مربوط به آن را اضافه کنید:

| افزونه | قابلیت |
| --- | --- |
| `sentence-transformers` | کلاس `SentenceTransformerEmbedder` |
| `rerank` | کلاس `CrossEncoderReranker` |
| `faiss` | کلاس `FaissVectorStore` |
| `qdrant` | کلاس `QdrantVectorStore` |
| `pgvector` | کلاس `PgVectorStore` |
| `pdf` | خواندن PDF با pypdf |
| `docx` | خواندن Word با python-docx |
| `server` | برنامه FastAPI و فرمان `persian-rag serve` |
| `all` | همه موارد بالا |
| `docs` | ابزار MkDocs برای ساختن همین مستندات |

```bash
pip install "persian-rag[pdf,docx,rerank,server]"
```

پایتون ۳٫۱۰ یا جدیدتر لازم است.

## نخستین سامانه

```python
from persian_rag import HashingEmbedder, RAGPipeline, build_retriever

texts = [
    "حافظ در شهر شیراز زاده شد و دیوان او مشهور است.",
    "فوتبال ورزشی گروهی با دو تیم است.",
]
retriever = build_retriever(texts, HashingEmbedder())


def llm(prompt: str) -> str:
    return "شیراز"  # call your model here


answer = RAGPipeline(retriever, llm).query("حافظ در کدام شهر زاده شد؟", k=2)
print(answer.answer)
```

کلاس `HashingEmbedder` قطعی است و بدون اینترنت کار می کند، پس برای آزمون ها عالی است، اما معنا را درک نمی کند. برای داده واقعی یکی از این دو را به کار ببرید:

```python
from persian_rag import OpenAICompatibleEmbedder, SentenceTransformerEmbedder

local = SentenceTransformerEmbedder()  # intfloat/multilingual-e5-small
remote = OpenAICompatibleEmbedder(model="text-embedding-3-small")
```

نسخه دوم کلید را از متغیر محیطی `OPENAI_API_KEY` می خواند.

## مدل های بردارساز و پیشوندها

از نسخه 0.4.0 مدل پیش فرض `SentenceTransformerEmbedder` مدل [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) است (۱۱۸ میلیون پارامتر و ۳۸۴ بعد). این مدل در مجموعه های بازیابی فارسی با همان اندازه بسیار بهتر از مدل پیش فرض قبلی، `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`، کار می کند. عددها در [محک ها](benchmarks.md#embedding-models) آمده اند.

برای دقت بیشتر از [`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base) استفاده کنید (۲۷۸ میلیون پارامتر و ۷۶۸ بعد). نام های کوتاه `e5-small` و `e5-base` هر جا که نام مدل پذیرفته می شود کار می کنند:

```python
SentenceTransformerEmbedder("e5-base")  # same as "intfloat/multilingual-e5-base"
```

```bash
persian-rag ingest ./library --index .persian-rag --embedder sentence-transformers --model e5-base
```

مدل e5-base در MIRACL fa dev به nDCG@10 برابر 0.7657 می رسد، در برابر 0.7461 برای e5-small، و روی پرسش های پرخطا بسیار بهتر است. اما حدود ۲.۴ برابر بزرگ تر است، ایندکس آن دو برابر جا می گیرد و روی CPU حدود ۲.۴ برابر کندتر بردار می سازد. مقایسه کامل در [محک ها](benchmarks.md#e5-small-or-e5-base) آمده است. ایندکس نام کامل مدل را نگه می دارد، پس `query`، `chat`، `eval` و `serve` برای ایندکسی که با e5-base ساخته شده دوباره همین مدل را بار می کنند.

مدل های E5 پیش از پرسش `query: ` و پیش از سند `passage: ` می خواهند. بردارساز این پیشوندها را خودش اضافه می کند:

| `prefix` | رفتار |
| --- | --- |
| `"auto"` (پیش فرض) | اگر مدل پرامپت های `query` و `document` غیر خالی داشته باشد همان ها به کار می روند. در غیر این صورت برای مدل های E5 و مدل های `persian-rag-embed` پیشوندهای E5 اضافه می شود و برای مدل های دیگر هیچ پیشوندی اضافه نمی شود. |
| `"e5"` | همیشه `query: ` و `passage: ` اضافه می شود. |
| `"none"` | هیچ پیشوندی اضافه نمی شود. |

با `query_prefix` و `passage_prefix` می توانید نتیجه را تغییر دهید:

```python
SentenceTransformerEmbedder("intfloat/multilingual-e5-base")  # E5 prefixes
SentenceTransformerEmbedder("BAAI/bge-m3")  # no prefix
SentenceTransformerEmbedder("my-org/my-e5-copy", prefix="e5")  # force E5 prefixes
SentenceTransformerEmbedder("my-org/model", query_prefix="Query: ", passage_prefix="")
```

اگر پیشوندها با پرامپت های خود مدل یکی باشند، بردارساز `encode_query` و `encode_document` را صدا می زند. پیشوندهای نهایی در ایندکس ذخیره می شوند، پس هر ایندکس همیشه به همان شکلی که ساخته شده جست و جو می شود.

## ارتقا از نسخه 0.3.x

* کد و ایندکس هایی که نام مدل را مشخص کرده اند بدون تغییر کار می کنند. هر ایندکس مدل سازنده اش را نگه می دارد و دستورهای `query`، `chat`، `eval` و `serve` همان مدل را بار می کنند.
* ایندکس های ساخته شده پیش از 0.4.0 پیشوند ذخیره شده ندارند، پس بدون پیشوند جست و جو می شوند، درست مثل زمان ساخت. برای ایندکس های E5 قدیمی یک هشدار هم نمایش داده می شود: با `persian-rag ingest` دوباره بسازید تا پیشوندها و بازیابی بهتر را بگیرید.
* `SentenceTransformerEmbedder()` بدون نام مدل حالا e5-small را بار می کند. بردارهای دو مدل با هم سازگار نیستند، هرچند هر دو ۳۸۴ بعد دارند. ایندکس هایی را که با مدل پیش فرض قبلی ساخته اید دوباره بسازید، یا مدل قبلی را صریح مشخص کنید:

```bash
persian-rag ingest ./library --index .persian-rag --embedder sentence-transformers \
  --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

* اگر ایندکسی با بردارساز مدل دیگری بار شود، حالا به جای نتیجه نادرست خطای `IndexFormatError` می گیرید. تفاوت پیشوندها فقط هشدار می دهد.

## از روی فایل ها

```python
from persian_rag import load_documents

documents = load_documents("./library")
retriever = build_retriever(
    [doc.text for doc in documents],
    embedder,
    document_ids=[doc.id for doc in documents],
    metadatas=[doc.metadata for doc in documents],
)
```

یا از خط فرمان:

```bash
persian-rag ingest ./library --index .persian-rag
persian-rag query "حافظ اهل کجا بود؟" --index .persian-rag
```

## فراخوانی مدل زبانی

تابع `make_openai_llm` برای هر سرویس گفت و گوی سازگار با OpenAI یک تابع قابل فراخوانی می سازد:

```python
from persian_rag import make_openai_llm

llm = make_openai_llm(model="gpt-4o-mini", api_key="...")
```

قالب پیش فرض پرامپت فارسی است (`DEFAULT_PROMPT_FA`). برای انگلیسی `prompt_template=DEFAULT_PROMPT_EN` را بدهید. قالب دلخواه باید `{context}` و `{question}` را داشته باشد.
