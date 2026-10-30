# پایگاه های برداری

همه پایگاه ها یک رابط مشترک دارند: افزودن با `add`، جست و جو با `search` و شمارش با `len`. امتیاز بیشتر همیشه یعنی شباهت بیشتر. پایگاه را به `build_retriever` یا `HybridRetriever` بدهید.

| پایگاه | افزونه | محل نگهداری | حذف |
| --- | --- | --- | --- |
| `InMemoryVectorStore` | ندارد | حافظه برنامه | خیر |
| `FaissVectorStore` | `faiss` | حافظه برنامه | خیر |
| `QdrantVectorStore` | `qdrant` | سرور، سرویس ابری، پوشه محلی یا حافظه | بله |
| `PgVectorStore` | `pgvector` | پایگاه PostgreSQL | بله |

## پایگاه Qdrant

```python
from persian_rag.vectorstores import QdrantVectorStore

server = QdrantVectorStore("persian_docs", dimension=384, url="http://localhost:6333")
on_disk = QdrantVectorStore("persian_docs", dimension=384, path="./qdrant-data")
in_memory = QdrantVectorStore("persian_docs", dimension=384)
```

شناسه های متنی به UUID ثابت تبدیل می شوند و شناسه اصلی، متن و فراداده در payload نگهداری می شود. مجموعه در نخستین استفاده ساخته می شود. اگر مجموعه از قبل وجود داشته باشد، ابعاد بردار از خود آن خوانده می شود.

## پایگاه PostgreSQL با pgvector

```python
from persian_rag.vectorstores import PgVectorStore

store = PgVectorStore(
    "postgresql://user:pass@localhost:5432/db",
    dimension=384,
    table="persian_chunks",
    distance="cosine",
    create_index=True,
)
```

این کلاس افزونه `vector`، جدول لازم و در صورت درخواست نمایه HNSW را می سازد. برای استفاده از اتصال موجود، پارامتر `connection=` را بدهید. نام جدول بررسی می شود و همه مقدارها به صورت پارامتر به پایگاه فرستاده می شوند.

برای راه اندازی پایگاه محلی:

```bash
docker run -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16
```

## کتابخانه FAISS

```python
from persian_rag import FaissVectorStore

store = FaissVectorStore(dimension=384, metric="cosine")
```
