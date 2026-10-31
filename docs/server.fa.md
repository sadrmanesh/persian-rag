# رابط HTTP

```bash
pip install "persian-rag[server]"
persian-rag serve --index .persian-rag --host 127.0.0.1 --port 8000
```

یا با داکر و ایمیجی که در هر انتشار منتشر می شود (صفحه [داکر](docker.md) را ببینید):

```bash
docker run -d -p 8000:8000 -v persian-rag-data:/data ghcr.io/sadrmanesh/persian-rag:0.4.0
```

یا از درون کد:

```python
import uvicorn
from persian_rag.server import create_app

app = create_app(pipeline, title="My Persian search", max_k=50)
uvicorn.run(app, host="127.0.0.1", port=8000)
```

خروجی `create_app` یک برنامه معمولی FastAPI است، پس می توانید میان افزار یا مسیرهای دیگری به آن اضافه کنید. پاسخ گویی ناهمگام است و فراخوانی کند مدل، درخواست های دیگر را معطل نمی کند.

## مسیرها

| روش | مسیر | بدنه | خروجی |
| --- | --- | --- | --- |
| `GET` | `/health` | | وضعیت، نسخه، تعداد تکه ها، وجود مدل و رتبه بند و `embedder` (نوع، مدل، بعد و پیشوندها) |
| `POST` | `/retrieve` | `{"query": "...", "k": 5, "mode": "hybrid"}` | پرسش و فهرست نتیجه ها |
| `POST` | `/query` | `{"question": "...", "k": 5}` | پرسش، پاسخ، متن ها، ارجاع ها و در صورت درخواست پرامپت |
| `POST` | `/chat` | `{"message": "...", "session_id": null}` | فیلدهای `/query` به همراه `session_id`، `standalone_question` و `turn` |
| `DELETE` | `/chat/{session_id}` | | `{"deleted": true}` یا `false` برای گفت و گوی ناشناخته |

مسیر `/chat` برای هر `session_id` یک گفت و گو نگه می دارد. صفحه [ارجاع به منبع و گفت و گو](citations-chat.md) را ببینید. پارامترهای `max_sessions=1000` و `max_turns=6` در `create_app` مصرف حافظه را محدود می کنند.

درخواست با پرسش خالی، حالت نامعتبر یا `k` بیشتر از `max_k` خطای 422 می گیرد. مستندات تعاملی در مسیر `/docs` در دسترس است.

```bash
curl -s localhost:8000/retrieve -H 'content-type: application/json' \
  -d '{"query": "حافظ اهل کجا بود؟", "k": 3}'
```

این سرور احراز هویت ندارد. پیش از انتشار عمومی، آن را پشت دروازه امن خودتان قرار دهید.
