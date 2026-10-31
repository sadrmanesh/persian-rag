# داکر

از نسخه 0.3.1 هر انتشار یک ایمیج داکر از رابط HTTP را در GitHub Container Registry منتشر می کند:

```text
ghcr.io/sadrmanesh/persian-rag
```

| برچسب | به چه اشاره می کند |
| --- | --- |
| `0.4.0` | دقیقا همین نسخه |
| `0.4` | تازه ترین نسخه 0.4.x |
| `latest` | تازه ترین نسخه |

ایمیج ها برای `linux/amd64` و `linux/arm64` ساخته می شوند. هر ایمیج بسته `persian-rag[server,pdf,docx]` را روی پایتون 3.12 دارد، با کاربر غیر ریشه (شناسه 10001) اجرا می شود و روی درگاه 8000 گوش می دهد. پوشه کاری و حجم داده `/data` است و فرمان پیش فرض، ایندکس موجود در `/data/index` را سرو می کند.

## ساخت ایندکس و اجرای سرور

ابتدا از پوشه اسناد در یک حجم نام دار ایندکس بسازید و سپس آن را سرو کنید:

```bash
docker volume create persian-rag-data

docker run --rm \
  -v "$PWD/library:/library:ro" \
  -v persian-rag-data:/data \
  ghcr.io/sadrmanesh/persian-rag:0.4.0 ingest /library --index /data/index

docker run -d --name persian-rag -p 8000:8000 \
  -v persian-rag-data:/data \
  ghcr.io/sadrmanesh/persian-rag:0.4.0
```

```bash
curl -s localhost:8000/health
curl -s localhost:8000/retrieve -H 'content-type: application/json' \
  -d '{"query": "حافظ اهل کجا بود؟", "k": 3}'
```

نقطه ورود ایمیج فرمان `persian-rag` است، پس همه زیرفرمان ها کار می کنند: `ingest`، `query`، `chat`، `eval` یا `serve` با گزینه های دلخواه شما. ایندکسی را که روی سیستم خودتان ساخته اید هم می توانید مستقیم با `-v "$PWD/.persian-rag:/data/index:ro"` وصل کنید.

## پاسخ با مدل زبانی

اتصال های OpenAI و Ollama در ایمیج هستند. کلید را با متغیر محیطی و گزینه ها را به `serve` بدهید:

```bash
docker run -d -p 8000:8000 -v persian-rag-data:/data \
  -e OPENAI_API_KEY \
  ghcr.io/sadrmanesh/persian-rag:0.4.0 \
  serve --index /data/index --host 0.0.0.0 --port 8000 --llm openai --citations
```

برای Ollama روی همان سیستم، `-e OLLAMA_HOST=http://host.docker.internal:11434` را اضافه کنید (در لینوکس `--add-host=host.docker.internal:host-gateway` هم لازم است) و از `--llm ollama` استفاده کنید.

## بررسی سلامت

ایمیج یک `HEALTHCHECK` دارد که هر ۳۰ ثانیه مسیر `GET /health` را صدا می زند. وقتی ایندکس بارگذاری شد، `docker ps` وضعیت ظرف را `healthy` نشان می دهد. اگر درگاه را در فرمان تغییر دادید، `-e PERSIAN_RAG_PORT=<port>` را هم تنظیم کنید تا بررسی سلامت از همان درگاه استفاده کند.

## ساخت ایمیج توسط خودتان

```bash
docker build -t persian-rag .
docker build -t persian-rag --build-arg EXTRAS=server,pdf,docx,sentence-transformers .
```

آرگومان `EXTRAS` بسته های اضافه داخل ایمیج را تعیین می کند. ایمیج منتشر شده `sentence-transformers` را ندارد چون چند گیگابایت حجم اضافه می کند. برای سرو کردن ایندکسی که با `--embedder sentence-transformers` ساخته شده، ایمیج خودتان را با این بسته بسازید. ایمیج متغیر `HF_HOME=/data/.cache/huggingface` را تنظیم می کند، پس مدل (مثلا مدل پیش فرض `intfloat/multilingual-e5-small`) یک بار در حجم `/data` دانلود می شود و پس از راه اندازی دوباره هم می ماند. مسیر `GET /health` مدلی را که ایندکس با آن ساخته شده نشان می دهد.

این سرور احراز هویت ندارد. پیش از انتشار عمومی، آن را پشت دروازه امن خودتان قرار دهید.
