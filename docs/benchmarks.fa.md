# محک ها

از نسخه 0.3.0 مخزن یک محک کوچک بازیابی فارسی در پوشه `benchmarks/` دارد:

* `data/corpus.jsonl`: ۳۴ سند کوتاه درباره ادبیات، تاریخ، جغرافیا، فرهنگ و موضوع های روزمره. چند سند عمدا واژه های مشترک با سندهای دیگر دارند تا بازیابی را دشوارتر کنند.
* `data/qa.jsonl`: ۴۲ پرسش همراه شناسه سندهای مرتبط. دوازده پرسش برچسب `spelling` دارند و املای دیگری از واژه های پیکره را به کار می برند (اطاق به جای اتاق، طهران به جای تهران، هیات به جای هیئت، باطری به جای باتری، بلیط به جای بلیت، مساله به جای مسئله، آئین به جای آیین، پائیز به جای پاییز و رییس به جای رئیس).
* `run_benchmark.py`: ایندکس را می سازد و همه پیکربندی ها را می سنجد.

```bash
python benchmarks/run_benchmark.py
python benchmarks/run_benchmark.py --json results.json
python benchmarks/run_benchmark.py --embedder sentence-transformers --model intfloat/multilingual-e5-small
```

پیکره و پرسش ها در Hugging Face Hub هم با نام [sadrmanesh/persian-rag-bench](https://huggingface.co/datasets/sadrmanesh/persian-rag-bench) منتشر شده اند و [نسخه نمایشی زنده](https://huggingface.co/spaces/sadrmanesh/persian-rag-demo) روی همین پیکره جست و جو می کند:

```python
from datasets import load_dataset

corpus = load_dataset("sadrmanesh/persian-rag-bench", "corpus", split="test")
qa = load_dataset("sadrmanesh/persian-rag-bench", "qa", split="test")
```

## نتیجه ها

نسخه 0.3.0، پایتون 3.12، بردارساز درهم سازی (۵۱۲ بعد، n-gram های ۲ تا ۴ نویسه ای)، تکه های ۴۰۰ نویسه ای و `k=10`. معیار Hit@k سهم پرسش هایی است که سند مرتبط در k نتیجه نخست آنها هست. MRR و nDCG بر پایه سند حساب می شوند. تاخیر زمان یک فراخوانی `retrieve` روی یک ماشین لینوکس ۸ هسته ای است (میانه و صدک ۹۵ در پنج دور).

| پیکربندی | Hit@1 | Hit@5 | MRR@10 | nDCG@10 | Hit@1 (املایی) | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| sparse | 0.881 | 0.881 | 0.881 | 0.879 | 0.583 | 0.05 | 0.07 |
| dense | 0.952 | 0.976 | 0.957 | 0.962 | 0.917 | 0.64 | 0.73 |
| hybrid | 0.952 | 0.976 | 0.964 | 0.967 | 0.833 | 0.68 | 0.75 |
| sparse + rewrite | 0.952 | 0.976 | 0.964 | 0.967 | 0.833 | 0.21 | 0.46 |
| dense + rewrite | 0.952 | 0.976 | 0.957 | 0.962 | 0.917 | 1.49 | 2.91 |
| hybrid + rewrite | 0.952 | 0.976 | 0.964 | 0.967 | 0.833 | 1.55 | 3.10 |

## مدل های بردارساز {#embedding-models}

از نسخه 0.4.0 مدل پیش فرض `sentence-transformers` از `paraphrase-multilingual-MiniLM-L12-v2` به `intfloat/multilingual-e5-small` با پیشوندهای `query: ` و `passage: ` تغییر کرده است. هر دو مدل ۱۱۸ میلیون پارامتر و ۳۸۴ بعد دارند. عددها nDCG@10 هستند، روی CPU با sentence-transformers 6.2.0، شباهت کسینوسی بردارهای نرمال شده، پرسش ها و سندهای یکدست شده با `persian_rag.normalize()` و رتبه بندی روی کل مجموعه:

| مدل | MIRACL fa dev | PQuAD validation | persian-rag-bench |
|---|---:|---:|---:|
| paraphrase-multilingual-MiniLM-L12-v2 (پیش فرض قبلی) | 0.4424 | 0.6437 | 0.8637 |
| multilingual-e5-small با پیشوند (پیش فرض جدید) | 0.7461 | 0.9242 | 0.9881 |
| multilingual-e5-base با پیشوند (`e5-base`) | **0.7657** | **0.9247** | **1.0000** |

| مدل | مجموعه | MRR@10 | Recall@10 |
|---|---|---:|---:|
| MiniLM-L12-v2 | MIRACL fa dev | 0.4271 | 0.6356 |
| e5-small | MIRACL fa dev | 0.7138 | 0.9521 |
| MiniLM-L12-v2 | PQuAD validation | 0.5905 | 0.8113 |
| e5-small | PQuAD validation | 0.9042 | 0.9845 |
| e5-base | MIRACL fa dev | 0.7369 | 0.9590 |
| e5-base | PQuAD validation | 0.9043 | 0.9858 |

### e5-small یا e5-base {#e5-small-or-e5-base}

مدل `intfloat/multilingual-e5-base` به عنوان گزینه دقیق تر پشتیبانی می شود (از نسخه بعدی). از `SentenceTransformerEmbedder("e5-base")` یا `--model e5-base` استفاده کنید. نام کامل مدل هم کار می کند و پیشوندهای E5 خودکار اضافه می شوند.

| | e5-small (پیش فرض) | e5-base |
|---|---:|---:|
| پارامترها | 118M | 278M |
| ابعاد (اندازه ایندکس) | 384 | 768 (دو برابر) |
| حجم دانلود مدل | حدود 470 MB | حدود 1.1 GB |
| سرعت بردارسازی روی CPU، بندهای PQuAD | 54 per second | 22 per second |
| MIRACL fa dev، یکدست شده | 0.7461 | **0.7657** |
| MIRACL fa dev، خام | 0.7448 | **0.7653** |
| MIRACL fa dev، پرسش های مخدوش | 0.6645 | **0.7089** |
| PQuAD validation، یکدست شده | 0.9242 | **0.9247** |
| PQuAD validation، پرسش های مخدوش | 0.8639 | **0.8906** |
| persian-rag-bench | 0.9881 | **1.0000** |
| persian-rag-bench، Hit@1 املایی | 0.9167 | **1.0000** |

در پرسش های مخدوش «ی» و «ک» عربی به کار رفته و نیم فاصله ها به فاصله تبدیل شده اند و هیچ طرف یکدست نشده است. این همان متنی است که با صفحه کلید عربی تایپ می شود. `persian_rag.normalize()` بیشتر این تفاوت را از بین می برد، پس برای کاربرد معمولی سطرهای یکدست شده را مقایسه کنید. بهبود e5-base روی پرسش های تمیز PQuAD کم است و روی MIRACL و پرسش های پرخطا بیشتر. وقتی دقت از زمان ایندکس کردن و حافظه مهم تر است e5-base انتخاب خوبی است. e5-small پیش فرض می ماند چون روی CPU بسیار سریع تر است و روی متن تمیز نزدیک به همان نتیجه را می دهد. بردارهای این دو مدل با هم سازگار نیستند: تغییر مدل یعنی ساخت دوباره ایندکس، و باز کردن ایندکس e5-small با e5-base پذیرفته نمی شود.

سطرهای e5-base با همان اسکریپت ارزیابی روی GPU از نوع T4 در Colab اندازه گیری شده اند. e5-small روی CPU و GPU دقیقا همان امتیازها را می دهد و امتیازهای MIRACL و persian-rag-bench مدل e5-base روی CPU هم تکرار شدند (محک با `run_benchmark.py --model e5-base`). سرعت با sentence-transformers 6.2.0 روی ۸ هسته Xeon، ۵۱۲ بند PQuAD با پیشوند، دسته های ۳۲ تایی و حداکثر ۵۱۲ توکن اندازه گیری شد. یک پرسش کوتاه با e5-small حدود 6 میلی ثانیه و با e5-base حدود 13 میلی ثانیه طول می کشد.

* **MIRACL fa dev**: ۶۳۲ پرسش روی مجموعه ای از ۶٬۴۰۵ متن داوری شده برای همین پرسش ها، نه کل مجموعه ۲.۲ میلیونی MIRACL. این کار از نسخه رسمی ساده تر است، پس این عددها با نتیجه های منتشر شده MIRACL قابل مقایسه نیستند و فقط برای مقایسه مدل ها با هم به کار می آیند.
* **PQuAD validation**: ۵٬۹۹۵ پرسش پاسخ دار روی ۱٬۱۰۳ بند یکتای این بخش.
* **persian-rag-bench**: همان ۳۴ سند و ۴۲ پرسش بالا، فقط بازیابی برداری.
* پیشوندها مهم هستند: در persian-rag-bench مدل e5-small بدون پیشوند به nDCG@10 برابر 0.9824 و Hit@1 املایی 0.8333 می رسد و با پیشوند به 0.9881 و 0.9167.

سطرهای persian-rag-bench را با `run_benchmark.py` (سطر `dense`) دوباره بسازید:

```bash
pip install "persian-rag[sentence-transformers]"
python benchmarks/run_benchmark.py --embedder sentence-transformers
python benchmarks/run_benchmark.py --embedder sentence-transformers \
  --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
python benchmarks/run_benchmark.py --embedder sentence-transformers --model e5-base
```

## تفسیر

* BM25 به تنهایی پنج پرسش از دوازده پرسش املایی را از دست می دهد، چون املای دیگر یک واژه برای آن واژه ای دیگر است. [بازنویس پرسش](query-rewriting.md) بیشتر آنها را بازمی گرداند: Hit@1 روش واژگانی روی پرسش های املایی از 0.583 به 0.833 و در کل از 0.881 به 0.952 می رسد.
* بردارساز درهم سازی n-gram های نویسه ای را مقایسه می کند و به طور طبیعی تفاوت یک حرفی را تحمل می کند. به همین دلیل بازیابی برداری حتی بدون بازنویسی روی پرسش های املایی خوب است. مدل های معنایی رفتار دیگری دارند. اسکریپت را با مدل خودتان اجرا کنید.
* بازنویسی تعداد فراخوانی های بازیابی را به اندازه تعداد گونه ها بیشتر می کند و این در ستون های تاخیر دیده می شود، ولی برای ایندکس های حافظه ای در حد چند میلی ثانیه می ماند.
* مجموعه کوچک است و امتیازها عمدا بالا هستند. این محک برای کنترل رفتار ویژه فارسی در برابر پسرفت است، نه جدول رده بندی. برای تصمیم های مهم پرسش هایی از حوزه خودتان به `qa.jsonl` اضافه کنید.

این محک در هر تغییر همراه آزمون ها اجرا می شود تا اسکریپت و داده ها معتبر بمانند.
