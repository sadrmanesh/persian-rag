# صفحه های وب و مارک داون

این امکانات از نسخه 0.3.0 اضافه شده اند و فقط به کتابخانه استاندارد پایتون نیاز دارند.

## فایل HTML و نشانی وب

```python
from persian_rag import extract_main_content, html_to_text, load_html, load_url, load_urls

docs = load_url("https://fa.wikipedia.org/wiki/حافظ")
docs = load_urls(["https://example.com/a", "https://example.com/b"])
docs = load_html("saved/article.html")
```

تابع `load_url` صفحه را دانلود می کند، تغییر مسیرها را دنبال می کند، کدگذاری اعلام شده در سرآیند یا `<meta charset>` را به کار می برد، متن اصلی را جدا می کند و فهرستی با یک سند برمی گرداند که شناسه آن همان نشانی است. فراداده شامل `source`، `url` (پس از تغییر مسیر)، `format` و در صورت وجود `title`، `lang`، `description`، `published`، `author` و `site_name` است. پاسخ های مارک داون و متن ساده همان طور که هستند خوانده می شوند. فقط نشانی های `http` و `https` پذیرفته می شوند و صفحه های بزرگ تر از ۱۰ مگابایت رد می شوند.

با پارامتر `fetcher` می توانید تابع دانلود خودتان را بدهید، مثلا با احراز هویت، پراکسی یا حافظه نهان. این تابع نشانی را می گیرد و `FetchedPage(url, content, content_type, charset)` برمی گرداند.

### جدا کردن متن اصلی

حالت پیش فرض `extract="main"` مانند ابزارهای حالت خواندن کار می کند:

1. اسکریپت، استایل، فرم، منوی ناوبری، سرصفحه، پاصفحه، ستون کناری و عنصرهای پنهان (با `hidden`، `aria-hidden="true"` یا `display:none`) حذف می شوند.
2. اگر `<article>` یا `<main>` متن کافی داشته باشند، همان ها انتخاب می شوند.
3. در غیر این صورت هر بلوک بر پایه طول متن، تعداد ویرگول و نقطه فارسی و لاتین و تراکم پیوندها امتیاز می گیرد و بهترین بلوک همراه با بلوک های هم سطح قوی نگه داشته می شود. نام کلاس هایی مانند `content`، `article`، `post` و `matn` امتیاز را بالا و نام هایی مانند `comment`، `sidebar`، `footer`، `share`، `related` و `ads` امتیاز را پایین می آورند.

اگر بهترین بلوک کمتر از ۲۰۰ نویسه داشته باشد، کل متن قابل دیدن صفحه برگردانده می شود تا صفحه های کوتاه از دست نروند. حالت `extract="all"` همیشه همه بلوک های قابل دیدن را نگه می دارد. عنوان ها، بندها و آیتم های فهرست بندهای جدا می شوند، `<br>` به شکست خط تبدیل می شود و هر ردیف جدول به شکل `cell | cell` درمی آید.

```python
page = extract_main_content(html)  # ExtractedPage(title, text, lang, metadata)
everything = html_to_text(html)
```

تابع `load_documents("./site")` نیز فایل های `.html` و `.htm` را با همین روش می خواند و حالت آن با `html_extract=` تعیین می شود.

## مارک داون

```python
from persian_rag import load_markdown, load_markdown_text, split_markdown_sections

doc = load_markdown("guide.md")
sections = load_markdown("guide.md", split_sections=True)
```

* سرآیند فایل (بلوک `---` با خط های `key: value`) به فراداده اضافه می شود. گیومه دور مقدارها حذف می شود و ساختارهای تودرتو نادیده گرفته می شوند.
* اگر سرآیند عنوان نداشته باشد، نخستین عنوان سطح یک به عنوان `title` به کار می رود.
* نشانه گذاری مارک داون حذف می شود: از پیوند فقط متن آن، از تصویر متن جایگزین آن می ماند، نشانه های تاکید، فهرست و نقل قول حذف می شوند و بلوک های کد کنار گذاشته می شوند.
* با `split_sections=True` هر عنوان یک سند تازه می سازد. شناسه ها مانند `guide.md#نصب` هستند، عنوان های تکراری پسوند `-2` و `-3` می گیرند و `metadata["section"]` مسیر عنوان ها را نگه می دارد (مثلا `"Install > Linux"`). عنوان های درون بلوک کد نادیده گرفته می شوند.

تابع `load_documents(path, split_sections=True)` همین تقسیم را روی همه فایل های مارک داون یک پوشه انجام می دهد.

## خط فرمان

```bash
persian-rag ingest ./docs https://example.com/blog/post --split-sections --html-extract main
```

فرمان `ingest` هر ترکیبی از فایل، پوشه و نشانی وب را می پذیرد.
