# یکدست سازی متن

متن فارسی در عمل ترکیبی از حروف عربی و فارسی، چند نوع فاصله، سه دستگاه رقم و نویسه های تزئینی است. تابع `normalize` اسناد و پرسش ها را با هم قابل مقایسه می کند.

```python
from persian_rag import NormalizeConfig, normalize

normalize("يك كتاب با اعداد ٤٥٦")
# 'یک کتاب با اعداد ۴۵۶'

normalize("کِتاب", NormalizeConfig(remove_diacritics=True, digits="ascii"))
# 'کتاب'
```

اجرای دوباره `normalize` با یک پیکربندی ثابت نتیجه را تغییر نمی دهد. برای اسناد و پرسش ها از یک `NormalizeConfig` استفاده کنید. توابع `build_retriever` و `HybridRetriever` این کار را خودکار انجام می دهند.

| گزینه | پیش فرض | اثر |
| --- | --- | --- |
| `unify_arabic_chars` | `True` | تبدیل ی و ک عربی و چند حرف دیگر به شکل فارسی |
| `compatibility_forms` | `True` | تبدیل شکل های نمایشی حروف به حرف پایه با NFKC |
| `zwnj` | `"normalize"` | یکی کردن نیم فاصله های تکراری، حذف آنها یا تبدیل به فاصله |
| `correct_half_spaces` | `True` | چسباندن پیشوندهایی مانند «می» و پسوندهایی مانند «ها» با نیم فاصله |
| `digits` | `"persian"` | رقم فارسی، انگلیسی یا بدون تغییر |
| `remove_diacritics` | `False` | حذف اعراب |
| `remove_tatweel` | `True` | حذف کشیده |
| `clean_punctuation` | `True` | حذف نشانه های تکراری و مرتب کردن فاصله اطراف آنها |
| `punctuation` | `"persian"` | تبدیل ویرگول، نقطه ویرگول و علامت سؤال به شکل فارسی یا برعکس |

## واژه ها و تکه ها

```python
from persian_rag import ChunkingConfig, chunk_text, split_sentences, tokenize

tokenize("کتاب های خوب")
split_sentences("سلام. خوبی؟ نسخه 1.2 آماده است.")
chunks = chunk_text(text, ChunkingConfig(strategy="recursive", chunk_size=800, overlap=120))
```

واژه یابی نیم فاصله درون واژه را نگه می دارد، واژه های لاتین را کوچک می کند و عددها را جدا می کند. روش های تکه کردن:

- `fixed`: پنجره هایی با طول ثابت که روی فاصله تنظیم می شوند و به اندازه `overlap` هم پوشانی دارند.
- `sentence`: جمله ها تا سقف `chunk_size` کنار هم چیده می شوند.
- `recursive`: تقسیم بر اساس بند، خط، نشانه های فارسی و سپس فاصله.
