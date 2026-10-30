# بازنویسی پرسش

در فارسی بسیاری از واژه ها چند املای پذیرفته دارند و کاربران گاهی حروف عربی، املای قدیمی یا فاصله گذاری متفاوت به کار می برند. از نسخه 0.3.0 بازنویس پرسش از هر پرسش چند گونه می سازد و نتیجه آنها را با هم ترکیب می کند.

```python
from persian_rag import QueryRewriter, RAGPipeline, RewritingRetriever

rewriter = QueryRewriter()
rewriter.variants("اطاق های هتل طهران")

pipeline = RAGPipeline(retriever, llm=llm, query_rewriter=rewriter)
expanded = RewritingRetriever(retriever, rewriter)
```

## قاعده ها

| قاعده | نمونه |
| --- | --- |
| گروه های املایی | اتاق و اطاق، تهران و طهران، هیئت و هیات و هیأت، مسئله و مساله، رئیس و رییس، بلیت و بلیط، باتری و باطری، پاییز و پائیز، آیین و آئین |
| املای همزه و مد | شکل های أ، ؤ، ئی، آ، ۀ و ة |
| فاصله پیشوند و پسوند | می و نمی و ها و های و تر و ترین به شکل چسبیده، با فاصله یا با نیم فاصله |
| حروف عربی | ي و ك، به طور پیش فرض خاموش (`arabic_variants=True`) |

تنظیمات با `RewriteConfig(max_variants=6, spelling=True, orthography=True, affix_spacing=True, arabic_variants=False, llm_variants=3)` انجام می شود. گروه های خودتان را با `QueryRewriter(extra_groups=[("ایمیل", "رایانامه")])` یا `persian_rag.rewrite.spelling_groups_from_mapping({...})` اضافه کنید.

## بازنویسی با مدل زبانی

```python
rewriter = QueryRewriter(llm=my_llm)  # هر تابع str -> str، مثلا OllamaLLM()
```

از مدل خواسته می شود چند صورت دیگر از پرسش را هر کدام در یک خط بنویسد. پاسخ پاک سازی و پس از گونه های قاعده محور اضافه می شود. اگر فراخوانی مدل خطا بدهد نادیده گرفته می شود و بازیابی ادامه پیدا می کند.

## ترکیب نتیجه ها

`RewritingRetriever` هر گونه را جداگانه جست و جو می کند و رتبه ها را با روش RRF (`rrf_k=60`) ترکیب می کند. وزن پرسش اصلی دو برابر است (`original_weight=2.0`)، پس گسترش پرسش پوشش را بیشتر می کند بدون اینکه تطابق های دقیق پایین بروند. گونه هایی که هر نتیجه را پیدا کرده اند در `metadata["matched_queries"]` ثبت می شوند. اثر آن روی پرسش های با املای دیگر در صفحه [محک ها](benchmarks.md) آمده است.

## خط فرمان

```bash
persian-rag query "اطاق پنج دری" --expand rules
persian-rag query "اطاق پنج دری" --expand llm --llm ollama
```
