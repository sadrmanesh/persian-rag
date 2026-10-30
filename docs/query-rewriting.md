# Query rewriting

New in 0.3.0. Persian has several accepted spellings for many words, and users type Arabic letters, old orthography, and different word spacing. The query rewriter turns one query into a few variants and fuses their results.

```python
from persian_rag import QueryRewriter, RAGPipeline, RewritingRetriever

rewriter = QueryRewriter()
rewriter.variants("اطاق های هتل طهران")
# ['اطاق های هتل طهران', 'اطاق های هتل تهران', 'اتاق های هتل طهران', ...]

pipeline = RAGPipeline(retriever, llm=llm, query_rewriter=rewriter)
# or wrap a retriever directly
expanded = RewritingRetriever(retriever, rewriter)
```

## Rules

| Rule | Examples |
| --- | --- |
| Spelling groups | اتاق / اطاق, تهران / طهران, هیئت / هیات / هیأت, مسئله / مساله, رئیس / رییس, بلیت / بلیط, باتری / باطری, پاییز / پائیز, آیین / آئین |
| Orthography | hamza forms (أ, ؤ, ئی), alef madda, ۀ and ة |
| Affix spacing | می / نمی prefixes and ها / های / تر / ترین suffixes written attached, with a space, or with a half space |
| Arabic letters | ي and ك forms, off by default (`arabic_variants=True`) |

Tune it with `RewriteConfig(max_variants=6, spelling=True, orthography=True, affix_spacing=True, arabic_variants=False, llm_variants=3)`. Add your own groups with `QueryRewriter(extra_groups=[("ایمیل", "رایانامه")])` or `persian_rag.rewrite.spelling_groups_from_mapping({...})`.

## LLM rewriting

```python
rewriter = QueryRewriter(llm=my_llm)  # any callable str -> str, for example OllamaLLM()
```

The model is asked for alternative phrasings, one per line. Its replies are cleaned and added after the rule-based variants; a failing model call is ignored so retrieval still works.

## Fusion

`RewritingRetriever` runs every variant through the wrapped retriever and fuses the rankings with reciprocal rank fusion (`rrf_k=60`). The original query counts double (`original_weight=2.0`), so expansion adds recall without pushing exact matches down. Each result records the variants that found it in `metadata["matched_queries"]`. See the [benchmarks](benchmarks.md) for the effect on misspelled queries.

## Command line

```bash
persian-rag query "اطاق پنج دری" --expand rules
persian-rag query "اطاق پنج دری" --expand llm --llm ollama
```
