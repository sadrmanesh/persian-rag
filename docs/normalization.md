# Normalization

Persian text found in the wild mixes Arabic and Persian code points, several kinds of spaces, three digit systems, and decorative characters. `normalize` makes documents and queries comparable.

```python
from persian_rag import NormalizeConfig, normalize

normalize("يك كتاب با اعداد ٤٥٦")
# 'یک کتاب با اعداد ۴۵۶'

normalize("کِتاب", NormalizeConfig(remove_diacritics=True, digits="ascii"))
# 'کتاب'
```

`normalize` is idempotent for a fixed config. Use the same `NormalizeConfig` for documents and queries; `build_retriever` and `HybridRetriever` do this automatically.

| Option | Default | Effect |
| --- | --- | --- |
| `unify_arabic_chars` | `True` | `ي`/`ى` to `ی`, `ك` to `ک`, `أ`/`إ`/`ٱ` to `ا`, `ؤ` to `و`, `ة` to `ه`. |
| `compatibility_forms` | `True` | NFKC: presentation forms fold into base letters. |
| `zwnj` | `"normalize"` | Collapse repeated half-spaces, `"remove"` them, or turn them into `"space"`. |
| `correct_half_spaces` | `True` | Join prefixes such as `می` and suffixes such as `ها` with a half-space. |
| `digits` | `"persian"` | `"persian"`, `"ascii"`, or `"keep"`. |
| `remove_diacritics` | `False` | Strip Arabic vowel marks. |
| `remove_tatweel` | `True` | Remove kashida. |
| `clean_punctuation` | `True` | Collapse repeats and tidy spacing around punctuation. |
| `punctuation` | `"persian"` | Map `,;?%` to Persian forms, the reverse with `"ascii"`, or `"keep"`. |

## Tokens and chunks

```python
from persian_rag import ChunkingConfig, chunk_text, split_sentences, tokenize

tokenize("کتاب های خوب")
split_sentences("سلام. خوبی؟ نسخه 1.2 آماده است.")
chunks = chunk_text(text, ChunkingConfig(strategy="recursive", chunk_size=800, overlap=120))
```

Tokenization keeps half-spaces inside words, lowercases Latin tokens, and separates numbers. Chunking strategies:

- `fixed`: character windows that snap to a space and overlap by `overlap` characters.
- `sentence`: sentences packed up to `chunk_size`, repeating `sentence_overlap` sentences.
- `recursive`: split on paragraphs, lines, Persian punctuation, then spaces.
