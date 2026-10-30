"""Index two short Persian passages and answer with a stub model.

The hashing embedder is only here so the example runs without a model
download. Swap in SentenceTransformerEmbedder or OpenAICompatibleEmbedder
for semantic vectors.
"""

from persian_rag import HashingEmbedder, RAGPipeline, build_retriever


def main() -> None:
    texts = [
        "حافظ در شهر شیراز زاده شد و دیوان او مشهور است.",
        "فوتبال ورزشی گروهی با دو تیم است.",
    ]
    retriever = build_retriever(texts, HashingEmbedder(dimension=128))
    result = RAGPipeline(retriever, _answer).query("حافظ در کدام شهر زاده شد؟")
    print(result.answer)
    if result.contexts:
        print(result.contexts[0].text)


def _answer(prompt: str) -> str:
    if "شیراز" in prompt:
        return "شیراز"
    return "نمی‌دانم"


if __name__ == "__main__":
    main()
