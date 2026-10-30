# Citations and chat

New in 0.3.0.

## Citation-grounded answers

```python
from persian_rag import RAGPipeline, format_sources

pipeline = RAGPipeline(retriever, llm=llm, citations=True)
answer = pipeline.query("حافظ اهل کجا بود؟")

print(answer.answer)  # "حافظ اهل شیراز بود [1]."
for citation in answer.citations:
    print(citation.marker, citation.document_id, citation.quote, citation.start, citation.end)
print(answer.uncited)  # sentences without a marker
```

With `citations=True` the default prompt numbers each context (`[1]`, `[2]`, ...) and asks the model to put a marker after every claim. The answer is then parsed:

* Markers like `[1]`, `[1, 3]`, `[۲]`, and `[٢،٣]` are recognised.
* For every cited claim, `Citation` holds the context it points to, the best supporting sentence in that context (`quote`), and its character span. `start` / `end` are offsets in the normalized document text; `chunk_start` / `chunk_end` are offsets in the chunk; `answer_start` / `answer_end` locate the claim in the answer.
* `answer.uncited` lists content sentences without a marker, so you can flag weakly grounded answers. `extract_citations` also reports markers that point to no context in `invalid_markers`.

`extract_citations(answer_text, contexts)` and `format_sources(report)` work on their own with any model output.

## Conversation memory

```python
from persian_rag import ChatSession, ConversationMemory

chat = ChatSession(pipeline, memory=ConversationMemory(max_turns=6))
chat.ask("حافظ اهل کجا بود؟")
reply = chat.ask("آرامگاهش کجاست؟")
print(reply.standalone_question)  # the follow-up rewritten with context
```

Follow-up questions are rewritten into standalone questions before retrieval:

| `condense` | Behaviour |
| --- | --- |
| `auto` (default) | Use the condense LLM (or the pipeline LLM when it is synchronous); fall back to `heuristic` without one |
| `llm` | Always ask the LLM |
| `heuristic` | When the message looks like a follow-up (very short, opens with و or اما, uses a referring word or a possessive suffix such as پایتختش), prepend the previous question |
| `none` | Never rewrite |

Recent turns are also added to the answer prompt. `ConversationMemory` keeps the last `max_turns` turns, trims long answers, and can be stored with `to_dict()` / `from_dict()`. `aask` is the async variant, and `reset()` clears the session.

## HTTP

`POST /chat` with `{"message": "...", "session_id": null}` starts a session and returns its `session_id`; send it back to continue. `DELETE /chat/{session_id}` ends a session. `/query` and `/chat` include `citations` and `uncited` when the pipeline has `citations=True`. Sessions live in server memory (at most `max_sessions`, oldest dropped first).

## Command line

```bash
persian-rag query "حافظ اهل کجا بود؟" --llm ollama --citations
persian-rag chat --llm openai --citations   # /reset clears the memory, /exit quits
```
