"""Multi-turn conversations on top of :class:`~persian_rag.pipeline.RAGPipeline`.

A follow-up such as ``و پایتختش کجاست؟`` cannot be searched on its own.
:class:`ChatSession` keeps the recent turns in a :class:`ConversationMemory`,
rewrites each follow-up into a standalone question, retrieves with that
question, and shows the model a short transcript of the conversation.

The rewrite uses the LLM when one is available (``condense="llm"`` or
``"auto"``) and otherwise a heuristic that adds the previous question to
short or referential follow-ups, which is enough for keyword and hybrid
search. Memory is kept in process; persist :meth:`ConversationMemory.to_dict`
if sessions must survive restarts.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from persian_rag.pipeline import RAGAnswer, RAGPipeline, _is_async_callable, _require_text
from persian_rag.retrieval import RetrievalMode

CondenseMode = Literal["auto", "llm", "heuristic", "none"]

CONDENSE_PROMPT_FA = (
    "با توجه به گفتگوی زیر، آخرین پرسش کاربر را به یک پرسش مستقل و کامل به فارسی "
    "بازنویسی کن که بدون گفتگو هم قابل فهم باشد. فقط پرسش بازنویسی شده را بنویس.\n\n"
    "گفتگو:\n{history}\n\n"
    "آخرین پرسش: {question}\n\n"
    "پرسش مستقل:"
)

CONDENSE_PROMPT_EN = (
    "Given the conversation below, rewrite the user's last question as a "
    "standalone question that can be understood without the conversation. "
    "Reply with the rewritten question only.\n\n"
    "Conversation:\n{history}\n\n"
    "Last question: {question}\n\n"
    "Standalone question:"
)

_REFERENCE_WORDS = frozenset(
    (
        "آن",
        "این",
        "او",
        "اون",
        "اونجا",
        "آنجا",
        "اینجا",
        "آنها",
        "ایشان",
        "همین",
        "همان",
        "دیگر",
        "بیشتر",
        "it",
        "its",
        "they",
        "them",
        "that",
        "this",
        "there",
        "he",
        "she",
        "his",
        "her",
        "more",
    )
)
_PRONOUN_SUFFIX_RE = re.compile(r"[\u0600-\u06ff]{2,}(?:ش|شان|اش)$")
_WORD_RE = re.compile(r"[\w\u0600-\u06ff]+")


@dataclass(frozen=True, slots=True)
class Turn:
    """One exchange: the user's message and the assistant's answer."""

    question: str
    answer: str
    standalone_question: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "question": self.question,
            "answer": self.answer,
            "standalone_question": self.standalone_question,
        }


class ConversationMemory:
    """The last ``max_turns`` turns of a conversation."""

    def __init__(self, max_turns: int = 6, *, max_answer_chars: int = 600) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if max_answer_chars < 1:
            raise ValueError("max_answer_chars must be >= 1")
        self.max_turns = max_turns
        self.max_answer_chars = max_answer_chars
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    def __len__(self) -> int:
        return len(self._turns)

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    def add(self, question: str, answer: str, standalone_question: str = "") -> None:
        self._turns.append(Turn(question, answer, standalone_question))

    def clear(self) -> None:
        self._turns.clear()

    def format(self, *, user_label: str = "کاربر", assistant_label: str = "دستیار") -> str:
        """Render the turns as a plain transcript for a prompt."""

        lines: list[str] = []
        for turn in self._turns:
            answer = " ".join(turn.answer.split())
            if len(answer) > self.max_answer_chars:
                answer = answer[: self.max_answer_chars - 3].rstrip() + "..."
            lines.append(f"{user_label}: {turn.question}")
            if answer:
                lines.append(f"{assistant_label}: {answer}")
        return "\n".join(lines)

    def messages(self) -> list[dict[str, str]]:
        """Turns as chat messages (``role``/``content``)."""

        out: list[dict[str, str]] = []
        for turn in self._turns:
            out.append({"role": "user", "content": turn.question})
            if turn.answer:
                out.append({"role": "assistant", "content": turn.answer})
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"max_turns": self.max_turns, "turns": [turn.to_dict() for turn in self._turns]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ConversationMemory:
        memory = cls(int(data.get("max_turns", 6)))
        turns = data.get("turns", [])
        if isinstance(turns, Iterable):
            for raw in turns:
                if isinstance(raw, Mapping):
                    memory.add(
                        str(raw.get("question", "")),
                        str(raw.get("answer", "")),
                        str(raw.get("standalone_question", "")),
                    )
        return memory


def needs_context(message: str) -> bool:
    """Guess whether ``message`` depends on earlier turns.

    True for very short messages, messages that open with a conjunction
    such as ``و`` or ``اما``, and messages with a referring word or a
    possessive suffix such as ``پایتختش``.
    """

    words = _WORD_RE.findall(message.lower())
    if not words:
        return False
    if len(words) <= 3:
        return True
    if words[0] in {"و", "اما", "ولی", "پس", "and", "but", "so", "also"}:
        return True
    if any(word in _REFERENCE_WORDS for word in words):
        return True
    return any(_PRONOUN_SUFFIX_RE.fullmatch(word) for word in words if len(word) > 4)


def heuristic_standalone(message: str, memory: ConversationMemory) -> str:
    """Prefix a dependent follow-up with the previous standalone question."""

    if not len(memory) or not needs_context(message):
        return message
    last = memory.turns[-1]
    previous = last.standalone_question or last.question
    return f"{previous} {message}"


class ChatSession:
    """A conversation with a :class:`RAGPipeline`.

    ``condense`` picks how follow-ups become standalone questions:

    * ``"auto"``: the LLM when the pipeline (or ``condense_llm``) has a
      synchronous one, otherwise the heuristic;
    * ``"llm"``: always the LLM (raises without one);
    * ``"heuristic"``: :func:`heuristic_standalone`;
    * ``"none"``: search with the message as typed.
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        *,
        memory: ConversationMemory | None = None,
        condense: CondenseMode = "auto",
        condense_llm: Callable[[str], str] | None = None,
        condense_prompt: str = CONDENSE_PROMPT_FA,
        history_prefix: str = "گفتگوی پیشین:\n{history}\n\nپرسش فعلی: {question}",
    ) -> None:
        if condense not in ("auto", "llm", "heuristic", "none"):
            raise ValueError(f"Unknown condense mode: {condense!r}")
        for template in (condense_prompt, history_prefix):
            if "{history}" not in template or "{question}" not in template:
                raise ValueError("templates must contain {history} and {question}")
        self.pipeline = pipeline
        self.memory = memory if memory is not None else ConversationMemory()
        self.condense: CondenseMode = condense
        self.condense_llm = condense_llm
        self.condense_prompt = condense_prompt
        self.history_prefix = history_prefix
        if condense == "llm" and self._condenser() is None:
            raise ValueError("condense='llm' needs condense_llm or a synchronous pipeline LLM")

    def _condenser(self) -> Callable[[str], Any] | None:
        if self.condense_llm is not None:
            return self.condense_llm
        llm = self.pipeline.llm
        if llm is None:
            return None
        if self.condense == "auto" and _is_async_callable(llm):
            return None
        return llm

    def condense_prompt_for(self, message: str) -> str:
        return self.condense_prompt.replace("{history}", self.memory.format()).replace(
            "{question}", message
        )

    def standalone_question(self, message: str) -> str:
        """Rewrite ``message`` so it can be searched without the history."""

        if not len(self.memory) or self.condense == "none":
            return message
        if self.condense == "heuristic":
            return heuristic_standalone(message, self.memory)
        condenser = self._condenser()
        if condenser is None:
            return heuristic_standalone(message, self.memory)
        result = condenser(self.condense_prompt_for(message))
        if inspect.isawaitable(result):
            if hasattr(result, "close"):
                result.close()
            return heuristic_standalone(message, self.memory)
        return _clean_rewrite(_require_text(result)) or message

    def prompt_question(self, message: str) -> str:
        """The question text placed in the answer prompt, with the transcript."""

        if not len(self.memory):
            return message
        return self.history_prefix.replace("{history}", self.memory.format()).replace(
            "{question}", message
        )

    def ask(self, message: str, *, k: int = 5, mode: RetrievalMode = "hybrid") -> ChatAnswer:
        """Answer ``message`` and remember the turn."""

        message = message.strip()
        if not message:
            raise ValueError("message must not be empty")
        standalone = self.standalone_question(message)
        contexts = self.pipeline.retrieve(standalone, k=k, mode=mode)
        prompt = self.pipeline.build_prompt(self.prompt_question(message), contexts)
        answer = self.pipeline.make_answer(
            message, self.pipeline.generate(prompt), prompt, contexts
        )
        self.memory.add(message, answer.answer, standalone)
        return ChatAnswer(answer=answer, standalone_question=standalone, turn=len(self.memory))

    async def aask(self, message: str, *, k: int = 5, mode: RetrievalMode = "hybrid") -> ChatAnswer:
        """Async variant of :meth:`ask`; an async pipeline LLM is awaited."""

        message = message.strip()
        if not message:
            raise ValueError("message must not be empty")
        standalone = await asyncio.to_thread(self.standalone_question, message)
        contexts = await self.pipeline.aretrieve(standalone, k=k, mode=mode)
        prompt = self.pipeline.build_prompt(self.prompt_question(message), contexts)
        llm = self.pipeline.llm
        if llm is None:
            text = ""
        elif _is_async_callable(llm):
            text = _require_text(await llm(prompt))  # type: ignore[misc]
        else:
            generated = await asyncio.to_thread(llm, prompt)
            if inspect.isawaitable(generated):
                generated = await generated
            text = _require_text(generated)
        answer = self.pipeline.make_answer(message, text, prompt, contexts)
        self.memory.add(message, answer.answer, standalone)
        return ChatAnswer(answer=answer, standalone_question=standalone, turn=len(self.memory))

    def reset(self) -> None:
        self.memory.clear()


@dataclass(frozen=True, slots=True)
class ChatAnswer:
    """The pipeline answer for one turn and the question used for search."""

    answer: RAGAnswer
    standalone_question: str
    turn: int

    def to_dict(self, *, include_prompt: bool = False) -> dict[str, Any]:
        payload = self.answer.to_dict(include_prompt=include_prompt)
        payload["standalone_question"] = self.standalone_question
        payload["turn"] = self.turn
        return payload


def _clean_rewrite(text: str) -> str:
    line = next((item.strip() for item in text.splitlines() if item.strip()), "")
    for prefix in ("پرسش مستقل:", "Standalone question:"):
        if line.startswith(prefix):
            line = line[len(prefix) :].strip()
    return line.strip("\"'«»")
