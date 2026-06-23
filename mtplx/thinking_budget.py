"""Thinking-token budget enforcement for ``<think>...</think>`` reasoning models.

While a Qwen-style model generates its reasoning phase, ``</think>`` is just one
probabilistic token in the vocabulary — under hard prompts, repetition, quantization,
or long context the model can fail to emit it and loop *inside* ``<think>`` until it
hits ``max_tokens``. Post-hoc truncation cannot fix this: the model has to actually
transition to content mode. This module caps the reasoning phase at decode time —
once ``budget`` generated thinking tokens accumulate without a natural ``</think>``,
the next *committed* token is overridden to ``</think>``, forcing the transition.

We override the committed token (the ``primary`` in the MTP loop / the sampled token
in the AR loop), never the speculative draft, so the KV cache advances on ``</think>``
consistently and the lossless MTP draft/verify path is left untouched.

Enabled via ``MTPLX_THINKING_BUDGET`` (>0 to enable; 0/unset = disabled, zero overhead).
Behaviour follows oMLX's ``ThinkingBudgetProcessor`` and vLLM / mlx-node budget forcing.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence


def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _last_index(seq: Sequence[int], value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    for i in range(len(seq) - 1, -1, -1):
        if int(seq[i]) == value:
            return i
    return None


def resolve_think_ids(tokenizer) -> tuple[Optional[int], Optional[int]]:
    """Best-effort ``(<think>, </think>)`` single-token ids for this tokenizer."""
    open_id: Optional[int] = None
    close_id: Optional[int] = None
    unk = getattr(tokenizer, "unk_token_id", None)
    conv = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(conv):
        for tok, is_open in (("<think>", True), ("</think>", False)):
            try:
                tid = conv(tok)
            except Exception:
                tid = None
            if isinstance(tid, int) and tid >= 0 and tid != unk:
                if is_open:
                    open_id = tid
                else:
                    close_id = tid
    enc = getattr(tokenizer, "encode", None)
    if callable(enc):
        def _single(tok: str) -> Optional[int]:
            for kwargs in ({"add_special_tokens": False}, {}):
                try:
                    ids = enc(tok, **kwargs)
                except TypeError:
                    continue
                except Exception:
                    return None
                if isinstance(ids, (list, tuple)) and len(ids) == 1:
                    return int(ids[0])
            return None
        if open_id is None:
            open_id = _single("<think>")
        if close_id is None:
            close_id = _single("</think>")
    return open_id, close_id


class ThinkingBudget:
    """Stateful per-generation enforcer.

    Usage in a decode loop (``tokens`` is the committed-so-far generated list)::

        budget.observe(tokens)          # at the top of each step
        if budget.should_force():       # before sampling the next committed token
            primary = budget.forced_token()
        else:
            primary, _ = _sample_from_logits(...)
    """

    def __init__(
        self,
        *,
        budget: int,
        open_id: Optional[int],
        close_id: int,
        prompt_ids: Sequence[int],
        already_emitted: int = 0,
    ):
        self.budget = int(budget)
        self._open = open_id
        self._close = int(close_id)
        self._count = 0
        self._observed = int(already_emitted)
        last_open = _last_index(prompt_ids, open_id)
        last_close = _last_index(prompt_ids, close_id)
        # Thinking is "open" at the end of the prompt iff <think> appears after </think>
        # (Qwen reasoning-on ends the prompt with "<think>\n"; reasoning-off ends it with
        # "<think>\n\n</think>\n\n", i.e. already closed → we never engage).
        self._in_thinking = last_open is not None and (last_close is None or last_open > last_close)
        self._done = not self._in_thinking
        self.forced = 0
        self.triggered = False

    @property
    def thinking_tokens(self) -> int:
        return self._count

    def observe(self, tokens: Sequence[int]) -> None:
        """Fold newly-committed tokens into the thinking-state counter."""
        n = len(tokens)
        if n <= self._observed:
            return
        for i in range(self._observed, n):
            t = int(tokens[i])
            if self._done:
                # re-entry: a fresh <think> after we'd closed
                if self._open is not None and t == self._open:
                    self._in_thinking, self._done, self._count = True, False, 0
                continue
            if t == self._close:
                self._in_thinking, self._done = False, True
            elif self._open is not None and t == self._open:
                self._in_thinking, self._count = True, 0
            elif self._in_thinking:
                self._count += 1
        self._observed = n

    def should_force(self) -> bool:
        return (
            not self._done
            and self._in_thinking
            and self.budget > 0
            and self._count >= self.budget
        )

    def forced_token(self) -> int:
        self.forced += 1
        self.triggered = True
        return self._close


def make_thinking_budget(
    prompt_ids: Sequence[int],
    tokenizer,
    *,
    already_emitted: int = 0,
    budget: Optional[int] = None,
) -> Optional[ThinkingBudget]:
    """Build a :class:`ThinkingBudget` from ``MTPLX_THINKING_BUDGET`` (or an explicit
    ``budget``), or return ``None`` when disabled or the ``</think>`` id is unresolved."""
    b = int(budget) if budget is not None else _env_int("MTPLX_THINKING_BUDGET", 0)
    if b <= 0:
        return None
    open_id, close_id = resolve_think_ids(tokenizer)
    if close_id is None:
        return None
    return ThinkingBudget(
        budget=b,
        open_id=open_id,
        close_id=close_id,
        prompt_ids=prompt_ids,
        already_emitted=already_emitted,
    )
