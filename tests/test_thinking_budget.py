"""Unit tests for the thinking-token budget (``mtplx.thinking_budget``)."""

from mtplx.thinking_budget import make_thinking_budget, resolve_think_ids

OPEN, CLOSE = 248068, 248069


class _FakeTokenizer:
    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return {"<think>": OPEN, "</think>": CLOSE}.get(token, 0)


def test_resolve_think_ids():
    assert resolve_think_ids(_FakeTokenizer()) == (OPEN, CLOSE)


def test_caps_reasoning_at_budget_and_forces_close():
    # prompt ends inside an open <think>; the model "wants" to think for 30 tokens
    budget = make_thinking_budget([1, 2, OPEN], _FakeTokenizer(), budget=5)
    tokens: list[int] = []
    for nxt in range(1000, 1030):
        budget.observe(tokens)
        tokens.append(budget.forced_token() if budget.should_force() else nxt)
    budget.observe(tokens)
    close_at = tokens.index(CLOSE)
    assert close_at == 5            # reasoning capped at exactly the budget, then </think> forced
    assert all(1000 <= t < 2000 for t in tokens[:close_at])  # those five were thinking tokens
    assert budget.triggered and budget.forced == 1


def test_natural_close_is_not_forced():
    budget = make_thinking_budget([OPEN], _FakeTokenizer(), budget=100)
    budget.observe([10, 11, CLOSE])
    assert not budget.should_force()


def test_reasoning_off_prompt_never_engages():
    # prompt already closed (<think>...</think>) -> generation starts in content mode
    budget = make_thinking_budget([OPEN, 5, CLOSE], _FakeTokenizer(), budget=5)
    tokens: list[int] = []
    for nxt in range(2000, 2030):
        budget.observe(tokens)
        assert not budget.should_force()
        tokens.append(nxt)


def test_reentry_rearms_budget():
    budget = make_thinking_budget([OPEN], _FakeTokenizer(), budget=3)
    budget.observe([50, CLOSE, 60, 61, OPEN])        # close, content, re-open
    assert not budget.should_force()                  # just re-entered, count reset
    budget.observe([50, CLOSE, 60, 61, OPEN, 70, 71, 72])
    assert budget.should_force()                      # 3 thinking tokens after re-open


def test_disabled_when_budget_not_positive():
    assert make_thinking_budget([OPEN], _FakeTokenizer(), budget=0) is None


def test_env_var_enables(monkeypatch):
    monkeypatch.setenv("MTPLX_THINKING_BUDGET", "7")
    budget = make_thinking_budget([OPEN], _FakeTokenizer())
    assert budget is not None and budget.budget == 7


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MTPLX_THINKING_BUDGET", raising=False)
    assert make_thinking_budget([OPEN], _FakeTokenizer()) is None
