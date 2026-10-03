"""Token and cost counting: what each turn used, and what the session has used so far.

Every model reply reports how many tokens went in (the whole conversation
resent, plus the system prompt and tool list) and how many came out. The
agent adds these up in `agent.usage`, along with how many model calls it
made. One turn can be several calls: one per tool round, plus any sub-agent
calls and context compaction, which all count toward the same totals.

After each answer the chat shows a line like:

    (this turn: 3 model calls, 4,210 in + 180 out tokens, $0.0204 · session: 9,876 in + 512 out tokens, $0.0497)

Cost is tokens times the model's price per million tokens. Prices for Claude
models are built in (DEFAULT_PRICES below); add or override any model under
[prices] in config.toml. Local models (LM Studio, Ollama) have no price, so
the line shows tokens only.

Turn it off with show_usage = false in config.toml, --no-usage, or /usage off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# US dollars per million tokens: (input, output). Anthropic's list prices as of
# 2026-09-25; check https://claude.com/pricing if they look out of date.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def prices_from_config(section: dict) -> dict[str, tuple[float, float]]:
    """The built-in prices plus the [prices] table: model = { input = 4.0, output = 20.0 }."""
    prices = dict(DEFAULT_PRICES)
    for model, price in section.items():
        if not isinstance(price, dict) or set(price) != {"input", "output"}:
            raise ValueError(f'[prices] "{model}" needs exactly input and output (dollars per million tokens)')
        prices[model] = (float(price["input"]), float(price["output"]))
    return prices


def cost(usage: dict[str, int], model: str, prices: dict[str, tuple[float, float]]) -> float | None:
    """What these tokens cost in dollars, or None if the model has no price (e.g. a local model)."""
    if model not in prices:
        return None
    price_in, price_out = prices[model]
    return (usage.get("input_tokens", 0) * price_in + usage.get("output_tokens", 0) * price_out) / 1_000_000


def difference(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    """What was added between two snapshots of agent.usage, e.g. during one turn."""
    return {key: value - before.get(key, 0) for key, value in after.items()}


def describe(usage: dict[str, int], model: str, prices: dict[str, tuple[float, float]], calls: bool = False) -> str:
    """'3 model calls, 4,210 in + 180 out tokens, $0.0204' (the parts that apply)."""
    parts = []
    if calls:
        n = usage.get("model_calls", 0)
        parts.append(f"{n} model call{'' if n == 1 else 's'}")
    parts.append(f"{usage.get('input_tokens', 0):,} in + {usage.get('output_tokens', 0):,} out tokens")
    dollars = cost(usage, model, prices)
    if dollars is not None:
        parts.append(format_dollars(dollars))
    return ", ".join(parts)


def format_dollars(dollars: float) -> str:
    # Turns are often fractions of a cent, so show enough digits to see them.
    return f"${dollars:.4f}" if dollars < 1 else f"${dollars:.2f}"


def turn_line(turn: dict[str, int], total: dict[str, int], model: str, prices: dict[str, tuple[float, float]]) -> str:
    """The line printed after each answer."""
    return f"(this turn: {describe(turn, model, prices, calls=True)} · session: {describe(total, model, prices)})"


@dataclass
class UsageMeter:
    """What the chat needs to show the line after each answer, and whether to show it."""

    model: str
    prices: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_PRICES))
    show: bool = True
    _before: dict[str, int] = field(default_factory=dict)

    def start_turn(self, usage: dict[str, int]) -> None:
        self._before = dict(usage)

    def turn_line(self, usage: dict[str, int]) -> str:
        return turn_line(difference(usage, self._before), usage, self.model, self.prices)

    def session_line(self, usage: dict[str, int]) -> str:
        return describe(usage, self.model, self.prices, calls=True)
