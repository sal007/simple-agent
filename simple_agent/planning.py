"""Planning: a to-do list the model keeps for itself while it works.

This adds one tool, `update_plan`. The model calls it with its whole plan,
a short list of steps, each "pending", "in_progress" or "done", and calls
it again whenever a step starts or finishes.

The model could just write a plan in its reply, and strong models often do.
Keeping it in a tool has three advantages:

  1. The plan is state, not chat. It lives here, outside the history, and
     is added to the system prompt at every model call. So context
     management (context.py) can clear or summarize old messages without
     the model losing track of its plan.
  2. You can see progress: the CLI prints the list as items tick off.
  3. You can measure it: the eval runner records each task's final plan and
     can check that one was made and finished.

It matters most for small local models, which tend to lose the thread of a
long task. Whether it actually helps a given model is a question for the
eval runner (compare with --no-planning).
"""

from __future__ import annotations

from .agent import Agent
from .providers.base import ToolSpec
from .trace import render_plan as render

TOOL_NAME = "update_plan"
STATUSES = ("pending", "in_progress", "done")

TOOL_DESCRIPTION = (
    "Write or update your plan for the current task: the full list of steps, each with a status of "
    "pending, in_progress or done. Use it for tasks that need three or more steps; skip it for simple "
    "questions. Make a plan before you start, mark a step in_progress when you begin it and done as soon "
    "as it's finished, and change the plan if you learn something new. Send the whole list every time. "
    "Your current plan is always shown to you at the end of the system prompt."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "description": "The whole plan, in order.",
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "string", "description": "What to do, in a few words."},
                    "status": {"type": "string", "enum": list(STATUSES)},
                },
                "required": ["step", "status"],
            },
        }
    },
    "required": ["steps"],
}


class Planner:
    """Holds the current plan for one agent. Registered as an Agent extension."""

    def __init__(self, agent: Agent):
        self.agent = agent
        self.steps: list[dict] = []

    def update_plan(self, steps: list) -> str:
        """The tool itself: replace the plan and show it back to the model."""
        cleaned = []
        for i, item in enumerate(steps, 1):
            if not isinstance(item, dict) or not str(item.get("step", "")).strip():
                return f"Error: item {i} needs a 'step' text. Nothing was changed."
            status = item.get("status", "pending")
            if status not in STATUSES:
                return f"Error: item {i} has status {status!r}; use one of {', '.join(STATUSES)}. Nothing was changed."
            cleaned.append({"step": str(item["step"]).strip(), "status": status})
        self.steps = cleaned
        self.agent.events.on_plan(self.steps)
        return "Plan updated:\n" + render(self.steps)

    # --- the Extension interface (see agent.py) --------------------------------

    def system_note(self) -> str:
        if not self.steps:
            return ""
        return "Your current plan (keep it up to date with update_plan):\n" + render(self.steps)

    def reset(self) -> None:
        self.steps = []

    def done(self) -> bool:
        return bool(self.steps) and all(s["status"] == "done" for s in self.steps)


def enable(agent: Agent) -> Planner:
    """Give the agent the update_plan tool and show its plan at every step."""
    planner = Planner(agent)
    agent.tools = agent.tools.copy()  # Don't add it to a registry other agents share.
    agent.tools.add(ToolSpec(TOOL_NAME, TOOL_DESCRIPTION, SCHEMA), planner.update_plan)
    agent.extensions.append(planner)
    return planner


def planner_of(agent: Agent) -> Planner | None:
    return next((e for e in agent.extensions if isinstance(e, Planner)), None)
