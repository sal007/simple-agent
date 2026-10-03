"""Sub-agents: let the model hand a subtask to a fresh helper agent.

This adds one tool, `delegate`. When the model calls it with a task, we start
a brand-new Agent (same model, same tools) with an empty history, run the task
to completion, and give only its final answer back to the main agent as the
tool result.

Why bother? Everything the helper reads (whole files, long listings, its own
tool calls) stays in the helper's history, which is thrown away afterwards.
The main conversation only grows by the short answer. That matters most for
small local models with short context windows: a task that reads ten files
can fit even when the ten files together wouldn't.

The cost: the helper starts from nothing, so the model has to write a task
that contains everything the helper needs, and the helper's model calls use
tokens too (they are added to the main agent's usage).

A helper can't delegate again, so there is only ever one level of sub-agents.
"""

from __future__ import annotations

from . import instructions, planning
from .agent import Agent, AgentEvents
from .providers.base import ToolSpec

TOOL_NAME = "delegate"

TOOL_DESCRIPTION = (
    "Hand a self-contained subtask to a helper agent. The helper has the same tools but starts with an "
    "empty conversation: it can't see this one, so the task must include everything it needs (file names, "
    "what to look for, what to report back). It returns only its final answer. Use it for work that needs a "
    "lot of reading, such as going through many files, so the details stay out of this conversation."
)

SUBAGENT_PROMPT = (
    "You are a helper agent. Another agent gave you one task. Do it using the available tools, then reply "
    "with a concise but complete answer that includes every detail the other agent asked for (exact values, "
    "file names, quotes). You can't ask questions, so if something is unclear, make a reasonable choice and "
    "say what you assumed."
)


def enable(agent: Agent) -> None:
    """Add the delegate tool to this agent's tools.

    The helper gets the parent's tools as they are when it is called (so
    plugins loaded later are included), minus the delegate tool itself.
    """
    agent.tools = agent.tools.copy()  # Don't add it to a registry other agents share.

    def delegate(task: str) -> str:
        report = agent.events.on_subagent
        tools = agent.tools.copy()
        tools.remove(TOOL_NAME)  # No sub-sub-agents.
        steps = 0

        def count_step(step, reply):
            nonlocal steps
            steps = step

        helper = Agent(
            provider=agent.provider,
            tools=tools,
            system_prompt=SUBAGENT_PROMPT,
            max_steps=agent.max_steps,
            stream=False,
            context=agent.context,
            events=AgentEvents(
                on_model_reply=count_step,
                on_tool_call=lambda call: report("tool_call", {"name": call.name, "arguments": call.arguments}),
                on_tool_result=lambda call, result: report("tool_result", {"name": call.name, "result": result}),
            ),
        )
        project = instructions.instructions_of(agent)
        if project:
            instructions.enable(helper, project)  # The project's rules apply to the helper too.
        if planning.planner_of(agent):
            # The parent plans (planning.py), so the helper gets a plan of its
            # own rather than the parent's update_plan tool.
            planning.enable(helper)
        report("start", {"task": task})
        try:
            answer = helper.ask(task)
        finally:
            # Whatever happens, the helper's tokens count toward the session.
            for key, value in helper.usage.items():
                agent.usage[key] = agent.usage.get(key, 0) + value
        report("end", {"answer": answer, "steps": steps, **helper.usage, "messages": len(helper.history)})
        return answer

    schema = {
        "type": "object",
        "properties": {"task": {"type": "string", "description": "The complete task for the helper."}},
        "required": ["task"],
    }
    agent.tools.add(ToolSpec(TOOL_NAME, TOOL_DESCRIPTION, schema), delegate)
