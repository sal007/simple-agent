# simple-agent

A small, readable command-line AI agent, built for learning how agents work and
as a base for your own experiments. It talks to a **local model** (LM Studio,
Ollama, or anything OpenAI-compatible) or to **Claude** in the cloud, and it can
call tools.

The whole thing is about 700 lines of Python (lots of it comments) with no framework, so you can read
all of it in one sitting.

```
you> what's 17% of 2,340, and what time is it?
  tool> calculator({"expression": "2340 * 0.17"})
  tool> get_current_time({})
agent> 17% of 2,340 is 397.8, and it's 14:05 on 28 Sep 2026.
```

## Quick start

Requires Python 3.11 or newer.

```bash
git clone https://github.com/sal007/simple-agent
cd simple-agent
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

### With LM Studio (local, free)

1. In LM Studio, download a model that supports tool use (for example
   Qwen 2.5 7B Instruct or Llama 3.1 8B Instruct) and load it.
2. Open the **Developer** tab and click **Start Server**. It listens on
   `http://localhost:1234/v1` by default.
3. Run:

```bash
simple-agent --model qwen2.5-7b-instruct
```

Use the model identifier LM Studio shows for the loaded model. The default
`--base-url` is already LM Studio's.

Other OpenAI-compatible servers work the same way. Just change the URL:

```bash
simple-agent --base-url http://localhost:11434/v1 --model llama3.1          # Ollama
OPENAI_API_KEY=sk-... simple-agent --base-url https://api.openai.com/v1 --model gpt-4.1-mini
```

### With Claude (cloud)

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # from console.anthropic.com
simple-agent --provider anthropic          # uses claude-opus-5-5 by default
simple-agent --provider anthropic --model claude-haiku-4-5    # cheaper and faster
```

### Using it

Type messages at the `you>` prompt. Lines starting with `/` are commands:

| Command    | What it does                          |
| ---------- | ------------------------------------- |
| `/help`    | list commands                         |
| `/tools`   | list the tools the agent can call     |
| `/history` | show the conversation so far          |
| `/usage`   | tokens used this session              |
| `/reset`   | start a new conversation              |
| `/exit`    | quit (Ctrl+D also works)              |

Add `-v` to also see what each tool returned. Pass a question as arguments to
ask once and exit: `simple-agent "what files are in this folder?"`.

### Config file

Instead of flags, copy `config.example.toml` to `config.toml` (in the folder you
run from, or `~/.config/simple-agent/config.toml`) and edit it. You can set the
provider, model, server URL, system prompt and step limit there. Flags always
win over the file. API keys are read from `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`
so they don't have to live in a file.

## How it works

```
 you ──> cli.py ──> agent.py ──> providers/  ──> LM Studio / Claude / ...
                       │
                       └──────> tools.py (calculator, read_file, ...)
```

| File | What it is |
| --- | --- |
| [`simple_agent/agent.py`](simple_agent/agent.py) | **The agent loop. Start reading here.** |
| [`simple_agent/providers/base.py`](simple_agent/providers/base.py) | The provider-neutral message types and the `Provider` interface. |
| [`simple_agent/providers/openai_compat.py`](simple_agent/providers/openai_compat.py) | Provider for LM Studio and any OpenAI-compatible server. |
| [`simple_agent/providers/anthropic_provider.py`](simple_agent/providers/anthropic_provider.py) | Provider for Claude via the Anthropic SDK. |
| [`simple_agent/tools.py`](simple_agent/tools.py) | The tool registry and the starter tools. |
| [`simple_agent/config.py`](simple_agent/config.py) | Merges defaults, `config.toml` and CLI flags. |
| [`simple_agent/cli.py`](simple_agent/cli.py) | The terminal REPL. |

### The loop

An LLM on its own can only produce text. An **agent** is a loop that lets the
model act:

1. Append the user's message to the conversation **history**.
2. Send the system prompt, the history and the **tool descriptions** to the model.
3. If the reply contains **tool calls**, run each tool, append the results to
   the history, and go back to step 2.
4. Otherwise the reply is the final answer. Show it and wait for the next message.

That's `Agent.ask()` in `agent.py`. `max_steps` caps step 2 so a confused model
can't loop forever. The history is the agent's only memory: every request
resends the whole conversation, which is why `/reset` makes it forget.

### Providers

Every API describes messages and tool calls a little differently. The agent
keeps history in its own tiny format (`Message`, `ToolCall` in `base.py`), and
each provider translates to and from its API. Comparing the two `_to_...`
functions is a good way to see the differences:

- **OpenAI-style APIs** have a separate `tool` role for tool results, and tool
  arguments arrive as a JSON *string* the model wrote (which small local models
  sometimes get wrong).
- **Anthropic's API** sends tool results back inside a `user` message as
  `tool_result` blocks, gives arguments as parsed JSON, and returns
  `thinking` blocks that must be sent back unchanged on the next request. The
  provider keeps the original reply in `Message.raw` for that.

The Anthropic provider also turns on server-side fallbacks, so if a safety
classifier declines a request the API retries it on a suitable fallback model
instead of just refusing.

### Tools

A tool is a normal Python function plus a description and a JSON Schema for its
arguments. The model never runs code itself: it asks for a tool by name, the
agent runs it, and the result goes back as text. Errors are returned as text
too, so the model can see what went wrong and try again.

The starter tools are deliberately read-only: `get_current_time`, `calculator`
(a safe arithmetic evaluator, not `eval`), `list_files` and `read_file`.

## Extending it

**Add a tool.** In `tools.py`:

```python
@default_tools.tool(
    "Count the words in a piece of text.",
    {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)
def word_count(text: str) -> str:
    return str(len(text.split()))
```

Restart and it appears in `/tools`. The description matters: it's the only
thing the model knows about the tool.

**Add a provider.** Write a class with `name`, `model` and
`chat(system, messages, tools) -> Reply` (see `providers/base.py`), then add a
branch to `create_provider()` in `providers/__init__.py` and an entry to
`PROVIDER_DEFAULTS` in `config.py`.

**Ideas for research and learning**, roughly in order of difficulty:

- Stream tokens as they arrive instead of waiting for the full reply.
- Save and load conversations (the history is just a list of dataclasses).
- Ask for confirmation before running a tool, then add a `write_file` or
  `run_shell` tool behind that gate.
- Trim or summarize old history when the conversation gets long.
- Log every request and reply to a JSONL file and compare how different models
  use the same tools.
- Add a web search tool, or connect MCP servers as tool sources.
- Run the same task on a local model and on Claude and measure the difference.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests need no API key or running model: the agent loop is tested with a
scripted fake provider, and both real providers are run against tiny local fake
servers so the actual SDK request formats are checked.
