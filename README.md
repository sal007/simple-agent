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

| Command        | What it does                      |
| -------------- | --------------------------------- |
| `/help`        | list commands                     |
| `/tools`       | list the tools the agent can call |
| `/plugins`     | list plugin files and their tools |
| `/reload`      | load the plugin files again       |
| `/history`     | show the conversation so far      |
| `/usage`       | tokens used this session          |
| `/reset`       | start a new conversation          |
| `/context`     | how big the conversation is       |
| `/compact`     | summarize older turns now         |
| `/save [name]` | save the conversation             |
| `/load <name>` | load a saved conversation         |
| `/sessions`    | list saved conversations          |
| `/exit`        | quit (Ctrl+D also works)          |

Replies stream in as the model writes them. Add `--no-stream` to wait for whole
replies instead (useful if a server doesn't support streaming). Add `-v` to also
see what each tool returned. Pass a question as arguments to
ask once and exit: `simple-agent "what files are in this folder?"`.

### Saved sessions

Conversations can be saved and picked up later:

```
you> /save tea-research
  (saved to sessions/tea-research.json)
...
$ simple-agent --resume tea-research
(resumed tea-research: 6 messages, saved 2026-09-28T23:58:10+00:00)
```

When you quit, the conversation is also saved as `last`, so
`simple-agent --resume last` continues where you stopped. `/sessions` lists
what's saved and `/load <name>` switches to another conversation.

A session file is plain JSON: the history plus the provider, model and token
usage. Because the history is provider-neutral, you can save a conversation
with a local model and resume it with Claude (or the other way round) to see
how each continues it. Claude's replies are stored with their original blocks,
thinking included, so a resumed Claude conversation is replayed exactly.

### Trace mode

Add `--trace` to watch the agent loop step by step. You see what is sent to the
model, what comes back (text, tool calls, why it stopped, tokens, how long it
took), and every tool result:

```
$ simple-agent --trace "what is 6*7?"
  trace> logging to traces/20260928-234250-openai-qwen2.5-7b-instruct.jsonl
  trace> step 1: sending 1 message to the model (1 new)
  trace>   + [user] what is 6*7?
  trace> step 1: reply in 1.84s, stop=tool_calls, tokens: input 412, output 21
  trace>   tool call: calculator({"expression": "6*7"})
  trace>   tool result (0.1 ms): 42
  trace> step 2: sending 3 messages to the model (1 new)
  trace>   + [tool result] 42
  trace> step 2: reply in 0.93s, stop=stop, tokens: input 445, output 9
  trace>   text: 6 × 7 = 42.
agent> 6 × 7 = 42.
```

Notice that step 2 sends all three messages again: the model has no memory of
its own, so every request carries the whole conversation.

Each run is also saved to `traces/` as a JSONL file (one JSON object per line),
named after the time, provider and model. The events are `run_start` (system
prompt and tools), `turn_start`, `request` (only the messages that are new since
the last request), `reply` (with `usage`, `seconds` and `stop_reason`), `tool`,
`turn_end`, `error`, `reset` and `context`. That makes it easy to compare models on the
same questions, for example total tokens per run:

```bash
python -c "import json,sys; print(sum(e['usage'].get('output_tokens',0) for e in map(json.loads, open(sys.argv[1])) if e['event']=='reply'))" traces/<file>.jsonl
```

Set `trace = true` in `config.toml` to have it on all the time.

### Context management

Every request resends the whole conversation, so it keeps growing, and tool
results (a whole file, a long directory listing) make it grow fast. Once it no
longer fits the model's context window the request fails, or a local server
quietly cuts off the start of the conversation. `context.py` keeps the history
under a limit (`max_tokens`, 8000 by default) in two ways, cheapest first:

1. **Clear old tool results.** Before each model call, if the history is over
   the limit, every tool result except the newest 3 is replaced with a
   placeholder like `[old tool result cleared to save space: it was 9412
   characters]`. The model still sees which tools it called and with what
   arguments, so it can call one again if it needs the output back.
2. **Compact (summarize).** At the start of a turn, if the history is still
   over the limit, the model is asked to summarize everything except the last 2
   turns. Those older messages are replaced by the summary. This never happens
   in the middle of a turn, so a tool call is never split from its result.

You see it when it happens:

```
you> and what about the tests folder?
  context> cleared 5 old tool results (~12800 -> ~8410 tokens)
  context> history is ~8410 tokens; summarizing 14 older messages...
  context> replaced 14 older messages with a summary (~8410 -> ~1900 tokens)
```

`/context` shows the current size and settings, `/compact` summarizes right
away, and `-v` also prints the summary. Set the limit a little below the
context length you loaded the model with (in LM Studio that is the "Context
Length" setting, often 4096 by default) in the `[context]` section of
`config.toml`, where each strategy can also be turned off on its own.

The sizes are estimates (about 4 characters per token), which is enough to
decide when to act; the real token counts are in `/usage` and the trace.

**To compare the two strategies**, run the same long task with `--trace` and
different settings, for example `compact = false` (clearing only) against
`clear_tool_results = false` (summaries only), with a small `max_tokens` so
they kick in early. The trace shows each `context` event, and after one the
next request lists the whole history again, so you see exactly what the model
was left with. Things to look at: input tokens per request, whether the model
had to re-read files it had cleared, and what the summaries kept or lost.

With Claude, editing earlier history would normally break its saved thinking
blocks (the API rejects a request whose history changed underneath them), so
the Anthropic provider asks the API to drop the thinking blocks that no longer
match instead (`block_binding` with `drop_block`).

### Eval runner

`simple-agent eval` runs a fixed set of tasks against a model and scores the
answers, so you can compare models (or prompts, or settings) with real numbers:

```
$ simple-agent eval
Running 11 tasks x1 on openai · qwen2.5-7b-instruct

  PASS  files/read-a-file  (2 steps, 1291 tokens, 2.3s)
  FAIL  files/count-files  (2 steps, 1302 tokens, 2.1s)
        answer_number: 3 not found in the answer
  ...
9/11 passed (82%), 26 steps, 14210 input + 611 output tokens, 31.4s
Results saved to eval_results/20260930-101500-openai-qwen2.5-7b-instruct.json
```

Each task runs a fresh agent (the same loop, tools and plugins as the chat) in
its own empty temporary folder, so tasks can't affect each other or your files.
`write_file` is allowed without asking there; other tools that normally ask,
like `run_shell`, are declined. The results file has, for every task, whether
it passed, the reason for any failed check, the answer, the number of steps
(model calls), the tools called, tokens, time, and the whole conversation.

Compare saved runs side by side:

```
$ simple-agent eval --compare eval_results/*-qwen*.json eval_results/*-claude*.json
task                         openai/qwen2.5-7b-instruct   anthropic/claude-opus-5-5
files/count-files            FAIL 2 steps 1302 tok        pass 2 steps 2410 tok
...
```

Other options: `--only math/` runs only matching tasks, `--repeat 5` runs each
task several times (models don't answer the same way every time, so one run
can mislead), and `--provider`, `--model` and `--base-url` work as in the chat.

**Tasks** are TOML files in `evals/`. The starter set covers arithmetic
(`math.toml`), reading files (`files.toml`) and multi-step tool use
(`multi_step.toml`). A task is an id, a prompt, optional files to create in its
folder, and one or more checks that must all pass:

```toml
[[task]]
id = "sum-a-file"
prompt = "Add up all the numbers in numbers.txt and tell me the total."
files."numbers.txt" = "17\n42\n8\n133\n5\n"
check.answer_number = 205
check.tool_used = "read_file"
```

| Check | Passes when |
| --- | --- |
| `answer_contains` | the answer contains the text (or every text in a list), ignoring case |
| `answer_contains_any` | the answer contains at least one of a list of texts |
| `answer_not_contains` | the answer contains none of them |
| `answer_matches` | the answer matches a regular expression |
| `answer_number` | the number appears in the answer (`7,006,652` and `$64.80` count) |
| `file_exists` | the file (or files) exist in the task's folder afterwards |
| `file_contains` | `{ "file.txt" = "text" }`: the file exists and contains the text |
| `tool_used` | the model called this tool (or all of a list of tools) |
| `max_steps` | the task took at most this many model calls |

`approve = ["write_file", "run_shell"]` on a task changes which tools may run
without asking. Add your own `.toml` files to `evals/`, or run a different set
with `simple-agent eval path/to/tasks`.

### Plugins

To add a tool without touching the agent's code, put a `.py` file in a
`plugins/` folder (next to where you run the agent, or in
`~/.config/simple-agent/plugins/`). Every file there is loaded at start-up and
its tools join the built-in ones:

```python
# plugins/greet.py
from simple_agent.plugins import tool

@tool(
    "Greet someone by name.",
    {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
)
def greet(name: str) -> str:
    return f"Hello, {name}!"
```

```
$ simple-agent
(plugin plugins/greet.py: greet)
(plugin plugins/word_count.py: word_count)
```

`@tool` takes the same arguments as the built-in tools in `tools.py`, including
`confirm=True` to ask before each call. While experimenting, edit the file and
type `/reload`: the plugins are loaded again and the next message uses the new
version, with no restart and the conversation kept. A plugin tool with the same
name as a built-in one replaces it, which is an easy way to try a different
description of, say, `read_file` and see how the model's behavior changes.

A file that fails to load is reported and skipped. Files starting with `_` are
not loaded, so shared helpers can live there. `plugins/word_count.py` is a
working example to copy. Set `plugin_dirs` in `config.toml` to use other
folders, or start with `--no-plugins` to skip them.

Plugins are ordinary Python that runs with your permissions, so only use ones
you wrote or trust.

### MCP servers

[MCP](https://modelcontextprotocol.io) (Model Context Protocol) is a standard
way for programs to offer tools to agents. There are ready-made servers for
files, git, databases, web fetching and much more, written in any language.
Add one to `config.toml` and its tools appear next to the built-in ones:

```toml
[mcp_servers.files]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "."]

[mcp_servers.fetch]
command = "uvx"
args = ["mcp-server-fetch"]
```

```
$ simple-agent
(MCP server 'files': 14 tools (asks before each call))
(MCP server 'fetch': 1 tools (asks before each call))
you> /tools
  ...
  files__read_text_file (asks first): Read the complete contents of a file ... (from MCP server 'files')
```

Each tool is named `<server>__<tool>` so names can't clash. Because an MCP
server can do anything, every call asks y/N first, like `write_file`. Add
`trusted = true` to a server's section to let its tools run without asking.
A server can also get extra environment variables with `env = { KEY = "value" }`
(for example an API token). A server that fails to start is reported and
skipped, and `--no-mcp` starts without any servers.

`simple_agent/mcp.py` is a small hand-written client, so the whole protocol is
readable in one file: the agent starts the server as a subprocess and sends it
one line of JSON per message on stdin (`initialize`, then `tools/list`, then
`tools/call` for each call), reading the replies from stdout. The first run of
an `npx` or `uvx` server downloads it, which can take a while. Only servers
that run locally over stdio are supported; see the ideas at the end for HTTP.

### Config file

Instead of flags, copy `config.example.toml` to `config.toml` (in the folder you
run from, or `~/.config/simple-agent/config.toml`) and edit it. You can set the
provider, model, server URL, system prompt, step limit, streaming, trace mode,
context management, MCP servers, plugin folders and where traces and sessions are stored there. Flags always
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
| [`simple_agent/plugins.py`](simple_agent/plugins.py) | Loads extra tools from `.py` files in the plugin folders. |
| [`plugins/word_count.py`](plugins/word_count.py) | An example plugin. |
| [`simple_agent/mcp.py`](simple_agent/mcp.py) | A small MCP client: adds tools from MCP servers to the registry. |
| [`simple_agent/sessions.py`](simple_agent/sessions.py) | Saves and loads conversations as JSON. |
| [`simple_agent/context.py`](simple_agent/context.py) | Context management: clears old tool results and summarizes old turns. |
| [`simple_agent/evals.py`](simple_agent/evals.py) | The eval runner: runs tasks, checks answers, saves and compares results. |
| [`evals/`](evals/) | The starter eval tasks. |
| [`simple_agent/trace.py`](simple_agent/trace.py) | Trace mode: prints each loop step and writes the JSONL log. |
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

### Streaming

Without streaming, a request returns the whole reply at once, so you stare at a
blank line until the model is done. With streaming the server sends the reply
in small chunks as it's generated, and `Agent` passes each text chunk to an
`on_text` callback that the CLI prints straight away.

Each provider has both paths side by side, so you can compare them. Text is
easy to stream, but tool calls arrive in pieces too: the OpenAI-style stream
sends the tool name first and then the JSON arguments a few characters at a
time, and `_chat_streaming()` in `openai_compat.py` glues them back together.
The Anthropic SDK's `messages.stream()` helper does that assembly for you and
hands back the complete message at the end.

Trace mode turns streaming off so each step prints as one complete reply.

### Tools

A tool is a normal Python function plus a description and a JSON Schema for its
arguments. The model never runs code itself: it asks for a tool by name, the
agent runs it, and the result goes back as text. Errors are returned as text
too, so the model can see what went wrong and try again.

The starter tools come in two kinds. Four only read: `get_current_time`,
`calculator` (a safe arithmetic evaluator, not `eval`), `list_files` and
`read_file`. Two change things, `write_file` and `run_shell`, so they are
registered with `confirm=True` and the agent asks you first:

```
you> save a haiku about tea to tea.txt
  tool> write_file({"path": "tea.txt", "content": "Steam curls from the cup..."})
  approve> the agent wants to run write_file with:
    path: tea.txt
    content: Steam curls from the cup
             ...
  Allow? [y/N] y
agent> Saved the haiku to tea.txt.
```

Anything but `y` declines, and the model is told you said no. When nobody can
answer (input piped in, or tests), these tools are always declined. `/tools`
marks them "(asks first)". The check lives in `ToolRegistry.run()`, and the
CLI plugs in the prompt by setting `agent.tools.approve`. Read each command
before you allow it: `run_shell` runs exactly what the model wrote, with your
permissions.

## Extending it

**Add a tool.** The quickest way is a plugin (see [Plugins](#plugins)). To make
it one of the built-in tools instead, add it to `tools.py`:

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

Restart and it appears in `/tools`. Add `confirm=True` to the decorator if the
tool changes anything, so the user has to approve each call. The description matters: it's the only
thing the model knows about the tool.

**Add a provider.** Write a class with `name`, `model` and
`chat(system, messages, tools) -> Reply` (see `providers/base.py`), then add a
branch to `create_provider()` in `providers/__init__.py` and an entry to
`PROVIDER_DEFAULTS` in `config.py`.

**Ideas for research and learning**, roughly in order of difficulty:

- Measure time to first token with streaming on, and compare models.
- Remember "always allow" answers per tool for the rest of a session.
- Try other context strategies in `context.py`: a sliding window that drops
  the oldest turns, or a memory file the agent writes notes to.
- Write a small script that reads two trace logs and compares how different
  models used the same tools.
- Add a web search tool, or find an MCP server that has one.
- Support MCP servers that run over HTTP ("Streamable HTTP" in the MCP spec):
  the same JSON-RPC messages, sent as POST requests instead of stdin lines.
- Run the eval tasks on a local model and on Claude, then change one thing
  (the system prompt, a tool description via a plugin, the context limit) and
  see how the scores move.
- Add an eval check that asks a second model to grade the answer ("LLM as a
  judge") for tasks with no single right answer.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests need no API key or running model: the agent loop is tested with a
scripted fake provider, and both real providers are run against tiny local fake
servers so the actual SDK request formats are checked.
