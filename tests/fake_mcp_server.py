"""A tiny MCP server for the tests, speaking JSON-RPC over stdin/stdout.

It offers two tools, "add" and "fail", and also sends the kind of extra
messages real servers send (a log notification, a ping) to check the client
copes with them.
"""

import json
import sys

TOOLS = [
    {
        "name": "add",
        "description": "Add two numbers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
    {"name": "fail", "description": "Always fails.", "inputSchema": {"type": "object", "properties": {}}},
]


def send(message):
    print(json.dumps(message), flush=True)


print("fake server starting", file=sys.stderr)
for line in sys.stdin:
    message = json.loads(line)
    method, request_id = message.get("method"), message.get("id")
    if request_id is None:  # A notification, or our ping being answered.
        continue
    if method == "initialize":
        send({"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info", "data": "hello"}})
        result = {"protocolVersion": message["params"]["protocolVersion"], "capabilities": {"tools": {}},
                  "serverInfo": {"name": "fake", "version": "1"}}
    elif method == "tools/list":
        # Two pages, to check the client follows nextCursor.
        if message["params"].get("cursor") == "page2":
            result = {"tools": TOOLS[1:]}
        else:
            result = {"tools": TOOLS[:1], "nextCursor": "page2"}
    elif method == "tools/call":
        send({"jsonrpc": "2.0", "id": "ping-1", "method": "ping"})
        name, args = message["params"]["name"], message["params"]["arguments"]
        if name == "add":
            result = {"content": [{"type": "text", "text": str(args["a"] + args["b"])}]}
        else:
            result = {"content": [{"type": "text", "text": "something broke"}], "isError": True}
    else:
        send({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"no method {method}"}})
        continue
    send({"jsonrpc": "2.0", "id": request_id, "result": result})
