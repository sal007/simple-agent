"""An example plugin. Every .py file in this folder is loaded when the agent starts.

Copy this file to make your own tool, then type /reload in the agent to load
your changes without restarting. See "Plugins" in the README.
"""

from collections import Counter

from simple_agent.plugins import tool


@tool(
    "Count the words in a piece of text and list the most common ones.",
    {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The text to count."},
            "top": {"type": "integer", "description": "How many of the most common words to list. Defaults to 5."},
        },
        "required": ["text"],
    },
)
def word_count(text: str, top: int = 5) -> str:
    words = [w.strip(".,;:!?\"'()").lower() for w in text.split()]
    words = [w for w in words if w]
    common = ", ".join(f"{w} ({n})" for w, n in Counter(words).most_common(top))
    return f"{len(words)} words. Most common: {common or 'none'}"
