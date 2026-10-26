import json

import httpx


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web for scientific facts, papers and reference data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 128},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code",
            "description": (
                "Run Python for calculations, simulations or symbolic reasoning. "
                "Missing packages in explicit imports are installed automatically before execution. "
                "The last top-level expression is displayed automatically unless its value is None. "
                "Use print() for additional output. Each call starts a fresh interpreter; "
                "files persist only within this question."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    },
]


class ToolRegistry:
    schemas = TOOL_SCHEMAS

    def __init__(self, search, code):
        self.tools = {"search": search, "code": code}

    async def execute(self, call: dict) -> str:
        function = call["function"]
        try:
            arguments = json.loads(function["arguments"])
            result = await self.tools[function["name"]](**arguments)
        except httpx.HTTPStatusError as exc:
            result = {"error": f"Tool HTTP error: {exc.response.status_code}"}
        except (ValueError, TypeError, KeyError, OSError, httpx.RequestError) as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        return json.dumps(result, ensure_ascii=False)
