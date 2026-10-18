from .answer import extract_boxed
from .prompts import CONTINUE_PROMPT, SYSTEM_PROMPT


class ReasoningEngine:
    def __init__(self, model, tools, max_rounds: int = 20):
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        self.model = model
        self.tools = tools
        self.max_rounds = max_rounds

    async def run(self, question: str) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        trace = []
        status, answer = "max_rounds", None
        for step in range(1, self.max_rounds + 1):
            response = await self.model.complete(messages, self.tools.schemas)
            choice = response["choices"][0]
            raw = choice["message"]
            # Keep the wire response in the trace; replay only chat message fields.
            message = {"role": "assistant", "content": raw.get("content") or ""}
            for key in ("tool_calls", "reasoning_content", "reasoning"):
                if raw.get(key) is not None:
                    message[key] = raw[key]
            messages.append(message)
            trace.append({
                "round": step,
                "message": raw,
                "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage", {}),
            })
            # A truncated response can contain an incomplete tool call or tentative box.
            if choice.get("finish_reason") == "length":
                status = "max_tokens"
                break
            calls = message.get("tool_calls") or []
            if calls:
                for call in calls:
                    output = await self.tools.execute(call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": output,
                    })
                continue
            answer = extract_boxed(message["content"])
            if answer is not None:
                status = "completed"
                break
            if step < self.max_rounds:
                messages.append({"role": "user", "content": CONTINUE_PROMPT})
        return {
            "question": question,
            "status": status,
            "answer": answer,
            "rounds": len(trace),
            "messages": messages,
            "trace": trace,
        }
