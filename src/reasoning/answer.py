import re


def extract_boxed(text: str) -> str | None:
    """Last nonempty balanced box outside Qwen's thinking blocks."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.split("<think>", 1)[0]
    # Some servers return the opening think marker implicitly.
    text = text.rsplit("</think>", 1)[-1]
    answer = None
    for match in re.finditer(r"\\boxed\s*\{", text):
        depth = 1
        start = index = match.end()
        while index < len(text):
            char = text[index]
            if char == "\\":
                index += 2  # Escaped braces do not affect the nesting depth.
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    value = text[start:index].strip()
                    if value:
                        answer = value
                    break
            index += 1
    return answer
