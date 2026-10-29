"""Final-response parsing and deterministic local grading, without an LLM judge."""

import ast
import json
import math
import operator
import re


def final_content(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.split("<think>", 1)[0]
    return text.rsplit("</think>", 1)[-1].strip()


def last_box(text: str) -> str | None:
    answer = None
    for match in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        depth, start, i = 1, match.end(), match.end()
        while i < len(text):
            if text[i] == "\\":
                i += 2
                continue
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    answer = text[start:i].strip()
                    break
            i += 1
    return answer


def parse_answer(content: str, kind: str, labels: str = "ABCDEF") -> str | None:
    content = final_content(content)
    candidate = last_box(content)
    if candidate is None:
        try:
            value = json.loads(content.strip().removeprefix("```json").removesuffix("```").strip())
            candidate = str(value["answer"]) if isinstance(value, dict) and "answer" in value else None
        except (ValueError, TypeError):
            pass
    if candidate is None:
        marked = re.findall(r"(?:final\s+answer|answer|答案)\s*[:：]\s*(.+)", content, re.IGNORECASE)
        candidate = marked[-1].strip() if marked else (content.splitlines()[-1].strip() if content else None)
    if candidate is None:
        return None
    if kind != "choice":
        return candidate
    for _ in range(4):
        candidate = re.sub(r"\\(?:text|mathrm|mathbf)\s*\{([^{}]*)\}", r"\1", candidate)
    match = re.fullmatch(r"(?:option\s+)?[\(\[]?([A-Z])[\)\]]?\.?", candidate.strip("$ *"), re.IGNORECASE)
    return match.group(1).upper() if match and match.group(1).upper() in labels else None


def number(value: str | float | int) -> float | None:
    if isinstance(value, bool):
        return None
    text = str(value).strip().replace("−", "-").replace("，", ",").replace(",", "")
    if len(text) > 512:
        return None
    text = text.rsplit("=", 1)[-1].strip(" $")
    text = re.sub(r"\\(?:mathrm|text)\s*\{[^{}]*\}", "", text)
    for _ in range(8):
        old = text
        text = re.sub(r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", text)
        text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)
        if old == text:
            break
    for before, after in [(r"\times", "*"), (r"\cdot", "*"), ("×", "*"), (r"\pi", "pi"),
                          (r"\left", ""), (r"\right", ""), (r"\,", ""), (r"\!", ""),
                          ("^", "**"), ("{", "("), ("}", ")")]:
        text = text.replace(before, after)
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.Pow: operator.pow}

    def calculate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in ("pi", "e"):
            return getattr(math, node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = calculate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in binary:
            left, right = calculate(node.left), calculate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 308:
                raise ValueError("exponent is too large")
            return binary[type(node.op)](left, right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt" and len(node.args) == 1 and not node.keywords:
            return math.sqrt(calculate(node.args[0]))
        raise ValueError("not a numeric expression")

    try:
        tree = ast.parse(text.strip(), mode="eval")
        if len(list(ast.walk(tree))) > 80:
            return None
        result = calculate(tree.body)
        return float(result) if isinstance(result, (float, int)) and math.isfinite(result) else None
    except (ValueError, TypeError, SyntaxError, OverflowError, ZeroDivisionError):
        return None


def valid_reference(record: dict) -> bool:
    if record["kind"] != "choice":
        return True
    labels = "".join(record.get("options", {})) or "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    reference = str(record["reference"]).strip().upper()
    return len(reference) == 1 and reference in labels


def grade(record: dict, content: str, status: str) -> dict:
    prediction = parse_answer(content, record["kind"], "".join(record.get("options", {})) or "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    reference = record["reference"]
    if not valid_reference(record):
        return {"prediction": prediction, "reference": reference, "is_correct": None,
                "parsed": prediction is not None, "reference_valid": False,
                "reference_error": "Source reference is not a single available choice label; excluded from accuracy"}
    valid = status == "completed" and prediction is not None
    result = {"prediction": prediction, "reference": reference, "is_correct": False, "parsed": prediction is not None}
    if record["kind"] == "numeric":
        predicted_number = number(prediction) if prediction is not None else None
        result.update({"predicted_number": predicted_number, "relative_tolerance": 0.05, "absolute_tolerance": 0.0})
        result["parsed"] = predicted_number is not None
        result["is_correct"] = bool(valid and predicted_number is not None and math.isclose(predicted_number, reference, rel_tol=0.05, abs_tol=0.0))
    elif record["kind"] == "choice":
        result["is_correct"] = bool(valid and prediction.upper() == str(reference).strip().upper())
    else:
        # SciEval's published evaluator uses reference substring matching for these tasks.
        result["is_correct"] = bool(valid and str(reference).strip().casefold() in prediction.casefold())
    return result
