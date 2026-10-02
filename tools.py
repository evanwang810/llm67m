"""Tool use: the call format, the training conversations, and the tools.

One module for all three so that what the model is trained to emit and what
chat.py knows how to run cannot drift apart.

The format, in tokens:

    <user> question <assistant> <call> search: some query <result> [1] site: text ...
    <assistant> the answer <eot>

The model writes everything from <call> through <result>; seeing <result> is
the signal for the runtime to stop, run the tool, and append its output and a
fresh <assistant>. The tool output is not trained on, since the model never
has to produce it. Calls are plain "name: argument" text rather than JSON: a
354M model gets a prefix and a line right far more reliably than nested
brackets.

Two tools. search, because a model this size gets facts wrong but is decent at
reading an answer out of text in front of it. calc, because it cannot do
arithmetic, and a calculator never gets it wrong.
"""

from __future__ import annotations

import ast
import html
import json
import math
import operator
import random
import re
import urllib.parse
import urllib.request

TOOL_CALL = 50259
TOOL_RESULT = 50260
MAX_CALLS = 3            # per reply, so a confused model cannot loop forever
RESULT_WORDS = 55        # per search result; three of them fit the context easily

# --------------------------------------------------------------------------- #
# calc
# --------------------------------------------------------------------------- #

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "round": round, "abs": abs, "min": min, "max": max,
          "log": math.log, "log10": math.log10, "sin": math.sin, "cos": math.cos,
          "tan": math.tan, "exp": math.exp, "floor": math.floor, "ceil": math.ceil}
_NAMES = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    raise ValueError("unsupported expression")


def fmt_number(x) -> str:
    if isinstance(x, float):
        if x.is_integer() and abs(x) < 1e15:
            return str(int(x))
        return f"{x:.6g}" if abs(x) >= 1e6 or abs(x) < 1e-3 else f"{round(x, 4)}"
    return str(x)


def calc(expr: str) -> str:
    """Evaluate arithmetic safely: numbers, operators, a few math functions."""
    cleaned = expr.replace("×", "*").replace("÷", "/").replace("^", "**").replace(",", "")
    try:
        return fmt_number(_eval(ast.parse(cleaned.strip(), mode="eval")))
    except ZeroDivisionError:
        return "error: division by zero"
    except Exception:
        return "error: could not evaluate that expression"


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #

_UA = {"User-Agent": "Mozilla/5.0 (llm354m chat tool)"}


def _clip(text: str, words: int = RESULT_WORDS) -> str:
    parts = re.sub(r"\s+", " ", text).strip().split(" ")
    return " ".join(parts[:words]) + (" ..." if len(parts) > words else "")


def _domain(url: str) -> str:
    host = urllib.parse.urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host or "web"


def format_results(results: list[tuple[str, str]]) -> str:
    """[(url, text)] to the numbered block the model is trained to read."""
    if not results:
        return "No results found."
    return "\n".join(f"[{i}] {_domain(u)}: {_clip(t)}" for i, (u, t) in enumerate(results, 1))


def _wikipedia(query: str, k: int, timeout: float) -> list[tuple[str, str]]:
    """The opening sentences of the top k matching articles, in one request.

    Article intros read like the web passages the model trains on. The search
    API's own snippets are cut mid sentence around the matched words, which
    reads like nothing in training.
    """
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": k,
        "prop": "extracts", "exintro": 1, "explaintext": 1, "exsentences": 3,
        "redirects": 1, "format": "json"})
    with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=timeout) as r:
        pages = json.load(r).get("query", {}).get("pages", {})
    ordered = sorted(pages.values(), key=lambda pg: pg.get("index", 0))
    return [("https://en.wikipedia.org/wiki/" + pg["title"].replace(" ", "_"),
             f"{pg['title']}. {pg.get('extract', '')}") for pg in ordered if pg.get("extract")]


def search(query: str, k: int = 3, timeout: float = 8.0) -> str:
    """Wikipedia search, the top k article intros.

    Wikipedia rather than a general web search engine because it needs no key
    and does not block scripts: DuckDuckGo's HTML endpoint answers scripted
    requests with a bot challenge. For the factual questions a model this size
    needs help with, it is also the better source.
    """
    try:
        return format_results(_wikipedia(query, k, timeout))
    except Exception:
        return "Search is unavailable right now."


def run(call: str) -> tuple[str, str, str]:
    """'name: argument' as the model wrote it, to (name, argument, output)."""
    name, _, arg = call.partition(":")
    name, arg = name.strip().lower(), arg.strip()
    if name == "search":
        return name, arg, search(arg)
    if name == "calc":
        return name, arg, calc(arg)
    return name, arg, f"error: unknown tool '{name}'"


# --------------------------------------------------------------------------- #
# training conversations
# --------------------------------------------------------------------------- #
# Turns are (role, text) with two roles beyond user and assistant: "call" is
# the model's tool call without its markers, "result" is the tool output.


_QWORDS = {"what", "how", "when", "where", "who", "whom", "which", "why", "is", "are", "was",
           "were", "does", "do", "did", "can", "could", "should", "will", "would", "has", "have"}


def _clean_query(q: str) -> str:
    q = re.sub(r"^[^A-Za-z0-9]+", "", q.strip()).rstrip(" ?.")
    return q


def msmarco_conversations(n: int, rng: random.Random, split: str = "validation") -> list:
    """Search, read, answer: real queries, real web passages, human answers.

    The validation split, 100k rows, is plenty for this and a fraction of the
    download of train. Queries the annotators could not answer from the
    passages become "could not find it" answers, about one in ten, so the model
    learns to say so rather than invent something.
    """
    from datasets import load_dataset

    rows = load_dataset("microsoft/ms_marco", "v2.1", split=split)
    order = list(range(len(rows)))
    rng.shuffle(order)
    out, unanswered = [], 0
    for i in order:
        if len(out) >= n:
            break
        row = rows[i]
        query = _clean_query(row["query"])
        p = row["passages"]
        if len(query) < 8 or not p["passage_text"]:
            continue
        answer = (row["wellFormedAnswers"] or row["answers"] or [""])[0].strip()
        selected = [j for j, s in enumerate(p["is_selected"]) if s]
        if answer in ("", "No Answer Present.") or not selected:
            if unanswered >= n // 10:
                continue
            unanswered += 1
            picks = rng.sample(range(len(p["passage_text"])), min(3, len(p["passage_text"])))
            answer = rng.choice([
                "I searched, but the results don't clearly answer that.",
                "I couldn't find a clear answer to that in the search results.",
                "The results I found don't say. You may need to check a more specific source.",
            ])
        else:
            others = [j for j in range(len(p["passage_text"])) if j not in selected]
            picks = selected[:1] + rng.sample(others, min(2, len(others)))
            rng.shuffle(picks)
            answer = answer[0].upper() + answer[1:]
            if not answer.endswith((".", "!", "?")):
                answer += "."
        results = format_results([(p["url"][j], p["passage_text"][j]) for j in picks])
        first = query.split()[0].lower()
        question = query[0].upper() + query[1:] + ("?" if first in _QWORDS else "")
        out.append([("user", question), ("call", f"search: {query.lower()}"),
                    ("result", results), ("assistant", answer)])
    return out


def calc_conversations(n: int, rng: random.Random) -> list:
    """Arithmetic the model hands to the calculator, then states plainly."""
    def num(lo, hi):
        return rng.randint(lo, hi)

    def money(x):
        return f"${x:,.2f}" if isinstance(x, float) and not x.is_integer() else f"${int(x):,}"

    makers = []

    def mul():
        a, b = num(12, 9999), num(3, 999)
        q = rng.choice([f"What is {a} times {b}?", f"Calculate {a} * {b}.", f"what's {a} x {b}",
                        f"Multiply {a:,} by {b}."])
        r = calc(f"{a} * {b}")
        return q, f"{a} * {b}", f"{a:,} times {b:,} is {int(r):,}."

    def add():
        xs = [num(10, 99999) for _ in range(rng.randint(2, 4))]
        q = rng.choice([f"What is {' + '.join(map(str, xs))}?", f"Add up {', '.join(map(str, xs))}.",
                        f"what's the sum of {' and '.join(map(str, xs))}"])
        r = calc(" + ".join(map(str, xs)))
        return q, " + ".join(map(str, xs)), f"The sum is {int(r):,}."

    def div():
        a, b = num(100, 99999), num(2, 97)
        q = rng.choice([f"What is {a} divided by {b}?", f"Calculate {a} / {b}.",
                        f"how much is {a} split {b} ways"])
        r = calc(f"{a} / {b}")
        return q, f"{a} / {b}", f"{a:,} divided by {b} is {r}."

    def pct():
        p, x = rng.choice([5, 10, 12, 15, 18, 20, 25, 30, 35, 40, 75]), num(20, 9999)
        q = rng.choice([f"What is {p}% of {x}?", f"Calculate {p} percent of {x}.",
                        f"What's a {p}% tip on a ${x} bill?"])
        r = calc(f"{p} / 100 * {x}")
        tail = f"A {p}% tip on ${x} is {money(float(r))}." if "tip" in q else f"{p}% of {x:,} is {r}."
        return q, f"{p} / 100 * {x}", tail

    def sqrt():
        x = num(2, 99999)
        q = rng.choice([f"What is the square root of {x}?", f"sqrt of {x}?",
                        f"Find the square root of {x}."])
        r = calc(f"sqrt({x})")
        return q, f"sqrt({x})", f"The square root of {x:,} is about {r}."

    def shop():
        k, price = num(2, 40), rng.choice([num(1, 99) + 0.99, num(2, 300) * 1.0, num(1, 50) + 0.5])
        item = rng.choice(["notebooks", "tickets", "shirts", "plants", "books", "mugs"])
        q = f"I'm buying {k} {item} at ${price:.2f} each. How much is that in total?"
        r = calc(f"{k} * {price}")
        return q, f"{k} * {price}", f"{k} {item} at ${price:.2f} each comes to {money(float(r))}."

    def speed():
        d, h = num(20, 900), rng.choice([1.5, 2, 2.5, 3, 4, 5, 6, 7.5])
        q = f"If a car travels {d} miles in {h} hours, what is its average speed?"
        r = calc(f"{d} / {h}")
        return q, f"{d} / {h}", f"Its average speed is {r} miles per hour."

    def split():
        total, people = num(30, 2000), num(2, 12)
        q = f"We spent ${total} on dinner and there are {people} of us. How much does each person pay?"
        r = calc(f"{total} / {people}")
        return q, f"{total} / {people}", f"Each person pays {money(round(float(r), 2))}."

    makers = [mul, add, div, pct, sqrt, shop, speed, split]
    out = []
    for _ in range(n):
        q, expr, answer = rng.choice(makers)()
        out.append([("user", q), ("call", f"calc: {expr}"), ("result", calc(expr)),
                    ("assistant", answer)])
    return out


SPECIAL = {"@msmarco": msmarco_conversations, "@calc": calc_conversations}
