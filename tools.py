"""Tool calling: the wire format, the training conversations, and a registry.

The model does not know any particular tool. Whatever is calling it declares
what exists, in the prompt, and the model calls against that declaration:

    TOOLS
    web_search(query) - search the web for information
    calculator(expression) - do arithmetic

    <user> how tall is mount fuji <assistant> <call> web_search: height of
    mount fuji <result> [1] en.wikipedia.org: ... <assistant> Mount Fuji is
    3,776 metres tall. <eot>

The model writes from <call> to <result>; the <result> marker is the signal to
the runtime to stop, run the tool, and append the output with a fresh
<assistant>. Tool output is never trained on, since the model never produces
it.

That the tool list is in the prompt is the whole point. The training data
varies the names, the argument names, the descriptions, the number of tools,
their order, and includes tools that are declared and never needed, plus
conversations with no tool list at all. A model that memorised "search" would
score the same on data with one fixed tool and be useless with a tool it had
not seen; varying the list is what makes it read the list.

The tools here are examples for the chat tool to register, not part of the
model. register() takes any callable.
"""

from __future__ import annotations

import ast
import datetime
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
RESULT_WORDS = 55        # per search result; three of them fit the context
HEADER = "TOOLS"


# --------------------------------------------------------------------------- #
# the declaration block
# --------------------------------------------------------------------------- #


def render(specs: list[tuple[str, str, str]]) -> str:
    """[(name, arg, description)] to the block that goes in front of a prompt."""
    # str(), because a caller that passes the wrong argument order should get a
    # readable line rather than a function repr in the model's prompt.
    lines = [f"{name}({arg}) - {str(desc)}" for name, arg, desc in specs]
    return HEADER + "\n" + "\n".join(lines)


def parse_call(text: str) -> tuple[str, str]:
    """'name: argument' as the model wrote it. Tolerates name(argument) too."""
    text = text.strip()
    m = re.match(r"^([A-Za-z_][\w.]*)\s*(?::|\()\s*(.*?)\)?\s*$", text, re.S)
    if not m:
        return "", text
    return m.group(1).lower(), m.group(2).strip().strip('"\'')


# --------------------------------------------------------------------------- #
# a registry, so nothing is built in
# --------------------------------------------------------------------------- #


class Registry:
    """The tools one caller is offering: name to (arg, description, function)."""

    def __init__(self) -> None:
        self.tools: dict[str, tuple[str, str, callable]] = {}

    def register(self, name: str, arg: str, desc: str, fn) -> None:
        self.tools[name.lower()] = (arg, desc, fn)

    def remove(self, name: str) -> bool:
        return self.tools.pop(name.lower(), None) is not None

    def specs(self) -> list[tuple[str, str, str]]:
        return [(n, a, d) for n, (a, d, _) in self.tools.items()]

    def declaration(self) -> str:
        return render(self.specs()) if self.tools else ""

    def run(self, call: str) -> tuple[str, str, str]:
        """(name, argument, output). An unknown name is reported, not guessed."""
        name, arg = parse_call(call)
        entry = self.tools.get(name)
        if entry is None:
            known = ", ".join(sorted(self.tools)) or "none"
            return name, arg, f"error: no tool named '{name}'. Available: {known}"
        try:
            return name, arg, str(entry[2](arg))
        except Exception as exc:
            return name, arg, f"error: {type(exc).__name__}: {exc}"

    def __len__(self) -> int:
        return len(self.tools)


# --------------------------------------------------------------------------- #
# example tools
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
    """Arithmetic through the AST: numbers, operators, a few math functions."""
    cleaned = expr.replace("×", "*").replace("÷", "/").replace("^", "**").replace(",", "")
    try:
        return fmt_number(_eval(ast.parse(cleaned.strip(), mode="eval")))
    except ZeroDivisionError:
        return "error: division by zero"
    except Exception:
        return "error: could not evaluate that expression"


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


def search(query: str, k: int = 3, timeout: float = 8.0) -> str:
    """Wikipedia search: the opening sentences of the top k articles.

    Wikipedia rather than a general web engine because it needs no key and does
    not block scripts; DuckDuckGo's HTML endpoint answers scripted requests
    with a bot challenge. Article intros also read like the passages the model
    trains on, where the search API's own snippets are cut mid sentence.
    """
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": k,
        "prop": "extracts", "exintro": 1, "explaintext": 1, "exsentences": 3,
        "redirects": 1, "format": "json"})
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=timeout) as r:
            pages = json.load(r).get("query", {}).get("pages", {})
        ordered = sorted(pages.values(), key=lambda pg: pg.get("index", 0))
        return format_results([
            ("https://en.wikipedia.org/wiki/" + pg["title"].replace(" ", "_"),
             f"{pg['title']}. {html.unescape(pg.get('extract', ''))}")
            for pg in ordered if pg.get("extract")])
    except Exception:
        return "Search is unavailable right now."


_UNITS = {  # to a base unit, by dimension
    "length": ({"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0, "in": 0.0254, "inch": 0.0254,
                "ft": 0.3048, "feet": 0.3048, "foot": 0.3048, "yd": 0.9144, "mi": 1609.344,
                "mile": 1609.344, "miles": 1609.344}, "m"),
    "mass": ({"mg": 1e-6, "g": 0.001, "kg": 1.0, "oz": 0.0283495, "lb": 0.453592,
              "lbs": 0.453592, "pound": 0.453592, "pounds": 0.453592, "ton": 1000.0}, "kg"),
    "volume": ({"ml": 0.001, "l": 1.0, "liter": 1.0, "litre": 1.0, "cup": 0.236588,
                "pint": 0.473176, "quart": 0.946353, "gal": 3.78541, "gallon": 3.78541}, "l"),
}


def convert(arg: str) -> str:
    """'12 km to miles', including the temperature scales."""
    m = re.match(r"^\s*(-?[\d.,]+)\s*([a-zA-Z°]+)\s*(?:to|in|into|->)\s*([a-zA-Z°]+)\s*$", arg)
    if not m:
        return "error: use the form '12 km to miles'"
    value, src, dst = float(m.group(1).replace(",", "")), m.group(2).lower(), m.group(3).lower()
    temps = {"c", "celsius", "f", "fahrenheit", "k", "kelvin", "°c", "°f"}
    if src in temps and dst in temps:
        c = (value - 32) * 5 / 9 if src.endswith(("f", "fahrenheit")) else \
            value - 273.15 if src.startswith("k") else value
        out = c * 9 / 5 + 32 if dst.endswith(("f", "fahrenheit")) else \
            c + 273.15 if dst.startswith("k") else c
        return fmt_number(round(out, 4))
    for table, _ in _UNITS.values():
        if src in table and dst in table:
            return fmt_number(round(value * table[src] / table[dst], 6))
    return f"error: cannot convert {src} to {dst}"


def time_in(zone: str) -> str:
    """Current time in an IANA timezone, for example 'Asia/Tokyo'."""
    try:
        from zoneinfo import ZoneInfo
        name = zone.strip().replace(" ", "_")
        now = datetime.datetime.now(ZoneInfo(name))
        return now.strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        return f"error: unknown timezone '{zone}'"


def example_registry() -> Registry:
    """The tools the chat tool offers unless its caller changes them."""
    reg = Registry()
    reg.register("search", "query", "search the web for information", search)
    reg.register("calc", "expression", "evaluate a mathematical expression", calc)
    reg.register("convert", "value", "convert between units, like '12 km to miles'", convert)
    reg.register("time", "timezone", "the current time in a timezone", time_in)
    return reg


# --------------------------------------------------------------------------- #
# training: pools of tool declarations
# --------------------------------------------------------------------------- #
# Several names and wordings per kind, so no single spelling is the one the
# model learns. KINDS maps a kind to (names, argument names, descriptions).

KINDS = {
    "search": (
        ["search", "web_search", "lookup", "find", "google", "search_web", "wiki",
         "query", "browse", "web", "look_up", "find_info", "retrieve", "search_docs"],
        ["query", "q", "terms", "search_query", "text", "keywords"],
        ["search the web for information", "look up information online",
         "search the internet", "find information on the web",
         "search for facts about a topic", "look something up",
         "retrieve relevant web pages", "search an encyclopedia"],
    ),
    "calc": (
        ["calc", "calculator", "compute", "math", "evaluate", "arithmetic", "eval_math",
         "do_math", "calculate", "solve"],
        ["expression", "expr", "equation", "formula", "input", "math"],
        ["evaluate a mathematical expression", "do arithmetic",
         "calculate the result of an expression", "compute a numeric expression",
         "work out a sum exactly", "evaluate arithmetic"],
    ),
    "convert": (
        ["convert", "unit_convert", "units", "conversion", "convert_units", "uconv"],
        ["value", "amount", "conversion", "input", "quantity"],
        ["convert between units", "convert a value from one unit to another",
         "change units, like '12 km to miles'", "do a unit conversion"],
    ),
    "time": (
        ["time", "current_time", "clock", "time_in", "localtime", "now"],
        ["timezone", "zone", "location", "tz", "place"],
        ["the current time in a timezone", "get the current local time",
         "look up what time it is somewhere", "current time for a timezone"],
    ),
}

# Declared but never called in the conversation that lists them. Having tools
# present that the question does not need is what stops the model calling
# whatever happens to be first in the list.
DISTRACTORS = [
    ("weather", "location", "the current weather in a place"),
    ("translate", "text", "translate text to another language"),
    ("send_email", "message", "send an email"),
    ("set_timer", "duration", "set a timer"),
    ("play_music", "track", "play a song"),
    ("get_news", "topic", "get recent news headlines"),
    ("stock_price", "ticker", "look up a stock price"),
    ("define", "word", "look up a word's definition"),
    ("spell_check", "text", "check spelling in a piece of text"),
    ("random_number", "range", "generate a random number"),
    ("create_event", "details", "add an event to a calendar"),
    ("read_file", "path", "read the contents of a file"),
    ("map_route", "from and to", "get directions between two places"),
    ("currency", "amount", "convert between currencies"),
]


def pick_spec(kind: str, rng: random.Random) -> tuple[str, str, str]:
    names, args, descs = KINDS[kind]
    return rng.choice(names), rng.choice(args), rng.choice(descs)


def declaration_for(kind: str, rng: random.Random) -> tuple[str, str]:
    """A tool list containing `kind` plus distractors. Returns (block, name)."""
    spec = pick_spec(kind, rng)
    specs = [spec]
    # Other real kinds are better distractors than invented ones, since the
    # model has to tell apart tools it genuinely knows how to use.
    others = [k for k in KINDS if k != kind]
    rng.shuffle(others)
    for k in others[: rng.randint(0, 2)]:
        specs.append(pick_spec(k, rng))
    for d in rng.sample(DISTRACTORS, rng.randint(0, 3)):
        specs.append(d)
    rng.shuffle(specs)
    return render(specs), spec[0]


def idle_declaration(rng: random.Random) -> str:
    """A tool list for a conversation that needs no tool at all."""
    specs = [pick_spec(k, rng) for k in rng.sample(list(KINDS), rng.randint(0, 2))]
    specs += rng.sample(DISTRACTORS, rng.randint(1, 3))
    rng.shuffle(specs)
    return render(specs)


# --------------------------------------------------------------------------- #
# training conversations
# --------------------------------------------------------------------------- #
# Turns are (role, text): "tools" declares the list and leads the conversation,
# "call" is the model's call without its markers, "result" is the tool output.

_QWORDS = {"what", "how", "when", "where", "who", "whom", "which", "why", "is", "are", "was",
           "were", "does", "do", "did", "can", "could", "should", "will", "would", "has", "have"}


def _clean_query(q: str) -> str:
    return re.sub(r"^[^A-Za-z0-9]+", "", q.strip()).rstrip(" ?.")


def msmarco_conversations(n: int, rng: random.Random, split: str = "validation") -> list:
    """Search, read, answer: real queries, real web passages, human answers.

    The validation split is 100k rows, plenty for this and a fraction of the
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
        question = query[0].upper() + query[1:] + ("?" if query.split()[0].lower() in _QWORDS else "")
        block, name = declaration_for("search", rng)
        out.append([("tools", block), ("user", question), ("call", f"{name}: {query.lower()}"),
                    ("result", results), ("assistant", answer)])
    return out


def calc_conversations(n: int, rng: random.Random) -> list:
    """Arithmetic handed to a calculator, then stated plainly."""
    def num(lo, hi):
        return rng.randint(lo, hi)

    def money(x):
        return f"${x:,.2f}" if isinstance(x, float) and not float(x).is_integer() else f"${int(x):,}"

    def mul():
        a, b = num(12, 9999), num(3, 999)
        q = rng.choice([f"What is {a} times {b}?", f"Calculate {a} * {b}.", f"what's {a} x {b}",
                        f"Multiply {a:,} by {b}."])
        return q, f"{a} * {b}", f"{a:,} times {b:,} is {int(float(calc(f'{a} * {b}'))):,}."

    def add():
        xs = [num(10, 99999) for _ in range(rng.randint(2, 4))]
        expr = " + ".join(map(str, xs))
        q = rng.choice([f"What is {expr}?", f"Add up {', '.join(map(str, xs))}.",
                        f"what's the sum of {' and '.join(map(str, xs))}"])
        return q, expr, f"The sum is {int(float(calc(expr))):,}."

    def div():
        a, b = num(100, 99999), num(2, 97)
        q = rng.choice([f"What is {a} divided by {b}?", f"Calculate {a} / {b}.",
                        f"how much is {a} split {b} ways"])
        return q, f"{a} / {b}", f"{a:,} divided by {b} is {calc(f'{a} / {b}')}."

    def pct():
        p, x = rng.choice([5, 10, 12, 15, 18, 20, 25, 30, 35, 40, 75]), num(20, 9999)
        expr = f"{p} / 100 * {x}"
        r = calc(expr)
        if rng.random() < 0.4:
            return (f"What's a {p}% tip on a ${x} bill?", expr,
                    f"A {p}% tip on ${x} is {money(float(r))}.")
        q = rng.choice([f"What is {p}% of {x}?", f"Calculate {p} percent of {x}."])
        return q, expr, f"{p}% of {x:,} is {r}."

    def sqrt():
        x = num(2, 99999)
        q = rng.choice([f"What is the square root of {x}?", f"sqrt of {x}?",
                        f"Find the square root of {x}."])
        return q, f"sqrt({x})", f"The square root of {x:,} is about {calc(f'sqrt({x})')}."

    def shop():
        k, price = num(2, 40), rng.choice([num(1, 99) + 0.99, float(num(2, 300)), num(1, 50) + 0.5])
        item = rng.choice(["notebooks", "tickets", "shirts", "plants", "books", "mugs"])
        expr = f"{k} * {price}"
        return (f"I'm buying {k} {item} at ${price:.2f} each. How much is that in total?", expr,
                f"{k} {item} at ${price:.2f} each comes to {money(float(calc(expr)))}.")

    def speed():
        d, h = num(20, 900), rng.choice([1.5, 2, 2.5, 3, 4, 5, 6, 7.5])
        return (f"If a car travels {d} miles in {h} hours, what is its average speed?",
                f"{d} / {h}", f"Its average speed is {calc(f'{d} / {h}')} miles per hour.")

    def split():
        total, people = num(30, 2000), num(2, 12)
        expr = f"{total} / {people}"
        return (f"We spent ${total} on dinner and there are {people} of us. "
                f"How much does each person pay?", expr,
                f"Each person pays {money(round(float(calc(expr)), 2))}.")

    makers = [mul, add, div, pct, sqrt, shop, speed, split]
    out = []
    for _ in range(n):
        q, expr, answer = rng.choice(makers)()
        block, name = declaration_for("calc", rng)
        out.append([("tools", block), ("user", q), ("call", f"{name}: {expr}"),
                    ("result", calc(expr)), ("assistant", answer)])
    return out


def convert_conversations(n: int, rng: random.Random) -> list:
    """Unit conversions, including a tool the model only sees declared here."""
    pairs = [("km", "miles", "{v} km"), ("miles", "km", "{v} miles"), ("kg", "lbs", "{v} kg"),
             ("lbs", "kg", "{v} pounds"), ("cm", "in", "{v} cm"), ("in", "cm", "{v} inches"),
             ("m", "ft", "{v} metres"), ("ft", "m", "{v} feet"), ("l", "gallon", "{v} litres"),
             ("gallon", "l", "{v} gallons"), ("c", "f", "{v} degrees Celsius"),
             ("f", "c", "{v} degrees Fahrenheit")]
    pretty = {"c": "degrees Celsius", "f": "degrees Fahrenheit", "lbs": "pounds",
              "in": "inches", "ft": "feet", "l": "litres", "m": "metres"}
    out = []
    for _ in range(n):
        src, dst, phrase = rng.choice(pairs)
        v = rng.choice([rng.randint(1, 500), round(rng.uniform(0.5, 99), 1)])
        arg = f"{v} {src} to {dst}"
        result = convert(arg)
        q = rng.choice([f"How many {pretty.get(dst, dst)} is {phrase.format(v=v)}?",
                        f"Convert {phrase.format(v=v)} to {pretty.get(dst, dst)}.",
                        f"what's {phrase.format(v=v)} in {pretty.get(dst, dst)}"])
        block, name = declaration_for("convert", rng)
        out.append([("tools", block), ("user", q), ("call", f"{name}: {arg}"),
                    ("result", result),
                    ("assistant", f"{phrase.format(v=v)} is {result} {pretty.get(dst, dst)}.")])
    return out


def time_conversations(n: int, rng: random.Random) -> list:
    """Asking a clock, so the answer has to come from the result not the weights."""
    zones = [("Asia/Tokyo", "Tokyo"), ("Europe/London", "London"), ("America/New_York", "New York"),
             ("Europe/Paris", "Paris"), ("Australia/Sydney", "Sydney"), ("Asia/Kolkata", "India"),
             ("America/Los_Angeles", "Los Angeles"), ("Asia/Shanghai", "Shanghai"),
             ("Europe/Berlin", "Berlin"), ("America/Chicago", "Chicago"),
             ("Africa/Cairo", "Cairo"), ("America/Sao_Paulo", "Sao Paulo")]
    out = []
    for _ in range(n):
        zone, city = rng.choice(zones)
        # A plausible past timestamp: the point is reading it back, not the date.
        when = datetime.datetime(rng.randint(2024, 2026), rng.randint(1, 12), rng.randint(1, 28),
                                 rng.randint(0, 23), rng.choice([0, 5, 15, 20, 30, 40, 45, 55]))
        result = when.strftime("%Y-%m-%d %H:%M")
        q = rng.choice([f"What time is it in {city}?", f"what's the current time in {city}",
                        f"Do you know the time in {city} right now?"])
        block, name = declaration_for("time", rng)
        out.append([("tools", block), ("user", q), ("call", f"{name}: {zone}"),
                    ("result", result),
                    ("assistant", f"It's currently {when.strftime('%H:%M')} in {city} "
                                  f"({when.strftime('%B %d, %Y')}).")])
    return out


SPECIAL = {
    "@msmarco": msmarco_conversations,
    "@calc": calc_conversations,
    "@convert": convert_conversations,
    "@time": time_conversations,
}
