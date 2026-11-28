"""Мини-язык формул -> XML Mathcad 14 (.xmcd) + вычислитель тех же формул для сверки.

Строки документа:
  "## текст"            - заголовок раздела
  "# текст"             - текст
  "x := expr | подпись" - определение с показом значения (x := expr = ...)
  "x := expr ;"         - определение без показа (начальные приближения)
  "f(a,b) := expr"      - функция
  "f(T) := { e1 if c1 ; e2 if c2 }" - кусочная функция (program/ifThen)
Имя "T_к.1" -> база T, нижний индекс "к.1".
"""
import base64
import gzip
import math
import struct
import re
from xml.sax.saxutils import escape

from scipy.optimize import brentq

ML = "http://schemas.mathsoft.com/math30"
LETTER = "A-Za-zА-Яа-яЁёα-ωΑ-Ωϕ"
TOKEN = re.compile(
    rf"\s*(?:(?P<num>\d+\.?\d*(?:[eE][-+]?\d+)?)|(?P<id>[{LETTER}][{LETTER}0-9']*(?:_[{LETTER}0-9'.]+)?)"
    r"|(?P<op>:=|>=|<=|[-+*/^(),<>{};|\[\]]))"
)
BUILTIN = {"sin", "cos", "tan", "asin", "acos", "atan", "atan2", "ln", "exp", "max", "min", "if", "root", "sqrt"}


def tokens(s):
    pos, out = 0, []
    s = s.rstrip()
    while pos < len(s):
        m = TOKEN.match(s, pos)
        if not m or m.end() == pos:
            raise SyntaxError(f"не разобрано: {s[pos:]!r} в {s!r}")
        pos = m.end()
        k = m.lastgroup
        out.append((k, m.group(k)))
    return out


class P:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self, v=None):
        tok = self.peek()
        if v is not None and tok[1] != v:
            raise SyntaxError(f"ожидалось {v}, получено {tok}")
        self.i += 1
        return tok

    def expr(self):  # сравнение
        a = self.add()
        if self.peek()[1] in ("<", ">", ">=", "<="):
            op = self.take()[1]
            return ("cmp", op, a, self.add())
        return a

    def add(self):
        a = self.mul()
        while self.peek()[1] in ("+", "-"):
            op = self.take()[1]
            a = ("op", op, a, self.mul())
        return a

    def mul(self):
        a = self.unary()
        while self.peek()[1] in ("*", "/"):
            op = self.take()[1]
            a = ("op", op, a, self.unary())
        return a

    def unary(self):
        if self.peek()[1] == "-":
            self.take()
            return ("neg", self.unary())
        return self.pow()

    def pow(self):
        a = self.atom()
        if self.peek()[1] == "^":
            self.take()
            if self.peek()[1] == "-" and self.t[self.i + 1][0] == "num":  # 10^-5
                self.take()
                return ("op", "^", a, ("num", "-" + self.take()[1]))
            return ("op", "^", a, self.unary())
        return a

    def atom(self):
        k, v = self.take()
        if k == "num":
            return ("num", v)
        if v == "[":  # вектор-столбец [a; b; c]
            els = [self.expr()]
            while self.peek()[1] == ";":
                self.take()
                els.append(self.expr())
            self.take("]")
            return ("vec", els)
        if v == "(":
            e = self.expr()
            self.take(")")
            return ("paren", e)
        if k == "id":
            if self.peek()[1] == "(":
                self.take()
                args = [self.expr()]
                while self.peek()[1] == ",":
                    self.take()
                    args.append(self.expr())
                self.take(")")
                return ("call", v, args)
            return ("id", v)
        raise SyntaxError(f"неожиданно {v!r}")


def parse(s):
    p = P(tokens(s))
    e = p.expr()
    if p.i != len(p.t):
        raise SyntaxError(f"лишнее в {s!r}: {p.t[p.i:]}")
    return clean(e)


def _sum(e):
    return e[0] == "op" and e[1] in "+-"


def clean(e, keep=False):
    # оставляем скобки только там, где без них меняется смысл (дроби и показатели степени Mathcad рисует сам)
    t = e[0]
    if t == "paren":
        inner = clean(e[1])
        return ("paren", inner) if keep and inner[0] in ("op", "neg") else inner
    if t == "neg":
        return ("neg", clean(e[1], keep=_sum(_un(e[1]))))
    if t in ("op", "cmp"):
        op, a, b = e[1], _un(e[2]), _un(e[3])
        ka = (op == "*" and _sum(a)) or (op == "^" and a[0] in ("op", "neg"))
        kb = (op == "*" and _sum(b)) or (op == "-" and _sum(b))
        return (t, op, clean(e[2], ka), clean(e[3], kb))
    if t == "call":
        return ("call", e[1], [clean(a) for a in e[2]])
    if t == "vec":
        return ("vec", [clean(a) for a in e[1]])
    return e


def _un(e):
    while e[0] == "paren":
        e = e[1]
    return e


# ---------------- XML ----------------
def xid(name):
    base, _, sub = name.partition("_")
    a = f' subscript="{escape(sub)}"' if sub else ""
    return f'<ml:id xml:space="preserve"{a}>{escape(base)}</ml:id>'


def xnum(v):
    if "e" in v.lower():
        m, ex = re.split("[eE]", v)
        return f"<ml:apply><ml:mult/><ml:real>{m}</ml:real><ml:apply><ml:pow/><ml:real>10</ml:real><ml:real>{int(ex)}</ml:real></ml:apply></ml:apply>"
    return f"<ml:real>{v}</ml:real>"


OPX = {"+": "plus", "-": "minus", "*": "mult", "/": "div", "^": "pow", "<": "lessThan", ">": "greaterThan", ">=": "greaterOrEqual", "<=": "lessOrEqual"}


def x(e):
    t = e[0]
    if t == "num":
        return xnum(e[1])
    if t == "id":
        return xid(e[1])
    if t == "paren":
        return f"<ml:parens>{x(e[1])}</ml:parens>"
    if t == "neg":
        return f"<ml:apply><ml:neg/>{x(e[1])}</ml:apply>"
    if t in ("op", "cmp"):
        return f"<ml:apply><ml:{OPX[e[1]]}/>{x(e[2])}{x(e[3])}</ml:apply>"
    if t == "call":
        if e[1] == "sqrt":
            return f"<ml:apply><ml:sqrt/>{x(e[2][0])}</ml:apply>"
        args = "".join(x(a) for a in e[2])
        if len(e[2]) > 1:
            args = f"<ml:sequence>{args}</ml:sequence>"
        return f"<ml:apply>{xid(e[1])}{args}</ml:apply>"
    if t == "vec":
        return f'<ml:matrix rows="{len(e[1])}" cols="1">' + "".join(x(a) for a in e[1]) + "</ml:matrix>"
    raise ValueError(e)


# размеры для раскладки (Mathcad пересчитывает их сам при отображении)
def h(e):
    t = e[0]
    if t == "vec":
        return sum(h(a) + 2 for a in e[1]) + 8
    if t == "op" and e[1] == "/":
        return h(e[2]) + h(e[3]) + 5
    if t == "op" and e[1] == "^":
        return h(e[2]) + 0.55 * h(e[3])
    if t in ("op", "cmp"):
        return max(h(e[2]), h(e[3]))
    if t in ("paren", "neg"):
        return h(e[1]) + 1
    if t == "call":
        return max(h(a) for a in e[2]) + (4 if e[1] == "sqrt" else 2)
    return 14


def asc(e):  # высота над базовой линией
    t = e[0]
    if t == "vec":
        return h(e) / 2
    if t == "op" and e[1] == "/":
        return h(e[2]) + 3
    if t == "op" and e[1] == "^":
        return asc(e[2]) + 0.55 * h(e[3])
    if t in ("op", "cmp"):
        return max(asc(e[2]), asc(e[3]))
    if t in ("paren", "neg"):
        return asc(e[1]) + 1
    if t == "call":
        return max(asc(a) for a in e[2]) + 2
    return 10


def names(e):
    if e[0] == "piece":
        return set().union(*(names(c) | names(v) for c, v in e[1]))
    out = {e[1]} if e[0] in ("id", "call") else set()
    for c in e[1:]:
        if isinstance(c, tuple):
            out |= names(c)
        elif isinstance(c, list):
            for a in c:
                out |= names(a)
    return out


def has_ref(e):
    if e[0] in ("id", "call"):
        return True
    return any(has_ref(c) for c in e[1:] if isinstance(c, tuple))


def w(e):
    t = e[0]
    if t == "vec":
        return max(w(a) for a in e[1]) + 12
    if t in ("num", "id"):
        return 7 * len(e[1].replace("_", ""))
    if t == "op" and e[1] == "/":
        return max(w(e[2]), w(e[3])) + 4
    if t == "op" and e[1] == "^":
        return w(e[2]) + 0.7 * w(e[3])
    if t in ("op", "cmp"):
        return w(e[2]) + w(e[3]) + 12
    if t in ("paren", "neg"):
        return w(e[1]) + 10
    if t == "call":
        return 7 * len(e[1]) + sum(w(a) for a in e[2]) + 12
    return 10


# ---------------- вычислитель ----------------
class Env(dict):
    pass


def ev(e, env):
    t = e[0]
    if t == "vec":
        return [ev(c, env) for c in e[1]]
    if t == "num":
        return float(e[1])
    if t == "id":
        if e[1] == "π":
            return math.pi
        if e[1] == "NaN":
            return math.nan
        if e[1] == "deg":
            return math.pi / 180
        return env[e[1]]
    if t == "paren":
        return ev(e[1], env)
    if t == "neg":
        return -ev(e[1], env)
    if t == "op":
        a, b = ev(e[2], env), ev(e[3], env)
        return {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b, "/": lambda: a / b, "^": lambda: a**b}[e[1]]()
    if t == "cmp":
        a, b = ev(e[2], env), ev(e[3], env)
        return float({"<": a < b, ">": a > b, ">=": a >= b, "<=": a <= b}[e[1]])
    if t == "call":
        f = e[1]
        if f == "root":  # root(f(x), x): решение f = 0 с начальным приближением x
            var = e[2][1][1]
            x0 = env[var]

            def g(xv):
                env2 = Env(env)
                env2[var] = xv
                return ev(e[2][0], env2)

            lo, hi = x0 * 0.5, x0 * 1.5
            return brentq(g, lo, hi, xtol=1e-12, rtol=1e-14)
        if f == "if":
            return ev(e[2][1], env) if ev(e[2][0], env) else ev(e[2][2], env)
        args = [ev(a, env) for a in e[2]]
        if f == "atan2":  # Mathcad: atan2(x, y) - угол точки (x, y)
            return math.atan2(args[1], args[0])
        if f in ("max", "min"):
            return (max if f == "max" else min)(args)
        if f in BUILTIN:
            return getattr(math, {"ln": "log"}.get(f, f))(*args)
        params, body = env[f]
        env2 = Env(env)
        env2.update(zip(params, args))
        if body[0] == "piece":
            for cond, val in body[1]:
                if ev(cond, env2):
                    return ev(val, env2)
            raise ValueError(f"{f}: ни одно условие не выполнено")
        return ev(body, env2)
    raise ValueError(e)


# ---------------- документ ----------------
REG = ('<region region-id="{id}" left="{x:.1f}" top="{y:.1f}" width="{w:.1f}" height="{h:.1f}" align-x="{x:.1f}" '
       'align-y="{ay:.1f}" show-border="false" show-highlight="false" is-protected="true" z-order="0" background-color="inherit" tag="">')
P_ATTR = 'style="Normal" margin-left="inherit" margin-right="inherit" text-indent="inherit" text-align="inherit" list-style-type="inherit" tabs="inherit"'

X_LABEL, X_MATH, W_LABEL = 12, 200, 180
X_NOTE, W_NOTE = 320, 235  # колонка источников


class Doc:
    def __init__(self):
        self.regions, self.y, self.rid, self.env, self.items = [], 10.0, 1, Env(), []

    def _region(self, x0, y0, wd, ht, inner, ay=None):
        ay = y0 + 10 if ay is None else ay  # align-y - базовая линия строки
        self.regions.append(REG.format(id=self.rid, x=x0, y=y0, w=wd, h=ht, ay=ay) + inner + "</region>")
        self.rid += 1

    def text(self, s, size=None, bold=False, x0=X_LABEL, width=None, advance=True, y0=None, italic=False):
        body = escape(s)
        if bold:
            body = f"<b>{body}</b>"
        if italic:
            body = f"<i>{body}</i>"
        if size:
            body = f'<f size="{size}">{body}</f>'
        width = width or max(60, min(560, 6.2 * len(s) * (size / 10 if size else 1)))
        lines = math.ceil(6.2 * len(s) / width) if width else 1
        ht = 13 * lines * (size / 10 if size else 1)
        y0 = self.y if y0 is None else y0
        self._region(x0, y0, width, ht,
                     f'<text use-page-width="false" push-down="false" lock-width="true"><p {P_ATTR}>{body}</p></text>')
        if advance:
            self.y += ht + 8
        return ht

    def line(self, s):
        s = s.strip()
        if not s:
            self.y += 8
            return
        if s.startswith("## "):
            self.y += 14
            self.text(s[3:], size=12, bold=True, width=560)
            return
        if s.startswith("# "):
            self.text(s[2:])
            return
        label = note = None
        if " || " in s:  # третья колонка: рекомендуемый диапазон и источник
            s, note = (p.strip() for p in s.split(" || ", 1))
        if " | " in s:
            s, label = (p.strip() for p in s.split(" | ", 1))
        show = not s.endswith(";")
        s = s.rstrip(";").strip()
        lhs, rhs = (p.strip() for p in s.split(":=", 1))
        fm = re.fullmatch(rf"([{LETTER}][^\s(]*)\((.*)\)", lhs)
        if fm:  # функция
            name, params = fm.group(1), [p.strip() for p in fm.group(2).split(",")]
            if rhs.startswith("{"):
                pieces = []
                for part in rhs.strip("{} ").split(";"):
                    val, cond = part.rsplit(" if ", 1)
                    pieces.append((parse(cond), parse(val)))
                body = ("piece", pieces)
                xbody = "<ml:program>" + "".join(f"<ml:ifThen>{x(c)}{x(v)}</ml:ifThen>" for c, v in pieces) + "</ml:program>"
                ht = sum(max(h(v), h(c)) + 6 for c, v in pieces)
                a = ht / 2
                wd = max(w(v) + w(c) for c, v in pieces) + 60
            else:
                body = parse(rhs)
                xbody, ht, wd = x(body), h(body), w(body)
            a = asc(body)
            # Mathcad берёт глобальные имена функции в месте её определения - они должны быть уже заданы
            free = {n for n in names(body) if n not in params and n not in ("π", "deg") and n not in BUILTIN}
            missing = sorted(n for n in free if n not in self.env)
            assert not missing, f"{name}: ещё не определены {missing}"
            self.env[name] = (params, body)
            xm = (f'<ml:define xmlns:ml="{ML}"><ml:function>{xid(name)}<ml:boundVars>'
                  + "".join(xid(p) for p in params) + f"</ml:boundVars></ml:function>{xbody}</ml:define>")
            wd += 7 * len(lhs) + 20
        else:
            e = parse(rhs)
            self.env[lhs] = ev(e, self.env)
            show = show and has_ref(e)  # заданные числа - без "= значение"
            a = asc(e)
            inner = f"<ml:eval>{x(e)}</ml:eval>" if show else x(e)
            xm = f'<ml:define xmlns:ml="{ML}">{xid(lhs)}{inner}</ml:define>'
            ht, wd = h(e), w(e) + 7 * len(lhs) + (90 if show else 20)
        y0 = self.y
        ly = y0 + max(0, a - 10)  # подпись - на базовой линии формулы
        lh = self.text(label, x0=X_LABEL, width=W_LABEL, advance=False, y0=ly) if label else 0
        self._region(X_MATH, y0, wd, ht, f'<math optimize="false" disable-calc="false">{xm}</math>', ay=y0 + a)
        nh = 0
        if note and X_MATH + wd + 12 > X_NOTE:  # длинная формула - источник строкой ниже неё
            nh = self.text(note, x0=X_MATH, width=X_NOTE + W_NOTE - X_MATH, advance=False, y0=y0 + ht + 4, italic=True)
            nh += ht + 4 - (ly - y0)
        elif note:
            nh = self.text(note, x0=X_NOTE, width=W_NOTE, advance=False, y0=ly, italic=True)
        self.y = max(y0 + ht, ly + lh, ly + nh) + 10

    def lines(self, block):
        for s in block.strip("\n").split("\n"):
            self.line(s)

    def plot(self, item_b64, nw, nh):
        # график Mathcad: сам график хранится в двоичном item (шаблон снят с графика, построенного в Mathcad).
        # Размер поля графика - два int32 перед блоком "axisFormat", в единицах по 6 pt (найдено сравнением файлов)
        raw = bytearray(gzip.decompress(base64.b64decode(item_b64)))
        a = raw.index(b"\naxisFormat")
        assert raw[a - 6 : a] == b"\x01\x00\x01\x00\x00\x00", "формат графика Mathcad не узнан"
        struct.pack_into("<2i", raw, a - 14, nw, nh)
        self.items.append(base64.b64encode(gzip.compress(bytes(raw))).decode())
        wd, ht = 6 * nw + 90, 6 * nh + 60  # с подписями осей и легендой
        iid = len(self.items) + 1
        self._region(X_LABEL, self.y, wd, ht, f'<plot disable-calc="false" item-idref="{iid}"/>', ay=self.y)
        self.y += ht + 14

    def save(self, template, path):
        src = open(template, encoding="utf-8").read()
        head = src[: src.index("<regions>")].rstrip("\t")
        head = re.sub(r"<\?validation-md5-digest[^>]*\?>\s*", "", head)
        head = head.replace('automatic-recalculation="false"', 'automatic-recalculation="true"')
        # шаблон оформления графиков из настроек документа -> item 1
        tid = re.search(r'<plotTemplate>\s*<xy item-idref="(\d+)"', head).group(1)
        head = re.sub(r'(<plotTemplate>\s*<xy item-idref=")\d+', r"\g<1>1", head)
        templ = re.search(rf'<item item-id="{tid}"([^>]*)>(.*?)</item>', src, re.S)
        items = [f'<item item-id="1"{templ.group(1)}>{templ.group(2)}</item>']
        items += [f'<item item-id="{i}" content-encoding="gzip">{b}</item>' for i, b in enumerate(self.items, 2)]
        out = (head + "\t<regions>\n" + "".join(f"\t\t{r}\n" for r in self.regions) + "\t</regions>\n\t<binaryContent>\n"
               + "".join(f"\t\t{it}\n" for it in items) + "\t</binaryContent>\n</worksheet>\n")
        open(path, "w", encoding="utf-8", newline="\n").write(out)

