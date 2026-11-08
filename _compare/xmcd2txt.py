#!/usr/bin/env python3
"""Convert a Mathcad 14 (worksheet30) .xmcd XML file to readable text."""
import re
import sys
import xml.etree.ElementTree as ET

NS = '{http://schemas.mathsoft.com/worksheet30}'
ML = '{http://schemas.mathsoft.com/math30}'


def ids(e):
    base = (e.text or '').strip()
    sub = e.get('subscript')
    if sub:
        return f"{base}_{sub}"
    return base


def render(e, prec=0):
    tag = e.tag
    if not tag.startswith(ML):
        return render_children(e, prec)
    t = tag[len(ML):]
    if t == 'math':
        return render_children(e, prec)
    if t == 'apply':
        return render_apply(e)
    if t in ('real', 'complex', 'integer'):
        return (e.text or '').strip()
    if t == 'id':
        return ids(e)
    if t == 'string':
        return '"' + (e.text or '') + '"'
    if t == 'define':
        ch = list(e)
        return f"{render(ch[0])} := {render(ch[1])}"
    if t == 'eval':
        ch = list(e)
        return ' = '.join(render(c) for c in ch)
    if t == 'parens':
        return '(' + render_children(e, prec) + ')'
    if t == 'sequence':
        return ','.join(render(c) for c in e)
    if t == 'matrix':
        rows = []
        for r in e:
            if r.tag.startswith(ML) and r.tag[len(ML):] == 'row':
                rows.append(' | '.join(render(c) for c in r))
            else:
                rows.append(render(r))
        return '[' + ' ; '.join(rows) + ']'
    if t == 'function':
        return render_function(e)
    if t == 'program':
        return render_program(e)
    if t == 'ifThen':
        ch = list(e)
        return f"if {render(ch[0])} -> {render(ch[1])}"
    if t == 'boundVars':
        return render_children(e)
    if t == 'indexer':
        ch = list(e)
        return render(ch[0]) + '[' + ','.join(render(c) for c in ch[1:]) + ']'
    return render_children(e)


BIN = {
    'plus': '+', 'minus': '-', 'mult': '*', 'div': '/',
    'lessThan': '<', 'greaterThan': '>', 'lessOrEqual': '<=',
    'greaterOrEqual': '>=', 'equal': '=', 'notEqual': '!=',
}


def render_children(e, prec=0):
    return ''.join(render(c, prec) for c in e)


def render_apply(e):
    ch = list(e)
    op = ch[0]
    optag = op.tag[len(ML):] if op.tag.startswith(ML) else op.tag
    args = ch[1:]
    if optag in ('plus', 'mult'):
        parts = [render(a, 2) for a in args]
        sep = ' + ' if optag == 'plus' else '*'
        return '(' + sep.join(parts) + ')'
    if optag == 'minus':
        if len(args) == 1:
            return '-' + render(args[0], 3)
        return f"({render(args[0], 3)} - {render(args[1], 3)})"
    if optag == 'neg':
        return '-' + render(args[0], 3)
    if optag == 'div':
        return '(' + render(args[0], 2) + ')/(' + render(args[1], 2) + ')'
    if optag == 'pow':
        return '(' + render(args[0], 3) + ')^(' + render(args[1], 3) + ')'
    if optag == 'sqrt':
        return 'sqrt(' + ','.join(render(a) for a in args) + ')'
    if optag in BIN:
        return '(' + (' ' + BIN[optag] + ' ').join(render(a, 2) for a in args) + ')'
    fname = render(op)
    return fname + '(' + ', '.join(render(a) for a in args) + ')'


def render_function(e):
    parts = []
    name = ''
    for c in e:
        t = c.tag[len(ML):] if c.tag.startswith(ML) else c.tag
        if t == 'id':
            name = ids(c)
        else:
            parts.append(render(c))
    return name + '(' + ', '.join(parts) + ')'


def render_program(e):
    lines = []
    for c in e:
        t = c.tag[len(ML):] if c.tag.startswith(ML) else c.tag
        if t == 'bodyparam':
            lines.append('  inputs: ' + render_children(c))
        elif t == 'expr':
            lines.append('  ' + render_children(c))
        else:
            lines.append('  ' + render(c))
    return '\n  PROGRAM:\n' + '\n'.join(lines)


def render_text(e):
    buf = []

    def rec(n):
        t = n.tag
        if t == NS + 'p':
            buf.append('\n')
        if t == NS + 'sp':
            buf.append(' ')
            return
        if n.text:
            buf.append(n.text)
        for c in n:
            rec(c)
        if n.tail:
            buf.append(n.tail)

    if e.text:
        buf.append(e.text)
    for c in e:
        rec(c)
    out = ''.join(buf)
    out = re.sub(r'\n{2,}', '\n', out)
    return out.strip()


def scan_any(e, out):
    tag = e.tag
    if tag == NS + 'math':
        s = ''.join(render(g) for g in e)
        out.append(('math', s.strip()))
        return
    if tag == NS + 'text':
        out.append(('text', render_text(e)))
        return
    if tag == NS + 'image':
        out.append(('image', '[image]'))
        return
    if tag == NS + 'plot':
        out.append(('plot', '[plot]'))
        return
    for c in e:
        scan_any(c, out)


def main(path, outpath):
    tree = ET.parse(path)
    root = tree.getroot()
    region_nodes = [e for e in root.iter() if e.tag == NS + 'region']
    items = []
    for e in region_nodes:
        scan_any(e, items)
    lines = []
    for kind, s in items:
        if kind == 'text':
            if s:
                lines.append('### ' + s.replace('\n', ' ⏎ '))
        elif kind == 'math':
            lines.append(s)
        else:
            lines.append(s)
    text = '\n'.join(lines)
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write(text)
    print('regions:', len(region_nodes), 'items:', len(items), 'chars:', len(text))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
