#!/usr/bin/env python3
"""String-aware Rust function/test remover. Deletes named items (with a
preceding #[test] attr if present) by brace-matching that ignores braces inside
strings/comments. Usage: rm_fns.py <file> name1 name2 ..."""
import sys

def find_body_close(s, open_idx):
    """Given index of the opening '{', return index just after matching '}'."""
    depth = 0
    i = open_idx
    n = len(s)
    while i < n:
        c = s[i]
        # line comment
        if c == '/' and i + 1 < n and s[i+1] == '/':
            j = s.find('\n', i)
            i = n if j == -1 else j
            continue
        # block comment
        if c == '/' and i + 1 < n and s[i+1] == '*':
            j = s.find('*/', i+2)
            i = n if j == -1 else j + 2
            continue
        # raw string r"..." or r#"..."#
        if c == 'r' and i + 1 < n and s[i+1] in '#"':
            k = i + 1
            hashes = 0
            while k < n and s[k] == '#':
                hashes += 1; k += 1
            if k < n and s[k] == '"':
                term = '"' + '#' * hashes
                j = s.find(term, k+1)
                i = n if j == -1 else j + len(term)
                continue
        # normal string
        if c == '"':
            i += 1
            while i < n:
                if s[i] == '\\':
                    i += 2; continue
                if s[i] == '"':
                    i += 1; break
                i += 1
            continue
        # char literal '...'
        if c == "'":
            # could be lifetime; only treat as char if it closes shortly
            j = i + 1
            if j < n and s[j] == '\\':
                j += 2
            else:
                j += 1
            if j < n and s[j] == "'":
                i = j + 1
                continue
            # lifetime, skip the quote
            i += 1
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise RuntimeError("unbalanced braces")

def remove_item(src, name):
    # find `fn NAME` (word-boundary)
    import re
    m = re.search(r'\bfn\s+' + re.escape(name) + r'\b', src)
    if not m:
        print(f"  WARN: fn {name} not found", file=sys.stderr)
        return src, False
    fn_start = m.start()
    # back up to include attributes/doc lines (#[...] and /// ) immediately above
    line_start = src.rfind('\n', 0, fn_start) + 1
    # walk backwards over attribute / doc-comment / blank lines belonging to the item
    start = line_start
    while True:
        prev_nl = src.rfind('\n', 0, start - 1)
        prev_line_start = prev_nl + 1
        line = src[prev_line_start:start-1].strip()
        if line.startswith('#[') or line.startswith('///') or line.startswith('//'):
            start = prev_line_start
            continue
        break
    # find opening brace of the fn body
    open_idx = src.find('{', fn_start)
    end = find_body_close(src, open_idx)
    # consume one trailing newline
    if end < len(src) and src[end] == '\n':
        end += 1
    # consume one extra blank line after, if present
    if end < len(src) and src[end] == '\n':
        end += 1
    return src[:start] + src[end:], True

def main():
    path = sys.argv[1]
    names = sys.argv[2:]
    src = open(path).read()
    for nm in names:
        src, ok = remove_item(src, nm)
        print(f"  {'removed' if ok else 'MISSING'}: {nm}")
    open(path, 'w').write(src)

main()
