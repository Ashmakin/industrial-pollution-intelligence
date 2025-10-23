import io
from pathlib import Path
import tokenize

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {".git", "node_modules", "__pycache__", "build", "dist", ".ruff_cache", ".mypy_cache"}

PYTHON_EXTS = {".py"}
C_STYLE_EXTS = {
    ".rs",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".svelte",
    ".c",
    ".h",
    ".hpp",
    ".cc",
    ".cpp",
    ".mjs",
    ".css",
    ".scss",
    ".json",
    ".jsonc",
}
SQL_EXTS = {".sql"}
HASH_COMMENT_EXTS = {
    ".sh",
    ".bash",
    ".zsh",
    ".env",
    ".env.example",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".dockerfile",
}
SPECIAL_FILENAMES = {"Dockerfile"}


def iter_files():
    for path in PROJECT_ROOT.rglob("*"):
        if path.is_dir():
            continue
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def remove_python_comments(text: str) -> str:
    reader = io.StringIO(text)
    output_tokens = []
    try:
        tokens = list(tokenize.generate_tokens(reader.readline))
    except tokenize.TokenError:
        return text
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            continue
        output_tokens.append(tok)
    try:
        return tokenize.untokenize(output_tokens)
    except (tokenize.TokenError, ValueError):
        return text


def remove_hash_comments(text: str) -> str:
    lines = text.splitlines()
    new_lines = []
    for line in lines:
        stripped = remove_hash_comment_from_line(line)
        new_lines.append(stripped)
    if text.endswith("\n"):
        return "\n".join(new_lines) + "\n"
    return "\n".join(new_lines)


def remove_hash_comment_from_line(line: str) -> str:
    in_single = False
    in_double = False
    escape = False
    for idx, ch in enumerate(line):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if ch == "#" and not in_single and not in_double:
            return line[:idx].rstrip()
    return line.rstrip()


def remove_c_style_comments(text: str, line_marker: str = "//", block_start: str = "/*", block_end: str = "*/") -> str:
    result = []
    i = 0
    length = len(text)
    state = None                                       
    while i < length:
        ch = text[i]
        if state is None:
            if line_marker and text.startswith(line_marker, i):
                newline_idx = text.find("\n", i)
                if newline_idx == -1:
                    break
                i = newline_idx
                continue
            if block_start and text.startswith(block_start, i):
                end_idx = text.find(block_end, i + len(block_start))
                if end_idx == -1:
                    break
                comment_body = text[i + len(block_start):end_idx]
                newline_count = comment_body.count("\n")
                for _ in range(newline_count):
                    result.append("\n")
                i = end_idx + len(block_end)
                continue
            if ch == '"':
                hash_count = 0
                j = i - 1
                while j >= 0 and text[j] == '#':
                    hash_count += 1
                    j -= 1
                is_raw = False
                if j >= 0 and text[j] in {'r', 'R'}:
                    is_raw = True
                elif j >= 1 and text[j] in {'r', 'R'} and text[j - 1] in {'b', 'B'}:
                    is_raw = True
                if is_raw:
                    state = ('raw', hash_count)
                else:
                    state = '"'
                result.append(ch)
                i += 1
                continue
            if ch == "'":
                state = "'"
                result.append(ch)
                i += 1
                continue
            if ch == "`":
                state = "`"
                result.append(ch)
                i += 1
                continue
            result.append(ch)
            i += 1
            continue
        else:
            if state == '"':
                result.append(ch)
                if ch == '"' and not _is_escaped(text, i):
                    state = None
                i += 1
                continue
            if state == "'":
                result.append(ch)
                if ch == "'" and not _is_escaped(text, i):
                    state = None
                i += 1
                continue
            if state == "`":
                result.append(ch)
                if ch == "`" and not _is_escaped(text, i):
                    state = None
                i += 1
                continue
            if isinstance(state, tuple) and state[0] == 'raw':
                result.append(ch)
                if ch == '"':
                    hashes = '#' * state[1]
                    if text.startswith(hashes, i + 1):
                        for _ in hashes:
                            i += 1
                            if i < length:
                                result.append(text[i])
                        state = None
                        i += 1
                        continue
                i += 1
                continue
    return ''.join(result)


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    i = index - 1
    while i >= 0 and text[i] == '\\':
        backslashes += 1
        i -= 1
    return backslashes % 2 == 1


def process_file(path: Path) -> None:
    ext = path.suffix.lower()
    lower_name = path.name.lower()
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="utf-8", errors="ignore")
    original = content
    if path.name in SPECIAL_FILENAMES or lower_name == "makefile" or lower_name.endswith(".env") or lower_name.endswith(".env.example"):
        content = remove_hash_comments(content)
    elif ext in PYTHON_EXTS:
        content = remove_python_comments(content)
    elif ext in C_STYLE_EXTS:
        content = remove_c_style_comments(content)
    elif ext in SQL_EXTS:
        content = remove_c_style_comments(content, line_marker="--", block_start="/*", block_end="*/")
    elif ext in HASH_COMMENT_EXTS:
        content = remove_hash_comments(content)
    if content != original:
        path.write_text(content, encoding="utf-8")


def main():
    for file_path in iter_files():
        process_file(file_path)


if __name__ == "__main__":
    main()
