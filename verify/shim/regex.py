"""
regex shim -- drop-in replacement for the `regex` module.

Python's standard `re` module does NOT support:
  - Unicode property escapes:  \p{L}  \p{N}  \P{L}  \P{N}
  - Possessive quantifiers:    ?+  *+  ++  {m,n}+

This shim preprocesses patterns translating these features into
standard-re equivalents.

Key design decisions:
  - \p{L} outside [...]  ->  [^\W\d_]  (Unicode word chars minus digits/underscore)
  - \p{L} inside  [...]  ->  hardcoded Unicode letter ranges (common scripts)
  - \p{N} outside [...]  ->  \d
  - \p{N} inside  [...]  ->  \d  (re.UNICODE makes \d match all Unicode digits)
  - Possessive quantifiers -> greedy equivalents (safe for BPE splitting)
"""

import re as _re
import sys

# ---- Unicode letter ranges (inside-character-class fallback) -----------------
# Covers ASCII, Latin Extended, Cyrillic, CJK, Hangul, Kana.
# Generated subset of Unicode General Category L.

_UNICODE_LETTER_CHARS = (
    r"a-zA-Z"
    r"\u00AA\u00B5\u00BA"                               # Latin-1 letters outside ranges
    r"\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u024F"          # Latin-1 + Extended-A/B
    r"\u0250-\u02AF"                                      # IPA Extensions
    r"\u02B0-\u02FF"                                      # Spacing Modifier Letters
    r"\u0370-\u03FF"                                      # Greek & Coptic
    r"\u0400-\u04FF"                                      # Cyrillic
    r"\u0500-\u052F\u0531-\u0556\u0560-\u0588"           # Cyrillic Suppl + Armenian
    r"\u0590-\u05FF"                                      # Hebrew
    r"\u0600-\u06FF\u0750-\u077F"                         # Arabic + Supplement
    r"\u0900-\u097F"                                      # Devanagari
    r"\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF"  # Bengali etc.
    r"\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0D80-\u0DFF"  # Telugu etc.
    r"\u0E00-\u0E7F"                                      # Thai
    r"\u0E80-\u0EFF"                                      # Lao
    r"\u0F00-\u0FFF"                                      # Tibetan
    r"\u10A0-\u10FF"                                      # Georgian
    r"\u1100-\u11FF"                                      # Hangul Jamo
    r"\u1780-\u17FF\u1800-\u18AF"                         # Khmer + Mongolian
    r"\u1E00-\u1EFF\u1F00-\u1FFF"                         # Latin Extended Addl + Greek Ext
    r"\u2000-\u206F"                                      # General Punctuation (some letters)
    r"\u2C00-\u2C5F\u2C60-\u2C7F"                         # Glagolitic + Latin Ext-C
    r"\u2D00-\u2D2F"                                      # Georgian Supplement
    r"\u2D30-\u2D7F"                                      # Tifinagh
    r"\u3005-\u3007"                                      # CJK Symbols (letter-like)
    r"\u3040-\u309F"                                      # Hiragana
    r"\u30A0-\u30FF"                                      # Katakana
    r"\u3100-\u312F"                                      # Bopomofo
    r"\u3130-\u318F"                                      # Hangul Compatibility Jamo
    r"\u31A0-\u31BF"                                      # Bopomofo Extended
    r"\u31F0-\u31FF"                                      # Katakana Phonetic
    r"\u3400-\u4DBF"                                      # CJK Extension A
    r"\u4DC0-\u4DFF\u4E00-\u9FFF"                         # Yijing + CJK Unified
    r"\uA000-\uA4CF\uA4D0-\uA4FF"                         # Yi + Lisu
    r"\uA500-\uA63F"                                      # Vai
    r"\uA640-\uA69F\uA6A0-\uA6FF"                         # Cyrillic Ext-B + Bamum
    r"\uA700-\uA71F"                                      # Modifier Tone Letters
    r"\uA840-\uA87F"                                      # Phags-pa
    r"\uAC00-\uD7AF"                                      # Hangul Syllables
    r"\uF900-\uFAFF\uFB00-\uFB4F\uFB50-\uFDFF"           # CJK Compat + Latin Ligatures + Arabic Pres
    r"\uFE70-\uFEFF"                                      # Arabic Pres Forms-B
    r"\uFF21-\uFF3A\uFF41-\uFF5A\uFF66-\uFFDC"           # Fullwidth Latin + Halfwidth Katakana
)

# ---- Translation -------------------------------------------------------------

def _translate_standalone(pattern):
    """Replace standalone \\p{Prop} / \\P{Prop} patterns (outside char classes)."""
    result = []
    i = 0
    n = len(pattern)
    while i < n:
        if i + 2 < n and pattern[i] == "\\":
            ch = pattern[i + 1]
            if ch in ("p", "P") and i + 3 < n and pattern[i + 2] == "{":
                negate = ch == "P"
                j = i + 3
                while j < n and pattern[j] != "}":
                    j += 1
                if j < n:
                    prop = pattern[i + 3: j]
                    if prop == "N":
                        result.append(r"\D" if negate else r"\d")
                    elif prop == "L":
                        result.append(r"[\W\d_]" if negate else r"[^\W\d_]")
                    elif prop == "Z":
                        result.append(r"\S" if negate else r"\s")
                    else:
                        result.append(".")
                    i = j + 1
                    continue
        result.append(pattern[i])
        i += 1
    return "".join(result)


def _translate_charclass(pattern):
    """Replace \\p{Prop} inside [...] character classes."""
    result = []
    i = 0
    n = len(pattern)
    while i < n:
        if pattern[i] == "[":
            # Find the matching ']'
            j = i + 1
            depth = 1
            while j < n and depth > 0:
                if pattern[j] == "\\":
                    j += 1  # skip escaped char
                elif pattern[j] == "[":
                    depth += 1
                elif pattern[j] == "]":
                    depth -= 1
                j += 1
            inner = pattern[i + 1: j - 1]

            # Replace \p{Prop} inside the character class
            inner_translated = []
            k = 0
            m = len(inner)
            while k < m:
                if k + 2 < m and inner[k] == "\\":
                    ch = inner[k + 1]
                    if ch in ("p", "P") and k + 3 < m and inner[k + 2] == "{":
                        negate = ch == "P"
                        l = k + 3
                        while l < m and inner[l] != "}":
                            l += 1
                        if l < m:
                            prop = inner[k + 3: l]
                            if prop == "N":
                                # \d works inside char class
                                inner_translated.append(r"\D" if negate else r"\d")
                            elif prop == "L":
                                # Use hardcoded Unicode letter ranges
                                if negate:
                                    inner_translated.append(r"\W")
                                else:
                                    inner_translated.append(_UNICODE_LETTER_CHARS)
                            else:
                                inner_translated.append(
                                    r"\D" if negate else r"\d"
                                )
                            k = l + 1
                            continue
                    inner_translated.append(inner[k: k + 2])
                    k += 2
                    continue
                inner_translated.append(inner[k])
                k += 1

            result.append("[" + "".join(inner_translated) + "]")
            i = j
            continue
        result.append(pattern[i])
        i += 1
    return "".join(result)


def _translate_possessive(pattern):
    """Replace possessive quantifiers with greedy equivalents."""
    return _re.sub(r"([?*+]|\{[0-9,]+?\})\+", r"\1", pattern)


def _translate_pattern(pattern):
    """Full translation pipeline."""
    # Step 1: Translate \p{} inside character classes first
    pattern = _translate_charclass(pattern)
    # Step 2: Translate standalone \p{} patterns
    pattern = _translate_standalone(pattern)
    # Step 3: Remove possessive quantifiers
    pattern = _translate_possessive(pattern)
    return pattern


# ---- Compiled pattern wrapper ------------------------------------------------

class Pattern:
    """Compiled regex pattern wrapping a standard re.Pattern."""

    __slots__ = ("_pattern", "_original", "_flags")

    def __init__(self, pattern, flags=0):
        self._original = pattern
        self._flags = flags
        self._pattern = _re.compile(_translate_pattern(pattern), flags)

    def search(self, *a, **kw):        return self._pattern.search(*a, **kw)
    def match(self, *a, **kw):         return self._pattern.match(*a, **kw)
    def fullmatch(self, *a, **kw):     return self._pattern.fullmatch(*a, **kw)
    def split(self, *a, **kw):         return self._pattern.split(*a, **kw)
    def findall(self, *a, **kw):       return self._pattern.findall(*a, **kw)
    def finditer(self, *a, **kw):      return self._pattern.finditer(*a, **kw)
    def sub(self, *a, **kw):           return self._pattern.sub(*a, **kw)
    def subn(self, *a, **kw):          return self._pattern.subn(*a, **kw)
    def __repr__(self):                return f"shim.Pattern({self._original!r})"

    @property
    def flags(self):       return self._pattern.flags
    @property
    def groups(self):      return self._pattern.groups
    @property
    def groupindex(self):  return self._pattern.groupindex
    @property
    def pattern(self):     return self._original


# ---- Module-level functions --------------------------------------------------

def compile(pattern, flags=0):
    return Pattern(pattern, flags)

def search(pattern, string, flags=0):
    return Pattern(pattern, flags).search(string)

def match(pattern, string, flags=0):
    return Pattern(pattern, flags).match(string)

def fullmatch(pattern, string, flags=0):
    return Pattern(pattern, flags).fullmatch(string)

def split(pattern, string, maxsplit=0, flags=0):
    return Pattern(pattern, flags).split(string, maxsplit=maxsplit)

def findall(pattern, string, flags=0):
    return Pattern(pattern, flags).findall(string)

def finditer(pattern, string, flags=0):
    return Pattern(pattern, flags).finditer(string)

def sub(pattern, repl, string, count=0, flags=0):
    return Pattern(pattern, flags).sub(repl, string, count=count)

def subn(pattern, repl, string, count=0, flags=0):
    return Pattern(pattern, flags).subn(repl, string, count=count)

def escape(pattern):
    return _re.escape(pattern)

def purge():
    _re.purge()


# ---- Constants ---------------------------------------------------------------

A = _re.A
ASCII = _re.ASCII
I = _re.I
IGNORECASE = _re.IGNORECASE
L = _re.L
LOCALE = _re.LOCALE
M = _re.M
MULTILINE = _re.MULTILINE
S = _re.S
DOTALL = _re.DOTALL
U = _re.U
UNICODE = _re.UNICODE
X = _re.X
VERBOSE = _re.VERBOSE

# regex-specific flags (stubs)
V0 = 0
V1 = 1
DEFAULT_VERSION = V1
FULLCASE = 0
REVERSE = 0
WORD = 0
POSIX = 0
NOSPECIAL = 0
BESTMATCH = 0
ENHANCEMATCH = 0
DEBUG = 0
TEMPLATE = 0

Match = _re.Match
Regex = type(_re.compile(""))
Scanner = _re.Scanner
error = _re.error
