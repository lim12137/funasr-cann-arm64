"""
tiktoken shim -- fallback for environments where the native tiktoken
package cannot be installed (missing Rust compiler, etc.).

This module:
  - Tries to use the real `tiktoken` package first.
  - Falls back to a character-level encoder that preserves basic
    encode/decode API compatibility.

FOR PRODUCTION USE: install the real tiktoken package:
    pip install tiktoken
"""

import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, Union

__all__ = [
    "Encoding",
    "get_encoding",
    "list_encoding_names",
    "encoding_for_model",
    "tokenizer_for_model",
]

_ENCODING_NAMES = {
    "cl100k_base",
    "o200k_base",
    "p50k_base",
    "p50k_edit",
    "r50k_base",
}

_MODEL_TO_ENCODING = {
    "gpt-4": "cl100k_base",
    "gpt-4-0314": "cl100k_base",
    "gpt-4-0613": "cl100k_base",
    "gpt-4-32k": "cl100k_base",
    "gpt-4-32k-0314": "cl100k_base",
    "gpt-4-32k-0613": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    "gpt-3.5-turbo-0301": "cl100k_base",
    "gpt-3.5-turbo-0613": "cl100k_base",
    "gpt-3.5-turbo-1106": "cl100k_base",
    "gpt-3.5-turbo-0125": "cl100k_base",
    "text-davinci-003": "p50k_base",
    "text-davinci-002": "p50k_base",
    "text-davinci-001": "r50k_base",
    "davinci": "r50k_base",
    "text-curie-001": "r50k_base",
    "curie": "r50k_base",
    "text-babbage-001": "r50k_base",
    "babbage": "r50k_base",
    "text-ada-001": "r50k_base",
    "ada": "r50k_base",
    "code-davinci-002": "p50k_base",
    "code-davinci-001": "p50k_base",
    "code-cushman-002": "p50k_base",
    "code-cushman-001": "p50k_base",
}

_registry: Dict[str, "Encoding"] = {}


class _FallbackEncoding:
    """Minimal character-level encoder used when real tiktoken is unavailable.

    Each grapheme cluster is its own token.  Decode concatenates characters.
    This is NOT a real BPE tokenizer -- it exists only so that FunASR model
    loading does not crash with ModuleNotFoundError.
    """

    name: str

    def __init__(self, name: str):
        self.name = name
        self._special_tokens: Dict[str, int] = {}
        self.eot_token = 0

    def encode(
        self,
        text: str,
        *,
        allowed_special: Union[str, Set[str]] = "all",
        disallowed_special: Union[str, Tuple[str, ...]] = (),
    ) -> List[int]:
        if not text:
            return []
        return [ord(ch) for ch in text]

    def encode_ordinary(self, text: str) -> List[int]:
        return self.encode(text, allowed_special=set(), disallowed_special=())

    def encode_ordinary_batch(
        self, texts: List[str], *, num_threads: int = 1
    ) -> List[List[int]]:
        return [self.encode_ordinary(t) for t in texts]

    def decode(self, tokens: List[int]) -> str:
        return "".join(chr(max(0, min(t, 0x10FFFF))) for t in tokens)

    def decode_single_token_bytes(self, token: int) -> bytes:
        return self.decode([token]).encode("utf-8", errors="replace")

    @property
    def n_vocab(self) -> int:
        return 0x110000

    @property
    def max_token_value(self) -> int:
        return 0x10FFFF

    @property
    def special_tokens_set(self) -> Set[str]:
        return set(self._special_tokens.keys())

    def is_special_token(self, token: int) -> bool:
        return token in self._special_tokens.values()


class Encoding:
    """Unified encoding: delegates to real tiktoken if available."""

    def __init__(self, name: str):
        self._real: Optional[Any] = None
        try:
            import tiktoken as _real_tiktoken

            self._real = _real_tiktoken.get_encoding(name)
        except Exception:
            pass

        if self._real is not None:
            warnings.warn(
                f"tiktoken shim: using real tiktoken for {name!r}",
                stacklevel=1,
            )
        else:
            self._fallback = _FallbackEncoding(name)
            warnings.warn(
                f"tiktoken shim: using character-level fallback for {name!r}. "
                f"Install `pip install tiktoken` for accurate BPE tokenization.",
                stacklevel=1,
            )

    @property
    def name(self):
        if self._real is not None:
            return self._real.name
        return self._fallback.name

    @property
    def n_vocab(self):
        if self._real is not None:
            return self._real.n_vocab
        return self._fallback.n_vocab

    @property
    def max_token_value(self):
        if self._real is not None:
            return self._real.max_token_value
        return self._fallback.max_token_value

    @property
    def eot_token(self):
        if self._real is not None:
            return self._real.eot_token
        return self._fallback.eot_token

    def encode(self, text, *, allowed_special="all", disallowed_special=()):
        if self._real is not None:
            return self._real.encode(
                text,
                allowed_special=allowed_special,
                disallowed_special=disallowed_special,
            )
        return self._fallback.encode(
            text,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
        )

    def encode_ordinary(self, text):
        if self._real is not None:
            return self._real.encode_ordinary(text)
        return self._fallback.encode_ordinary(text)

    def encode_ordinary_batch(self, texts, *, num_threads=1):
        if self._real is not None:
            return self._real.encode_ordinary_batch(texts, num_threads=num_threads)
        return self._fallback.encode_ordinary_batch(texts, num_threads=num_threads)

    def decode(self, tokens):
        if self._real is not None:
            return self._real.decode(tokens)
        return self._fallback.decode(tokens)

    def decode_single_token_bytes(self, token):
        if self._real is not None:
            return self._real.decode_single_token_bytes(token)
        return self._fallback.decode_single_token_bytes(token)

    def __repr__(self):
        if self._real is not None:
            return f"Encoding({self.name!r}, real)"
        return f"Encoding({self.name!r}, fallback)"


def get_encoding(encoding_name: str) -> Encoding:
    if encoding_name not in _ENCODING_NAMES:
        raise ValueError(
            f"Unknown encoding {encoding_name!r}. "
            f"Known: {sorted(_ENCODING_NAMES)}"
        )
    if encoding_name not in _registry:
        _registry[encoding_name] = Encoding(encoding_name)
    return _registry[encoding_name]


def list_encoding_names() -> List[str]:
    return sorted(_ENCODING_NAMES)


def encoding_for_model(model_name: str) -> Encoding:
    enc_name = _MODEL_TO_ENCODING.get(model_name)
    if enc_name is None:
        raise KeyError(
            f"Could not map {model_name!r} to a tokenizer. "
            f"Supported: {sorted(_MODEL_TO_ENCODING.keys())}"
        )
    return get_encoding(enc_name)


def tokenizer_for_model(model_name: str) -> Encoding:
    return encoding_for_model(model_name)
