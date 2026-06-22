"""
Generic byte-level tokenizer wrapper.

Auto-detects byte token mapping for SentencePiece (LLaMA, Gemma, Mistral, etc.)
and GPT-2 BPE (GPT-2, GPT-J, GPT-NeoX, etc.) tokenizers.
"""

from typing import Dict, Tuple

SPIECE_UNDERLINE = "▁"


def _bytes_to_unicode() -> Dict[int, str]:
    """GPT-2 byte-to-unicode mapping (same as openai/gpt-2)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class ByteTokenizer:
    """Wraps a HuggingFace tokenizer with byte-level encode/decode."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._type = self._detect_type()
        self._byte_to_id, self._id_to_byte = self._build_maps()
        if self._type == "gpt2":
            self._unicode_to_byte = {v: k for k, v in _bytes_to_unicode().items()}
        self._adds_space_prefix = self._detect_space_prefix()

    def _detect_type(self) -> str:
        vocab = self.tokenizer.get_vocab()
        if "<0x00>" in vocab:
            return "sentencepiece"
        btu = _bytes_to_unicode()
        if btu[0] in vocab:
            return "gpt2"
        raise ValueError(
            "Could not detect byte token format. "
            "Tokenizer must have SentencePiece byte-fallback (<0xHH>) or GPT-2 byte-level BPE tokens."
        )

    def _build_maps(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        vocab = self.tokenizer.get_vocab()
        if self._type == "sentencepiece":
            byte_to_id = {}
            for b in range(256):
                key = f"<0x{b:02X}>"
                if key in vocab:
                    byte_to_id[b] = vocab[key]
            if len(byte_to_id) < 256:
                raise ValueError(
                    f"Incomplete SentencePiece byte map: {len(byte_to_id)}/256 tokens found"
                )
        else:
            btu = _bytes_to_unicode()
            byte_to_id = {b: vocab[btu[b]] for b in range(256) if btu[b] in vocab}
            if len(byte_to_id) < 256:
                raise ValueError(
                    f"Incomplete GPT-2 byte map: {len(byte_to_id)}/256 tokens found"
                )
        id_to_byte = {v: k for k, v in byte_to_id.items()}
        return byte_to_id, id_to_byte

    def _detect_space_prefix(self) -> bool:
        """Check if tokenizer prepends a space when encoding a single character."""
        ids = self.tokenizer.encode("A", add_special_tokens=False)
        raw = self.token_ids_to_bytes(ids)
        return raw[:1] == b" "

    @property
    def adds_space_prefix(self) -> bool:
        return self._adds_space_prefix

    def bytes_to_token_ids(self, data: bytes) -> list:
        return [self._byte_to_id[b] for b in data]

    def token_ids_to_bytes(self, token_ids: list) -> bytes:
        if self._type == "sentencepiece":
            return self._spiece_to_bytes(token_ids)
        return self._gpt2_to_bytes(token_ids)

    def _spiece_to_bytes(self, token_ids: list) -> bytes:
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        result = bytearray()
        for token in tokens:
            if token is None:
                continue
            if token.startswith("<0x") and token.endswith(">"):
                result.append(int(token[3:-1], 16))
            elif token == SPIECE_UNDERLINE:
                result.append(0x20)
            elif token.startswith(SPIECE_UNDERLINE):
                result.append(0x20)
                result.extend(token[1:].encode("utf-8"))
            else:
                result.extend(token.encode("utf-8"))
        return bytes(result)

    def _gpt2_to_bytes(self, token_ids: list) -> bytes:
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        result = bytearray()
        for token in tokens:
            if token is None:
                continue
            for ch in token:
                if ch in self._unicode_to_byte:
                    result.append(self._unicode_to_byte[ch])
                else:
                    result.extend(ch.encode("utf-8"))
        return bytes(result)

    def get_char_prefix_tokens(self, char: str, num_bytes: int) -> Tuple[list, bytes, bytes]:
        """Create byte-level prefix token IDs for a character.

        Returns (prefix_token_ids, prefix_bytes, remaining_bytes).
        """
        char_bytes = char.encode("utf-8")
        prefix_bytes = char_bytes[:num_bytes]
        remaining_bytes = char_bytes[num_bytes:]

        if not prefix_bytes:
            return [], b"", char_bytes

        if self._adds_space_prefix:
            input_bytes = b" " + prefix_bytes
        else:
            input_bytes = prefix_bytes

        token_ids = self.bytes_to_token_ids(input_bytes)
        return token_ids, prefix_bytes, remaining_bytes


def analyze_utf8_validity(byte_data: bytes) -> dict:
    errors = []
    position = 0
    while position < len(byte_data):
        try:
            byte_data[position:].decode("utf-8", errors="strict")
            break
        except UnicodeDecodeError as e:
            errors.append({
                "position": position + e.start,
                "end": position + e.end,
                "invalid_bytes": list(byte_data[position + e.start : position + e.end]),
            })
            position = position + e.end
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "total_bytes": len(byte_data),
        "error_count": len(errors),
    }
