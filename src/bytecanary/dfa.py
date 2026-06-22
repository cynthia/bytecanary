"""
UTF-8 DFA state machine and dual scoring for byte-level evaluation.

Provides V_partial, V_binary_strict, V_binary_soft scoring.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List


class UTF8State(Enum):
    START = auto()
    EXPECT_1 = auto()
    EXPECT_2 = auto()
    EXPECT_3 = auto()
    EXPECT_2_AFTER_1 = auto()
    EXPECT_3_AFTER_1 = auto()
    EXPECT_3_AFTER_2 = auto()
    INVALID = auto()


STATE_PROGRESS = {
    UTF8State.START: 1.0,
    UTF8State.EXPECT_1: 0.5,
    UTF8State.EXPECT_2: 0.333,
    UTF8State.EXPECT_2_AFTER_1: 0.667,
    UTF8State.EXPECT_3: 0.25,
    UTF8State.EXPECT_3_AFTER_1: 0.5,
    UTF8State.EXPECT_3_AFTER_2: 0.75,
    UTF8State.INVALID: 0.0,
}


@dataclass
class UTF8Analysis:
    valid_chars: int
    invalid_bytes: int
    total_bytes: int
    final_state: UTF8State
    pending_bytes: int
    expected_total: int
    char_boundaries: List[int] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.final_state == UTF8State.START

    @property
    def is_valid_prefix(self) -> bool:
        return self.final_state != UTF8State.INVALID and self.invalid_bytes == 0

    @property
    def incomplete_progress(self) -> float:
        return STATE_PROGRESS.get(self.final_state, 0.0)


class UTF8StateMachine:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = UTF8State.START
        self.pending_bytes = 0
        self.expected_total = 0
        self.valid_chars = 0
        self.invalid_bytes = 0
        self.total_bytes = 0
        self.char_boundaries: List[int] = []
        self.current_char_bytes: List[int] = []

    def _is_continuation(self, byte: int) -> bool:
        return 0x80 <= byte <= 0xBF

    def _complete_char(self):
        self.valid_chars += 1
        self.char_boundaries.append(self.total_bytes)
        self.state = UTF8State.START
        self.pending_bytes = 0
        self.expected_total = 0
        self.current_char_bytes = []

    def _mark_invalid(self, byte: int):
        self.invalid_bytes += 1
        self.current_char_bytes = []
        self.state = UTF8State.START
        self.pending_bytes = 0
        self.expected_total = 0

    def process_byte(self, byte: int):
        self.total_bytes += 1

        if self.state == UTF8State.START:
            if byte <= 0x7F:
                self.current_char_bytes = [byte]
                self._complete_char()
            elif byte <= 0xBF:
                self._mark_invalid(byte)
            elif byte <= 0xDF:
                if byte <= 0xC1:
                    self._mark_invalid(byte)
                else:
                    self.state = UTF8State.EXPECT_1
                    self.pending_bytes = 1
                    self.expected_total = 2
                    self.current_char_bytes = [byte]
            elif byte <= 0xEF:
                self.state = UTF8State.EXPECT_2_AFTER_1
                self.pending_bytes = 2
                self.expected_total = 3
                self.current_char_bytes = [byte]
            elif byte <= 0xF7:
                self.state = UTF8State.EXPECT_3_AFTER_1
                self.pending_bytes = 3
                self.expected_total = 4
                self.current_char_bytes = [byte]
            else:
                self._mark_invalid(byte)

        elif self.state == UTF8State.EXPECT_1:
            if self._is_continuation(byte):
                self.current_char_bytes.append(byte)
                self._complete_char()
            else:
                self.invalid_bytes += 1
                self.current_char_bytes = []
                self.state = UTF8State.START
                self.pending_bytes = 0
                self.expected_total = 0
                self.total_bytes -= 1
                self.process_byte(byte)

        elif self.state == UTF8State.EXPECT_2_AFTER_1:
            lead = self.current_char_bytes[0]
            if self._is_continuation(byte):
                if lead == 0xE0 and byte < 0xA0:
                    self._mark_invalid(byte)
                elif lead == 0xED and byte >= 0xA0:
                    self._mark_invalid(byte)
                else:
                    self.current_char_bytes.append(byte)
                    self.state = UTF8State.EXPECT_2
                    self.pending_bytes = 1
            else:
                self.invalid_bytes += 1
                self.current_char_bytes = []
                self.state = UTF8State.START
                self.pending_bytes = 0
                self.expected_total = 0
                self.total_bytes -= 1
                self.process_byte(byte)

        elif self.state == UTF8State.EXPECT_2:
            if self._is_continuation(byte):
                self.current_char_bytes.append(byte)
                self._complete_char()
            else:
                self.invalid_bytes += 2
                self.current_char_bytes = []
                self.state = UTF8State.START
                self.pending_bytes = 0
                self.expected_total = 0
                self.total_bytes -= 1
                self.process_byte(byte)

        elif self.state == UTF8State.EXPECT_3_AFTER_1:
            lead = self.current_char_bytes[0]
            if self._is_continuation(byte):
                if lead == 0xF0 and byte < 0x90:
                    self._mark_invalid(byte)
                elif lead == 0xF4 and byte >= 0x90:
                    self._mark_invalid(byte)
                elif lead > 0xF4:
                    self._mark_invalid(byte)
                else:
                    self.current_char_bytes.append(byte)
                    self.state = UTF8State.EXPECT_3_AFTER_2
                    self.pending_bytes = 2
            else:
                self.invalid_bytes += 1
                self.current_char_bytes = []
                self.state = UTF8State.START
                self.pending_bytes = 0
                self.expected_total = 0
                self.total_bytes -= 1
                self.process_byte(byte)

        elif self.state == UTF8State.EXPECT_3_AFTER_2:
            if self._is_continuation(byte):
                self.current_char_bytes.append(byte)
                self.state = UTF8State.EXPECT_3
                self.pending_bytes = 1
            else:
                self.invalid_bytes += 2
                self.current_char_bytes = []
                self.state = UTF8State.START
                self.pending_bytes = 0
                self.expected_total = 0
                self.total_bytes -= 1
                self.process_byte(byte)

        elif self.state == UTF8State.EXPECT_3:
            if self._is_continuation(byte):
                self.current_char_bytes.append(byte)
                self._complete_char()
            else:
                self.invalid_bytes += 3
                self.current_char_bytes = []
                self.state = UTF8State.START
                self.pending_bytes = 0
                self.expected_total = 0
                self.total_bytes -= 1
                self.process_byte(byte)

    def process_bytes(self, data: bytes) -> UTF8Analysis:
        self.reset()
        for byte in data:
            self.process_byte(byte)

        return UTF8Analysis(
            valid_chars=self.valid_chars,
            invalid_bytes=self.invalid_bytes,
            total_bytes=self.total_bytes,
            final_state=self.state,
            pending_bytes=self.pending_bytes,
            expected_total=self.expected_total,
            char_boundaries=self.char_boundaries.copy(),
        )


@dataclass
class DualScore:
    partial_score: float
    binary_strict: float
    binary_soft: float
    valid_chars: int
    invalid_bytes: int
    total_bytes: int
    is_complete: bool
    is_valid_prefix: bool
    incomplete_progress: float
    final_state: str


def compute_dual_score(byte_data: bytes) -> DualScore:
    sm = UTF8StateMachine()
    analysis = sm.process_bytes(byte_data)

    if analysis.total_bytes == 0:
        return DualScore(
            partial_score=1.0,
            binary_strict=1.0,
            binary_soft=1.0,
            valid_chars=0,
            invalid_bytes=0,
            total_bytes=0,
            is_complete=True,
            is_valid_prefix=True,
            incomplete_progress=1.0,
            final_state="START",
        )

    valid_char_bytes = analysis.char_boundaries[-1] if analysis.char_boundaries else 0

    incomplete_credit = 0.0
    if analysis.final_state not in (UTF8State.START, UTF8State.INVALID):
        bytes_in_incomplete = analysis.expected_total - analysis.pending_bytes
        incomplete_credit = bytes_in_incomplete * analysis.incomplete_progress

    partial_score = (valid_char_bytes + incomplete_credit) / analysis.total_bytes
    binary_strict = 1.0 if (analysis.is_complete and analysis.invalid_bytes == 0) else 0.0
    binary_soft = valid_char_bytes / analysis.total_bytes

    return DualScore(
        partial_score=partial_score,
        binary_strict=binary_strict,
        binary_soft=binary_soft,
        valid_chars=analysis.valid_chars,
        invalid_bytes=analysis.invalid_bytes,
        total_bytes=analysis.total_bytes,
        is_complete=analysis.is_complete,
        is_valid_prefix=analysis.is_valid_prefix,
        incomplete_progress=analysis.incomplete_progress,
        final_state=analysis.final_state.name,
    )
