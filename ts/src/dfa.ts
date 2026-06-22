/**
 * UTF-8 DFA state machine and dual scoring.
 *
 * Provides V_partial, V_binary_strict, V_binary_soft scoring
 * matching the Python bytecanary implementation exactly.
 */

export enum UTF8State {
  START,
  EXPECT_1,
  EXPECT_2,
  EXPECT_3,
  EXPECT_2_AFTER_1,
  EXPECT_3_AFTER_1,
  EXPECT_3_AFTER_2,
  INVALID,
}

const STATE_PROGRESS: Record<UTF8State, number> = {
  [UTF8State.START]: 1.0,
  [UTF8State.EXPECT_1]: 0.5,
  [UTF8State.EXPECT_2]: 0.333,
  [UTF8State.EXPECT_2_AFTER_1]: 0.667,
  [UTF8State.EXPECT_3]: 0.25,
  [UTF8State.EXPECT_3_AFTER_1]: 0.5,
  [UTF8State.EXPECT_3_AFTER_2]: 0.75,
  [UTF8State.INVALID]: 0.0,
};

export interface UTF8Analysis {
  validChars: number;
  invalidBytes: number;
  totalBytes: number;
  finalState: UTF8State;
  pendingBytes: number;
  expectedTotal: number;
  charBoundaries: number[];
  isComplete: boolean;
  isValidPrefix: boolean;
  incompleteProgress: number;
}

export class UTF8StateMachine {
  private state = UTF8State.START;
  private pendingBytes = 0;
  private expectedTotal = 0;
  private validChars = 0;
  private invalidBytes = 0;
  private totalBytes = 0;
  private charBoundaries: number[] = [];
  private currentCharBytes: number[] = [];

  reset(): void {
    this.state = UTF8State.START;
    this.pendingBytes = 0;
    this.expectedTotal = 0;
    this.validChars = 0;
    this.invalidBytes = 0;
    this.totalBytes = 0;
    this.charBoundaries = [];
    this.currentCharBytes = [];
  }

  private isContinuation(byte: number): boolean {
    return byte >= 0x80 && byte <= 0xbf;
  }

  private completeChar(): void {
    this.validChars++;
    this.charBoundaries.push(this.totalBytes);
    this.state = UTF8State.START;
    this.pendingBytes = 0;
    this.expectedTotal = 0;
    this.currentCharBytes = [];
  }

  private markInvalid(_byte: number): void {
    this.invalidBytes++;
    this.currentCharBytes = [];
    this.state = UTF8State.START;
    this.pendingBytes = 0;
    this.expectedTotal = 0;
  }

  processByte(byte: number): void {
    this.totalBytes++;

    if (this.state === UTF8State.START) {
      if (byte <= 0x7f) {
        this.currentCharBytes = [byte];
        this.completeChar();
      } else if (byte <= 0xbf) {
        this.markInvalid(byte);
      } else if (byte <= 0xdf) {
        if (byte <= 0xc1) {
          this.markInvalid(byte);
        } else {
          this.state = UTF8State.EXPECT_1;
          this.pendingBytes = 1;
          this.expectedTotal = 2;
          this.currentCharBytes = [byte];
        }
      } else if (byte <= 0xef) {
        this.state = UTF8State.EXPECT_2_AFTER_1;
        this.pendingBytes = 2;
        this.expectedTotal = 3;
        this.currentCharBytes = [byte];
      } else if (byte <= 0xf7) {
        this.state = UTF8State.EXPECT_3_AFTER_1;
        this.pendingBytes = 3;
        this.expectedTotal = 4;
        this.currentCharBytes = [byte];
      } else {
        this.markInvalid(byte);
      }
    } else if (this.state === UTF8State.EXPECT_1) {
      if (this.isContinuation(byte)) {
        this.currentCharBytes.push(byte);
        this.completeChar();
      } else {
        this.invalidBytes++;
        this.currentCharBytes = [];
        this.state = UTF8State.START;
        this.pendingBytes = 0;
        this.expectedTotal = 0;
        this.totalBytes--;
        this.processByte(byte);
      }
    } else if (this.state === UTF8State.EXPECT_2_AFTER_1) {
      const lead = this.currentCharBytes[0];
      if (this.isContinuation(byte)) {
        if (lead === 0xe0 && byte < 0xa0) {
          this.markInvalid(byte);
        } else if (lead === 0xed && byte >= 0xa0) {
          this.markInvalid(byte);
        } else {
          this.currentCharBytes.push(byte);
          this.state = UTF8State.EXPECT_2;
          this.pendingBytes = 1;
        }
      } else {
        this.invalidBytes++;
        this.currentCharBytes = [];
        this.state = UTF8State.START;
        this.pendingBytes = 0;
        this.expectedTotal = 0;
        this.totalBytes--;
        this.processByte(byte);
      }
    } else if (this.state === UTF8State.EXPECT_2) {
      if (this.isContinuation(byte)) {
        this.currentCharBytes.push(byte);
        this.completeChar();
      } else {
        this.invalidBytes += 2;
        this.currentCharBytes = [];
        this.state = UTF8State.START;
        this.pendingBytes = 0;
        this.expectedTotal = 0;
        this.totalBytes--;
        this.processByte(byte);
      }
    } else if (this.state === UTF8State.EXPECT_3_AFTER_1) {
      const lead = this.currentCharBytes[0];
      if (this.isContinuation(byte)) {
        if (lead === 0xf0 && byte < 0x90) {
          this.markInvalid(byte);
        } else if (lead === 0xf4 && byte >= 0x90) {
          this.markInvalid(byte);
        } else if (lead > 0xf4) {
          this.markInvalid(byte);
        } else {
          this.currentCharBytes.push(byte);
          this.state = UTF8State.EXPECT_3_AFTER_2;
          this.pendingBytes = 2;
        }
      } else {
        this.invalidBytes++;
        this.currentCharBytes = [];
        this.state = UTF8State.START;
        this.pendingBytes = 0;
        this.expectedTotal = 0;
        this.totalBytes--;
        this.processByte(byte);
      }
    } else if (this.state === UTF8State.EXPECT_3_AFTER_2) {
      if (this.isContinuation(byte)) {
        this.currentCharBytes.push(byte);
        this.state = UTF8State.EXPECT_3;
        this.pendingBytes = 1;
      } else {
        this.invalidBytes += 2;
        this.currentCharBytes = [];
        this.state = UTF8State.START;
        this.pendingBytes = 0;
        this.expectedTotal = 0;
        this.totalBytes--;
        this.processByte(byte);
      }
    } else if (this.state === UTF8State.EXPECT_3) {
      if (this.isContinuation(byte)) {
        this.currentCharBytes.push(byte);
        this.completeChar();
      } else {
        this.invalidBytes += 3;
        this.currentCharBytes = [];
        this.state = UTF8State.START;
        this.pendingBytes = 0;
        this.expectedTotal = 0;
        this.totalBytes--;
        this.processByte(byte);
      }
    }
  }

  processBytes(data: Uint8Array): UTF8Analysis {
    this.reset();
    for (let i = 0; i < data.length; i++) {
      this.processByte(data[i]);
    }
    return {
      validChars: this.validChars,
      invalidBytes: this.invalidBytes,
      totalBytes: this.totalBytes,
      finalState: this.state,
      pendingBytes: this.pendingBytes,
      expectedTotal: this.expectedTotal,
      charBoundaries: [...this.charBoundaries],
      isComplete: this.state === UTF8State.START,
      isValidPrefix:
        this.state !== UTF8State.INVALID && this.invalidBytes === 0,
      incompleteProgress: STATE_PROGRESS[this.state] ?? 0.0,
    };
  }
}

export interface DualScore {
  partialScore: number;
  binaryStrict: number;
  binarySoft: number;
  validChars: number;
  invalidBytes: number;
  totalBytes: number;
  isComplete: boolean;
  isValidPrefix: boolean;
  incompleteProgress: number;
  finalState: string;
}

export function computeDualScore(data: Uint8Array): DualScore {
  const sm = new UTF8StateMachine();
  const analysis = sm.processBytes(data);

  if (analysis.totalBytes === 0) {
    return {
      partialScore: 1.0,
      binaryStrict: 1.0,
      binarySoft: 1.0,
      validChars: 0,
      invalidBytes: 0,
      totalBytes: 0,
      isComplete: true,
      isValidPrefix: true,
      incompleteProgress: 1.0,
      finalState: "START",
    };
  }

  const validCharBytes =
    analysis.charBoundaries.length > 0
      ? analysis.charBoundaries[analysis.charBoundaries.length - 1]
      : 0;

  let incompleteCredit = 0.0;
  if (
    analysis.finalState !== UTF8State.START &&
    analysis.finalState !== UTF8State.INVALID
  ) {
    const bytesInIncomplete =
      analysis.expectedTotal - analysis.pendingBytes;
    incompleteCredit = bytesInIncomplete * analysis.incompleteProgress;
  }

  const partialScore =
    (validCharBytes + incompleteCredit) / analysis.totalBytes;
  const binaryStrict =
    analysis.isComplete && analysis.invalidBytes === 0 ? 1.0 : 0.0;
  const binarySoft = validCharBytes / analysis.totalBytes;

  return {
    partialScore,
    binaryStrict,
    binarySoft,
    validChars: analysis.validChars,
    invalidBytes: analysis.invalidBytes,
    totalBytes: analysis.totalBytes,
    isComplete: analysis.isComplete,
    isValidPrefix: analysis.isValidPrefix,
    incompleteProgress: analysis.incompleteProgress,
    finalState: UTF8State[analysis.finalState],
  };
}
