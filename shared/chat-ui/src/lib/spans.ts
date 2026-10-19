import type { SpanMarker } from "../types";

export type TextSegment = { text: string; marked: boolean };

const escapeRegExp = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** Splits text into plain and marked segments using the marker's open/close delimiters. */
export const splitMarkedSpans = (text: string, marker: SpanMarker): TextSegment[] => {
  const re = new RegExp(`${escapeRegExp(marker.open)}([\\s\\S]*?)${escapeRegExp(marker.close)}`, "g");
  const out: TextSegment[] = [];
  let last = 0;
  for (const m of text.matchAll(re)) {
    const start = m.index ?? 0;
    if (start > last) out.push({ text: text.slice(last, start), marked: false });
    out.push({ text: m[1] ?? "", marked: true });
    last = start + m[0].length;
  }
  if (last < text.length) out.push({ text: text.slice(last), marked: false });
  return out;
};

export const countMarkedSpans = (text: string, marker: SpanMarker): number =>
  splitMarkedSpans(text, marker).filter((s) => s.marked).length;

/**
 * Wraps `text[start, end)` with the marker, keeping exactly one space on each side of the
 * delimiters (the trigger must tokenize as standalone pieces).
 */
export const wrapSelection = (text: string, start: number, end: number, marker: SpanMarker): { text: string; cursor: number } => {
  const before = text.slice(0, start).replace(/\s+$/, "");
  const inner = text.slice(start, end).trim();
  const after = text.slice(end).replace(/^[ \t]+/, "");
  const lead = before.length > 0 && !before.endsWith("\n") ? " " : "";
  const trail = after.length > 0 && !after.startsWith("\n") ? " " : "";
  const wrapped = `${before}${lead}${marker.open} ${inner} ${marker.close}${trail}`;
  return { text: wrapped + after, cursor: wrapped.length };
};
