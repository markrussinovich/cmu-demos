import type { ChatModelAdapter, ThreadMessage } from "@assistant-ui/react";
import type { ChatParams } from "../types";

type OpenAIMessage = { role: "user" | "assistant" | "system"; content: string };

export const toOpenAIMessages = (messages: readonly ThreadMessage[]): OpenAIMessage[] =>
  messages
    .map((m) => ({
      role: m.role,
      content: m.content
        .filter((p): p is Extract<typeof p, { type: "text" }> => p.type === "text")
        .map((p) => p.text)
        .join(""),
    }))
    .filter((m) => m.content.length > 0);

/** Yields the payload of every `data:` line in a server-sent-events body. */
export async function* iterateSSE(body: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const drain = function* (block: string) {
    for (const line of block.split("\n")) {
      if (line.startsWith("data:")) yield line.slice(5).trim();
    }
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let idx: number;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        yield* drain(block);
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) yield* drain(buffer);
  } finally {
    reader.releaseLock();
  }
}

type Part = { type: "text"; text: string } | { type: "reasoning"; text: string };

const buildParts = (reasoning: string, text: string, final: boolean): Part[] => {
  const parts: Part[] = [];
  if (reasoning) parts.push({ type: "reasoning", text: reasoning });
  if (text || (final && parts.length === 0)) parts.push({ type: "text", text });
  return parts;
};

export type OpenAIStreamAdapterOptions = {
  /** Endpoint that accepts `{messages, ...params}` and streams OpenAI-style chat completion chunks. */
  url: string;
  /** Read sampling params at request time so UI controls apply without re-creating the runtime. */
  getParams?: () => ChatParams;
  headers?: Record<string, string>;
};

/**
 * assistant-ui ChatModelAdapter that talks to an OpenAI-compatible streaming
 * chat endpoint (vLLM behind our FastAPI proxy). Reasoning deltas
 * (`reasoning_content` / `reasoning`) become a reasoning part, content deltas
 * become the text part. Each yield carries the cumulative state.
 */
export function createOpenAIStreamAdapter(opts: OpenAIStreamAdapterOptions): ChatModelAdapter {
  return {
    async *run({ messages, abortSignal }) {
      const params = opts.getParams?.() ?? {};
      const res = await fetch(opts.url, {
        method: "POST",
        headers: { "content-type": "application/json", ...(opts.headers ?? {}) },
        body: JSON.stringify({ messages: toOpenAIMessages(messages), ...params }),
        signal: abortSignal,
      });
      if (!res.ok || !res.body) {
        const detail = (await res.text().catch(() => "")).slice(0, 600);
        throw new Error(`HTTP ${res.status} ${res.statusText}${detail ? `: ${detail}` : ""}`);
      }

      let text = "";
      let reasoning = "";
      let finish: string | null = null;
      for await (const data of iterateSSE(res.body)) {
        if (data === "[DONE]") break;
        let json: any;
        try {
          json = JSON.parse(data);
        } catch {
          continue;
        }
        if (json.error) {
          throw new Error(typeof json.error === "string" ? json.error : (json.error.message ?? JSON.stringify(json.error)));
        }
        const choice = json.choices?.[0];
        if (!choice) continue;
        const delta = choice.delta ?? {};
        const r = delta.reasoning_content ?? delta.reasoning;
        if (typeof r === "string") reasoning += r;
        if (typeof delta.content === "string") text += delta.content;
        if (choice.finish_reason) finish = choice.finish_reason;
        yield { content: buildParts(reasoning, text, false) };
      }
      yield {
        content: buildParts(reasoning, text, true),
        status:
          finish === "length"
            ? { type: "incomplete", reason: "length" }
            : { type: "complete", reason: "stop" },
      };
    },
  };
}
