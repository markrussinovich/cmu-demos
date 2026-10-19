import type { ChatModelAdapter, ThreadMessage } from "@assistant-ui/react";

import type { ChatParams, Scenario, ToolSchema } from "../types";
import { scenarioOf } from "./scenario";

type OpenAIToolCall = { id: string; type: "function"; function: { name: string; arguments: string } };
type OpenAIMessage =
  | { role: "user" | "system"; content: string }
  | { role: "assistant"; content: string; tool_calls?: OpenAIToolCall[] }
  | { role: "tool"; name: string; tool_call_id: string; content: string };

const textOf = (m: ThreadMessage): string =>
  m.content
    .filter((p): p is Extract<typeof p, { type: "text" }> => p.type === "text")
    .map((p) => p.text)
    .join("");

/**
 * Plain chat history: text parts only (tool-call parts of earlier assistant turns are dropped;
 * the demos never execute tools, so there is nothing to continue from).
 */
export const toOpenAIMessages = (messages: readonly ThreadMessage[]): OpenAIMessage[] =>
  messages
    .map((m) => ({ role: m.role, content: textOf(m) }) as OpenAIMessage)
    .filter((m) => m.content.length > 0);

/**
 * Scenario request: system prompt, the user's task, the retrieval tool call the agent already
 * made, and the tool's (untrusted) result. Earlier thread history is intentionally NOT included:
 * every scenario is a fresh agent episode, which is what the comparison is about.
 */
export const scenarioToRequest = (userText: string, s: Scenario): { messages: OpenAIMessage[]; tools?: ToolSchema[] } => {
  const callId = "call_0001";
  const messages: OpenAIMessage[] = [];
  if (s.system) messages.push({ role: "system", content: s.system });
  messages.push({ role: "user", content: userText });
  messages.push({
    role: "assistant",
    content: "",
    tool_calls: [{ id: callId, type: "function", function: { name: s.tool.name, arguments: JSON.stringify(s.tool.arguments ?? {}) } }],
  });
  messages.push({ role: "tool", name: s.tool.name, tool_call_id: callId, content: s.result });
  return { messages, tools: s.tools };
};

/** Build the request body (messages + tools) from the thread, honouring a scenario on the last user message. */
export const buildRequest = (messages: readonly ThreadMessage[]): { messages: OpenAIMessage[]; tools?: ToolSchema[] } => {
  const lastUser = [...messages].reverse().find((m) => m.role === "user");
  const scenario = lastUser ? scenarioOf(lastUser) : null;
  if (lastUser && scenario) return scenarioToRequest(textOf(lastUser), scenario);
  return { messages: toOpenAIMessages(messages) };
};

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

type ToolCallAcc = { id: string; name: string; argsText: string };
type Part =
  | { type: "text"; text: string }
  | { type: "reasoning"; text: string }
  | { type: "tool-call"; toolCallId: string; toolName: string; args: Record<string, any>; argsText: string };

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const parseArgs = (argsText: string): Record<string, any> => {
  try {
    const v = JSON.parse(argsText);
    return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, any>) : { value: v };
  } catch {
    return {};
  }
};

/**
 * Parts in arrival order: reasoning first, then text/tool calls interleaved as they came.
 * `order` records the sequence of segment kinds so a tool call emitted mid-text keeps its place.
 */
const buildParts = (reasoning: string, segments: Array<{ kind: "text"; text: string } | { kind: "tool"; call: ToolCallAcc }>, final: boolean): Part[] => {
  const parts: Part[] = [];
  if (reasoning) parts.push({ type: "reasoning", text: reasoning });
  for (const seg of segments) {
    if (seg.kind === "text") {
      if (seg.text) parts.push({ type: "text", text: seg.text });
    } else {
      parts.push({ type: "tool-call", toolCallId: seg.call.id, toolName: seg.call.name, args: parseArgs(seg.call.argsText), argsText: seg.call.argsText });
    }
  }
  if (final && parts.length === 0) parts.push({ type: "text", text: "" });
  return parts;
};

export type OpenAIStreamAdapterOptions = {
  /** Endpoint that accepts `{messages, tools?, ...params}` and streams OpenAI-style chat completion chunks. */
  url: string;
  /** Read sampling params at request time so UI controls apply without re-creating the runtime. */
  getParams?: () => ChatParams;
  headers?: Record<string, string>;
};

/**
 * assistant-ui ChatModelAdapter that talks to an OpenAI-compatible streaming chat endpoint
 * (vLLM or the countersteer server behind our FastAPI proxy). Reasoning deltas become a
 * reasoning part, content deltas text parts, `tool_calls` deltas tool-call parts. Each
 * yield carries the cumulative state.
 */
export function createOpenAIStreamAdapter(opts: OpenAIStreamAdapterOptions): ChatModelAdapter {
  return {
    async *run({ messages, abortSignal }) {
      const params = opts.getParams?.() ?? {};
      const req = buildRequest(messages);
      const res = await fetch(opts.url, {
        method: "POST",
        headers: { "content-type": "application/json", ...(opts.headers ?? {}) },
        body: JSON.stringify({ ...req, ...params }),
        signal: abortSignal,
      });
      if (!res.ok || !res.body) {
        const detail = (await res.text().catch(() => "")).slice(0, 600);
        throw new Error(`HTTP ${res.status} ${res.statusText}${detail ? `: ${detail}` : ""}`);
      }

      let reasoning = "";
      const segments: Array<{ kind: "text"; text: string } | { kind: "tool"; call: ToolCallAcc }> = [];
      const callsByIndex = new Map<number, ToolCallAcc>();
      let finish: string | null = null;
      const appendText = (t: string) => {
        const last = segments[segments.length - 1];
        if (last && last.kind === "text") last.text += t;
        else segments.push({ kind: "text", text: t });
      };
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
        if (typeof delta.content === "string" && delta.content) appendText(delta.content);
        if (Array.isArray(delta.tool_calls)) {
          for (const tc of delta.tool_calls) {
            const idx: number = typeof tc.index === "number" ? tc.index : callsByIndex.size;
            let acc = callsByIndex.get(idx);
            if (!acc) {
              acc = { id: tc.id ?? `call_${idx}`, name: tc.function?.name ?? "", argsText: "" };
              callsByIndex.set(idx, acc);
              segments.push({ kind: "tool", call: acc });
            }
            if (tc.function?.name && !acc.name) acc.name = tc.function.name;
            if (typeof tc.function?.arguments === "string") acc.argsText += tc.function.arguments;
          }
        }
        if (choice.finish_reason) finish = choice.finish_reason;
        yield { content: buildParts(reasoning, segments, false) };
      }
      yield {
        content: buildParts(reasoning, segments, true),
        status:
          finish === "length"
            ? { type: "incomplete", reason: "length" }
            : { type: "complete", reason: "stop" },
      };
    },
  };
}
