import {
  ErrorPrimitive,
  MessagePrimitive,
  type ReasoningMessagePartComponent,
  type TextMessagePartComponent,
  ThreadPrimitive,
  type ToolCallMessagePartComponent,
  useAuiState,
} from "@assistant-ui/react";
import { ArrowDownIcon, DatabaseIcon, ShieldAlertIcon, WrenchIcon } from "lucide-react";
import { type FC, type ReactNode, useEffect, useRef } from "react";

import { useChatUiConfig } from "../context";
import { cn } from "../lib/cn";
import { formatCall, matchesTarget, scenarioOf } from "../lib/scenario";
import { splitMarkedSpans } from "../lib/spans";
import type { Scenario } from "../types";
import { MarkdownText } from "./MarkdownText";

/** User text; spans wrapped in the demo's marker delimiters are rendered as "hidden" text. */
const UserText: TextMessagePartComponent = ({ text }) => {
  const { spanMarker } = useChatUiConfig();
  if (!spanMarker) return <p className="whitespace-pre-wrap break-words">{text}</p>;
  const segments = splitMarkedSpans(text, spanMarker);
  return (
    <p className="whitespace-pre-wrap break-words">
      {segments.map((seg, i) =>
        seg.marked ? (
          <mark
            key={i}
            title={spanMarker.label ?? "marked span"}
            className="rounded-sm bg-warning/15 px-1 text-warning/90 decoration-warning/70 underline decoration-dotted underline-offset-4"
          >
            <span className="select-none text-xs opacity-60">{spanMarker.open} </span>
            {seg.text}
            <span className="select-none text-xs opacity-60"> {spanMarker.close}</span>
          </mark>
        ) : (
          <span key={i}>{seg.text}</span>
        ),
      )}
    </p>
  );
};

const ReasoningPart: ReasoningMessagePartComponent = ({ text, status }) => {
  const running = status?.type === "running";
  return (
    <details open={running} className="my-2 rounded-lg border border-border/60 bg-muted/40 px-3 py-2 text-[0.9em]">
      <summary className="cursor-pointer select-none text-muted-foreground">
        {running ? "Thinking…" : "Thoughts"}
      </summary>
      <div className="mt-2 whitespace-pre-wrap break-words text-muted-foreground">{text}</div>
    </details>
  );
};

/** Shown while the assistant message has no parts yet (waiting for the first token). */
const EmptyAssistantContent: FC = () => {
  const running = useAuiState((s) => s.message.status?.type === "running");
  // a message that ends in tool calls carries an empty trailing text part: nothing to say
  const hasOtherParts = useAuiState((s) => s.message.content.some((p) => p.type !== "text" || p.text.length > 0));
  if (hasOtherParts) return null;
  if (running) {
    return (
      <span className="inline-flex items-center gap-1 text-muted-foreground" aria-label="Generating">
        <span className="animate-pulse">●</span>
      </span>
    );
  }
  return <span className="text-muted-foreground italic">(no output)</span>;
};

/** The scenario of the closest preceding user message (what this assistant turn is answering). */
const useCurrentScenario = (): Scenario | null =>
  useAuiState((s) => {
    const msgs = s.thread.messages;
    const idx = msgs.findIndex((m) => m.id === s.message.id);
    for (let i = (idx < 0 ? msgs.length : idx) - 1; i >= 0; i--) {
      const sc = scenarioOf(msgs[i]!);
      if (sc) return sc;
      if (msgs[i]!.role === "user") return null;
    }
    return null;
  });

/** A tool call the model emitted. Flagged when it matches the scenario's attacker target. */
const ToolCallCard: ToolCallMessagePartComponent = ({ toolName, args, argsText }) => {
  const scenario = useCurrentScenario();
  const hit = matchesTarget(scenario, toolName, argsText || JSON.stringify(args ?? {}));
  const entries = Object.entries((args ?? {}) as Record<string, unknown>);
  return (
    <div
      className={cn(
        "my-2 overflow-hidden rounded-xl border text-[0.95em]",
        hit ? "border-destructive/70 bg-destructive/5" : "border-border bg-muted/30",
      )}
    >
      <div className={cn("flex items-center gap-2 border-b px-3 py-1.5 font-mono text-sm", hit ? "border-destructive/30 text-destructive" : "border-border/60 text-foreground")}>
        <WrenchIcon className="size-4 shrink-0" />
        <span className="font-semibold">{toolName || "tool call"}</span>
        <span className="text-xs uppercase tracking-wide text-muted-foreground">tool call</span>
        {hit && (
          <span className="ms-auto inline-flex items-center gap-1 rounded-full bg-destructive px-2 py-0.5 text-xs font-semibold uppercase tracking-wide text-background">
            <ShieldAlertIcon className="size-3.5" /> {scenario?.target?.label ?? "attacker's goal reached"}
          </span>
        )}
      </div>
      <dl className="grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1 px-3 py-2">
        {entries.length === 0 && <dd className="col-span-2 font-mono text-xs text-muted-foreground">{argsText || "(no arguments)"}</dd>}
        {entries.map(([k, v]) => {
          const text = typeof v === "string" ? v : JSON.stringify(v);
          const flagged = hit && scenario?.target?.pattern && safeTest(scenario.target.pattern, text);
          return (
            <div key={k} className="contents">
              <dt className="font-mono text-sm text-muted-foreground">{k}</dt>
              <dd className={cn("break-words font-mono text-sm", flagged && "font-semibold text-destructive")}>{text}</dd>
            </div>
          );
        })}
      </dl>
    </div>
  );
};

const safeTest = (pattern: string, text: string): boolean => {
  try {
    return new RegExp(pattern, "i").test(text);
  } catch {
    return false;
  }
};

/** Untrusted tool result with the injected instruction highlighted. */
const ToolResultBlock: FC<{ scenario: Scenario }> = ({ scenario }) => {
  const { result, injection } = scenario;
  const i = injection ? result.indexOf(injection) : -1;
  const preRef = useRef<HTMLPreElement>(null);
  const markRef = useRef<HTMLElement>(null);
  // bring the injected sentence into view inside the card (never scroll the page)
  useEffect(() => {
    const pre = preRef.current;
    const mark = markRef.current;
    if (pre && mark) pre.scrollTop = Math.max(0, mark.offsetTop - pre.clientHeight / 2);
  }, [result, injection]);
  const body =
    i >= 0 && injection ? (
      <>
        {result.slice(0, i)}
        <mark ref={markRef} className="rounded-sm bg-warning/25 px-0.5 font-semibold text-warning" title="injected instruction">
          {injection}
        </mark>
        {result.slice(i + injection.length)}
      </>
    ) : (
      result
    );
  return (
    <div className="mt-2 overflow-hidden rounded-xl border border-warning/50 bg-warning/5">
      <div className="flex items-center gap-2 border-b border-warning/30 px-3 py-1.5 text-sm">
        <DatabaseIcon className="size-4 shrink-0 text-warning" />
        <span className="font-mono font-semibold">{formatCall(scenario.tool.name, scenario.tool.arguments)}</span>
        <span className="ms-auto text-xs uppercase tracking-wide text-warning">tool result · untrusted</span>
      </div>
      <pre ref={preRef} className="relative max-h-56 overflow-auto whitespace-pre-wrap break-words px-3 py-2 font-mono text-xs leading-5 text-foreground/80">{body}</pre>
    </div>
  );
};

const UserMessage: FC = () => {
  const scenario = useAuiState((s) => scenarioOf(s.message));
  return (
    <MessagePrimitive.Root data-role="user" className="flex flex-col items-end gap-1 px-1">
      <div className="max-w-[85%] rounded-2xl bg-muted px-4 py-2.5 text-foreground">
        <MessagePrimitive.Parts components={{ Text: UserText }} />
      </div>
      {scenario && (
        <div className="w-full">
          <ToolResultBlock scenario={scenario} />
        </div>
      )}
    </MessagePrimitive.Root>
  );
};

const AssistantMessage: FC = () => {
  const incomplete = useAuiState((s) =>
    s.message.status?.type === "incomplete" ? s.message.status.reason : null,
  );
  return (
    <MessagePrimitive.Root data-role="assistant" className="px-1 leading-relaxed text-foreground break-words">
      <MessagePrimitive.Parts
        components={{ Text: MarkdownText, Reasoning: ReasoningPart, Empty: EmptyAssistantContent, tools: { Fallback: ToolCallCard } }}
      />
      {incomplete && incomplete !== "error" && (
        <p className="mt-2 text-xs uppercase tracking-wide text-muted-foreground">
          {incomplete === "length" ? "stopped: max tokens reached" : `stopped: ${incomplete}`}
        </p>
      )}
      <MessagePrimitive.Error>
        <ErrorPrimitive.Root className="mt-2 rounded-md border border-destructive/60 bg-destructive/10 p-3 text-sm text-destructive">
          <ErrorPrimitive.Message className="break-words" />
        </ErrorPrimitive.Root>
      </MessagePrimitive.Error>
    </MessagePrimitive.Root>
  );
};

const ThreadMessage: FC = () => {
  const role = useAuiState((s) => s.message.role);
  return role === "user" ? <UserMessage /> : <AssistantMessage />;
};

export type ThreadProps = {
  /** Rendered centred while the thread has no messages. */
  empty?: ReactNode;
};

/**
 * A read-only assistant-ui thread (no per-thread composer): the demos drive
 * every pane from one shared composer. Must render inside an
 * AssistantRuntimeProvider.
 */
export const Thread: FC<ThreadProps> = ({ empty }) => {
  const isEmpty = useAuiState((s) => s.thread.messages.length === 0);
  return (
    <ThreadPrimitive.Root className="aui-root aui-thread-root flex h-full min-h-0 flex-col">
      <ThreadPrimitive.Viewport className="relative flex flex-1 flex-col overflow-y-auto scroll-smooth px-3 pt-4">
        {isEmpty && empty !== undefined && (
          <div className="m-auto max-w-md text-center text-muted-foreground">{empty}</div>
        )}
        <div className="flex flex-col gap-y-5 pb-4">
          <ThreadPrimitive.Messages>{() => <ThreadMessage />}</ThreadPrimitive.Messages>
        </div>
        <div className="pointer-events-none sticky bottom-2 mt-auto flex justify-center">
          <ThreadPrimitive.ScrollToBottom
            className="pointer-events-auto rounded-full border border-border bg-card p-2 text-muted-foreground shadow-lg hover:text-foreground disabled:hidden"
            aria-label="Scroll to bottom"
          >
            <ArrowDownIcon className="size-4" />
          </ThreadPrimitive.ScrollToBottom>
        </div>
      </ThreadPrimitive.Viewport>
    </ThreadPrimitive.Root>
  );
};
