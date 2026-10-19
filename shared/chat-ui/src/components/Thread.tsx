import {
  ErrorPrimitive,
  MessagePrimitive,
  type ReasoningMessagePartComponent,
  type TextMessagePartComponent,
  ThreadPrimitive,
  useAuiState,
} from "@assistant-ui/react";
import { ArrowDownIcon } from "lucide-react";
import type { FC, ReactNode } from "react";

import { useChatUiConfig } from "../context";
import { splitMarkedSpans } from "../lib/spans";
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
  if (running) {
    return (
      <span className="inline-flex items-center gap-1 text-muted-foreground" aria-label="Generating">
        <span className="animate-pulse">●</span>
      </span>
    );
  }
  return <span className="text-muted-foreground italic">(no output)</span>;
};

const UserMessage: FC = () => (
  <MessagePrimitive.Root data-role="user" className="flex justify-end px-1">
    <div className="max-w-[85%] rounded-2xl bg-muted px-4 py-2.5 text-foreground">
      <MessagePrimitive.Parts components={{ Text: UserText }} />
    </div>
  </MessagePrimitive.Root>
);

const AssistantMessage: FC = () => {
  const incomplete = useAuiState((s) =>
    s.message.status?.type === "incomplete" ? s.message.status.reason : null,
  );
  return (
    <MessagePrimitive.Root data-role="assistant" className="px-1 leading-relaxed text-foreground break-words">
      <MessagePrimitive.Parts
        components={{ Text: MarkdownText, Reasoning: ReasoningPart, Empty: EmptyAssistantContent }}
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
