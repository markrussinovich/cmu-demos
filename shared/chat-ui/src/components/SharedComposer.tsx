import { ArrowUpIcon, EraserIcon, SquareIcon } from "lucide-react";
import { type FC, type KeyboardEvent, useEffect, useRef, useState } from "react";

import { cn } from "../lib/cn";
import type { ChatParams, ReasoningEffort } from "../types";

export type SharedComposerProps = {
  disabled?: boolean;
  disabledReason?: string;
  anyRunning: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
  onClear: () => void;
  params: ChatParams;
  onParamsChange: (next: ChatParams) => void;
  showThinkingToggle?: boolean;
  showReasoningEffort?: boolean;
  placeholder?: string;
};

const EFFORTS: ReasoningEffort[] = ["low", "medium", "high"];

const fieldClass =
  "w-20 rounded-md border border-border bg-background px-2 py-1 text-sm text-foreground outline-none focus:border-primary";

/** One prompt box that fans out to every model pane. Enter sends, Shift+Enter inserts a newline. */
export const SharedComposer: FC<SharedComposerProps> = ({
  disabled = false,
  disabledReason,
  anyRunning,
  onSend,
  onStop,
  onClear,
  params,
  onParamsChange,
  showThinkingToggle = false,
  showReasoningEffort = false,
  placeholder = "Send the same prompt to every model…",
}) => {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);
  const canSend = !disabled && !anyRunning && text.trim().length > 0;

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, [text]);

  useEffect(() => {
    if (!disabled && !anyRunning) ref.current?.focus();
  }, [disabled, anyRunning]);

  const submit = () => {
    if (!canSend) return;
    onSend(text.trim());
    setText("");
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="rounded-2xl border border-border bg-card p-3 focus-within:border-primary/60">
      <textarea
        ref={ref}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        rows={1}
        disabled={disabled}
        placeholder={disabled ? (disabledReason ?? "Waiting for models…") : placeholder}
        aria-label="Prompt"
        className="block w-full resize-none bg-transparent px-2 py-1.5 text-lg leading-7 text-foreground outline-none placeholder:text-muted-foreground/70 disabled:cursor-not-allowed"
      />
      <div className="mt-2 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
          <label className="flex items-center gap-2">
            Temperature
            <input
              type="number"
              min={0}
              max={2}
              step={0.1}
              value={params.temperature ?? 0.7}
              onChange={(e) => onParamsChange({ ...params, temperature: Number(e.target.value) })}
              className={fieldClass}
            />
          </label>
          <label className="flex items-center gap-2">
            Max tokens
            <input
              type="number"
              min={16}
              max={32768}
              step={64}
              value={params.max_tokens ?? 1024}
              onChange={(e) => onParamsChange({ ...params, max_tokens: Number(e.target.value) })}
              className={cn(fieldClass, "w-24")}
            />
          </label>
          {showThinkingToggle && (
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={params.enable_thinking ?? false}
                onChange={(e) => onParamsChange({ ...params, enable_thinking: e.target.checked })}
                className="size-4 accent-primary"
              />
              Thinking
            </label>
          )}
          {showReasoningEffort && (
            <label className="flex items-center gap-2">
              Reasoning
              <select
                value={params.reasoning_effort ?? "low"}
                onChange={(e) => onParamsChange({ ...params, reasoning_effort: e.target.value as ReasoningEffort })}
                className={cn(fieldClass, "w-28")}
              >
                {EFFORTS.map((e) => (
                  <option key={e} value={e}>
                    {e}
                  </option>
                ))}
              </select>
            </label>
          )}
          <span className="hidden text-xs text-muted-foreground/70 md:inline">Enter to send · Shift+Enter for newline</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onClear}
            className="inline-flex items-center gap-1.5 rounded-full border border-border px-3 py-1.5 text-sm text-muted-foreground hover:bg-accent hover:text-foreground"
          >
            <EraserIcon className="size-4" /> Clear
          </button>
          {anyRunning ? (
            <button
              type="button"
              onClick={onStop}
              className="inline-flex items-center gap-1.5 rounded-full bg-destructive px-4 py-1.5 text-sm font-medium text-background hover:opacity-90"
            >
              <SquareIcon className="size-3.5 fill-current" /> Stop
            </button>
          ) : (
            <button
              type="button"
              onClick={submit}
              disabled={!canSend}
              className="inline-flex items-center gap-1.5 rounded-full bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <ArrowUpIcon className="size-4" /> Send to all
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
