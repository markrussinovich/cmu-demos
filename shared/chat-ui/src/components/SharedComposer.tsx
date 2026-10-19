import { ArrowUpIcon, DatabaseIcon, EraserIcon, EyeOffIcon, SquareIcon, XIcon } from "lucide-react";
import { type FC, type KeyboardEvent, useEffect, useRef, useState } from "react";

import { cn } from "../lib/cn";
import { formatCall } from "../lib/scenario";
import { countMarkedSpans, wrapSelection } from "../lib/spans";
import type { ChatParams, Preset, ReasoningEffort, Scenario, SpanMarker } from "../types";

export type ComposerMode = "chat" | "scenario";

export type SharedComposerProps = {
  disabled?: boolean;
  disabledReason?: string;
  anyRunning: boolean;
  /** `scenario` is set when the message carries an untrusted tool result (scenario composer). */
  onSend: (text: string, scenario: Scenario | null) => void;
  onStop: () => void;
  onClear: () => void;
  params: ChatParams;
  onParamsChange: (next: ChatParams) => void;
  showThinkingToggle?: boolean;
  showReasoningEffort?: boolean;
  placeholder?: string;
  /** Example prompts offered in a dropdown; picking one fills the composer. */
  presets?: Preset[];
  /** When set, a button wraps the current selection with these delimiters. */
  spanMarker?: SpanMarker | null;
  /** "chat": one prompt box. "scenario": user request + editable untrusted tool result. */
  mode?: ComposerMode;
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
  placeholder,
  presets,
  spanMarker,
  mode = "chat",
}) => {
  const [text, setText] = useState("");
  const [scenario, setScenario] = useState<Scenario | null>(null);
  const [result, setResult] = useState("");
  const [note, setNote] = useState<string | null>(null);
  const [hasSelection, setHasSelection] = useState(false);
  const ref = useRef<HTMLTextAreaElement>(null);
  const canSend = !disabled && !anyRunning && text.trim().length > 0;
  const markedCount = spanMarker ? countMarkedSpans(text, spanMarker) : 0;
  const isScenario = mode === "scenario";
  const effectivePlaceholder =
    placeholder ?? (isScenario ? "What the user asks the agent to do…" : "Send the same prompt to every model…");

  const updateSelection = () => {
    const el = ref.current;
    setHasSelection(!!el && el.selectionEnd > el.selectionStart);
  };

  const [wrapHint, setWrapHint] = useState<string | null>(null);
  const wrapCurrentSelection = () => {
    const el = ref.current;
    if (!el || !spanMarker) return;
    if (el.selectionEnd <= el.selectionStart) {
      setWrapHint("Select the text to hide first");
      el.focus();
      window.setTimeout(() => setWrapHint(null), 2500);
      return;
    }
    const next = wrapSelection(text, el.selectionStart, el.selectionEnd, spanMarker);
    setText(next.text);
    setWrapHint(null);
    requestAnimationFrame(() => {
      el.focus();
      el.setSelectionRange(next.cursor, next.cursor);
      setHasSelection(false);
    });
  };

  const applyPreset = (label: string) => {
    const preset = presets?.find((p) => p.label === label);
    if (!preset) return;
    setText(preset.text);
    setNote(preset.note ?? null);
    if (preset.scenario) {
      setScenario(preset.scenario);
      setResult(preset.scenario.result);
    } else {
      setScenario(null);
      setResult("");
    }
    requestAnimationFrame(() => ref.current?.focus());
  };

  const dropScenario = () => {
    setScenario(null);
    setResult("");
    setNote(null);
  };

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
    const sc = isScenario && scenario && result.trim().length > 0 ? { ...scenario, result } : null;
    onSend(text.trim(), sc);
    if (!isScenario) setText("");
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="rounded-2xl border border-border bg-card p-3 focus-within:border-primary/60">
      {isScenario && (
        <div className="mb-1 flex items-center gap-2 px-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          User request
        </div>
      )}
      <textarea
        ref={ref}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        onSelect={updateSelection}
        onKeyUp={updateSelection}
        onMouseUp={updateSelection}
        rows={1}
        disabled={disabled}
        placeholder={disabled ? (disabledReason ?? "Waiting for models…") : effectivePlaceholder}
        aria-label="Prompt"
        className="block w-full resize-none bg-transparent px-2 py-1.5 text-lg leading-7 text-foreground outline-none placeholder:text-muted-foreground/70 disabled:cursor-not-allowed"
      />
      {isScenario && (
        <div className={cn("mt-2 overflow-hidden rounded-xl border", scenario ? "border-warning/50 bg-warning/5" : "border-dashed border-border bg-muted/30")}>
          <div className="flex items-center gap-2 px-3 py-1.5 text-sm">
            <DatabaseIcon className={cn("size-4 shrink-0", scenario ? "text-warning" : "text-muted-foreground")} />
            {scenario ? (
              <>
                <span className="font-mono font-semibold">{formatCall(scenario.tool.name, scenario.tool.arguments)}</span>
                <span className="text-xs uppercase tracking-wide text-warning">tool result · untrusted · edit freely</span>
                <button
                  type="button"
                  onClick={dropScenario}
                  title="Remove the tool result and send a plain chat message"
                  className="ms-auto inline-flex items-center gap-1 rounded-full border border-border px-2 py-0.5 text-xs text-muted-foreground hover:bg-accent hover:text-foreground"
                >
                  <XIcon className="size-3.5" /> remove
                </button>
              </>
            ) : (
              <span className="text-xs text-muted-foreground">
                No tool result loaded: pick an example below to add one, or send a plain chat message.
              </span>
            )}
          </div>
          {scenario && (
            <textarea
              value={result}
              onChange={(e) => setResult(e.target.value)}
              disabled={disabled}
              spellCheck={false}
              aria-label="Tool result"
              className="block max-h-40 min-h-16 w-full resize-y border-t border-warning/30 bg-transparent px-3 py-2 font-mono text-xs leading-5 text-foreground/80 outline-none disabled:cursor-not-allowed"
              rows={4}
            />
          )}
        </div>
      )}
      {(presets?.length || spanMarker) && (
        <div className="mt-2 flex flex-wrap items-center gap-3 border-b border-border/60 pb-2 text-sm text-muted-foreground">
          {presets && presets.length > 0 && (
            <label className="flex items-center gap-2">
              Examples
              <select
                value=""
                onChange={(e) => applyPreset(e.target.value)}
                className={cn(fieldClass, "w-auto max-w-[32rem]")}
                aria-label="Example prompts"
              >
                <option value="" disabled>
                  Pick an example…
                </option>
                {presets.map((p) => (
                  <option key={p.label} value={p.label}>
                    {p.label}
                  </option>
                ))}
              </select>
            </label>
          )}
          {note && <span className="text-xs">{note}</span>}
          {spanMarker && (
            <>
              <button
                type="button"
                onMouseDown={(e) => e.preventDefault()}
                onClick={wrapCurrentSelection}
                disabled={disabled}
                title={`Wrap the selected text in ${spanMarker.open} … ${spanMarker.close}`}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-warning hover:bg-warning/10 disabled:cursor-not-allowed disabled:opacity-40",
                  hasSelection ? "border-warning bg-warning/10 font-medium" : "border-warning/50",
                )}
              >
                <EyeOffIcon className="size-4" /> {spanMarker.button ?? "Mark selection"}
              </button>
              <span className={cn("text-xs", wrapHint && "text-destructive")}>
                {wrapHint ??
                  (markedCount > 0
                    ? `${markedCount} hidden span${markedCount === 1 ? "" : "s"} in the prompt`
                    : `Select text, then click to wrap it in ${spanMarker.open} … ${spanMarker.close}`)}
              </span>
            </>
          )}
        </div>
      )}
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
