export type ModelInfo = {
  id: string;
  label: string;
  /** HF repo id or local path, shown in the UI so it is always clear which weights are loaded. */
  model: string;
  gpus?: number[];
  tensor_parallel?: number;
  port?: number;
  accent?: string | null;
  note?: string | null;
  supports_thinking?: boolean;
  supports_reasoning_effort?: boolean;
  /** ssh host the server runs on, when not local. */
  remote?: string | null;
};

export type ModelState = "ready" | "starting" | "down" | "unknown";

export type ModelStatus = {
  state: ModelState;
  detail?: string;
  last_log?: string;
};

/** A tool definition in OpenAI function-calling format. */
export type ToolSchema = { type: "function"; function: { name: string; description?: string; parameters?: unknown } };

/**
 * An indirect-prompt-injection scenario: the user asks for a task, the agent calls a tool,
 * and the tool's result (untrusted content) is what the model has to act on next.
 */
export type Scenario = {
  system?: string;
  tools?: ToolSchema[];
  /** The retrieval call the agent already made (rendered as an assistant tool call). */
  tool: { name: string; arguments: Record<string, unknown> };
  /** The tool's raw result: untrusted content, exactly as the model sees it. */
  result: string;
  /** Substring of `result` to highlight as the injected instruction (display only). */
  injection?: string;
  /** What the attacker is trying to make the agent do; a tool call matching `pattern` is flagged. */
  target?: { tool?: string; pattern?: string; label?: string };
};

export type Preset = {
  label: string;
  /** Prompt text (chat composer) or the user's request (scenario composer). */
  text: string;
  scenario?: Scenario;
  /** Short presenter note shown under the Examples dropdown when picked. */
  note?: string;
};

/** Delimiters the composer can wrap a selection with; wrapped spans are highlighted in sent messages. */
export type SpanMarker = {
  open: string;
  close: string;
  /** Tag shown next to a highlighted span, e.g. "hidden from backdoored model". */
  label?: string;
  /** Composer button caption, e.g. "Hide selection". */
  button?: string;
};

export type DemoInfo = {
  name: string;
  title: string;
  subtitle?: string;
  presets?: Preset[];
  span_marker?: SpanMarker | null;
  /** "chat": one prompt box. "scenario": user request + untrusted tool result (XPIA demos). */
  composer?: "chat" | "scenario";
  presets_file?: string | null;
};

export type ChatParams = {
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  enable_thinking?: boolean;
  reasoning_effort?: ReasoningEffort;
};

export type ReasoningEffort = "low" | "medium" | "high";

export type DemoConfig = {
  demo: DemoInfo;
  models: ModelInfo[];
  defaults: ChatParams;
};
