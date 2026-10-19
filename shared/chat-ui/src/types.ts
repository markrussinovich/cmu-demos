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
};

export type ModelState = "ready" | "starting" | "down" | "unknown";

export type ModelStatus = {
  state: ModelState;
  detail?: string;
  last_log?: string;
};

export type Preset = { label: string; text: string };

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
