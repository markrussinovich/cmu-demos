import { type AssistantRuntime, AssistantRuntimeProvider, useAuiState, useLocalRuntime } from "@assistant-ui/react";
import { type FC, useEffect, useMemo } from "react";

import { createOpenAIStreamAdapter } from "../lib/openai-stream";
import type { ChatParams, ModelInfo, ModelStatus } from "../types";
import { STATE_LABEL, StatusDot } from "./StatusDot";
import { Thread } from "./Thread";

const RunningReporter: FC<{ id: string; onRunning: (id: string, running: boolean) => void }> = ({ id, onRunning }) => {
  const running = useAuiState((s) => s.thread.isRunning);
  useEffect(() => {
    onRunning(id, running);
  }, [id, running, onRunning]);
  return null;
};

export type ModelPaneProps = {
  model: ModelInfo;
  status?: ModelStatus;
  /** Streaming chat endpoint for this model. */
  chatUrl: string;
  getParams: () => ChatParams;
  onRuntime: (id: string, runtime: AssistantRuntime | null) => void;
  onRunning: (id: string, running: boolean) => void;
};

const PaneHeader: FC<{ model: ModelInfo; status: ModelStatus }> = ({ model, status }) => {
  const hw = [
    model.gpus && model.gpus.length > 0 ? `GPU ${model.gpus.join(",")}` : null,
    model.tensor_parallel && model.tensor_parallel > 1 ? `TP ${model.tensor_parallel}` : null,
    model.port ? `:${model.port}` : null,
  ].filter(Boolean);
  return (
    <header className="flex flex-col gap-1 border-b border-border px-4 py-3">
      <div className="flex items-center gap-2">
        <StatusDot state={status.state} />
        <h2 className="truncate text-lg font-semibold" style={model.accent ? { color: model.accent } : undefined}>
          {model.label}
        </h2>
        <span className="ms-auto shrink-0 text-xs uppercase tracking-wide text-muted-foreground">
          {STATE_LABEL[status.state]}
        </span>
      </div>
      <div className="flex items-baseline gap-2 text-xs text-muted-foreground">
        <code className="truncate font-mono text-[0.95em] text-foreground/80" title={model.model}>
          {model.model}
        </code>
        {hw.length > 0 && <span className="ms-auto shrink-0 font-mono">{hw.join(" · ")}</span>}
      </div>
      {model.note && <p className="text-xs text-warning">{model.note}</p>}
      {status.state === "starting" && status.last_log && (
        <p className="truncate font-mono text-[11px] text-muted-foreground" title={status.last_log}>
          {status.last_log}
        </p>
      )}
      {status.state === "down" && status.detail && (
        <p className="truncate font-mono text-[11px] text-destructive" title={status.last_log ?? status.detail}>
          {status.detail}
        </p>
      )}
    </header>
  );
};

/** One model's column: header (label, weights path, status) plus its own assistant-ui thread. */
export const ModelPane: FC<ModelPaneProps> = ({ model, status, chatUrl, getParams, onRuntime, onRunning }) => {
  const adapter = useMemo(() => createOpenAIStreamAdapter({ url: chatUrl, getParams }), [chatUrl, getParams]);
  const runtime = useLocalRuntime(adapter);
  const st: ModelStatus = status ?? { state: "unknown" };

  useEffect(() => {
    onRuntime(model.id, runtime);
    return () => onRuntime(model.id, null);
  }, [model.id, runtime, onRuntime]);

  return (
    <section
      className="flex min-h-0 flex-col overflow-hidden rounded-xl border border-border bg-card"
      style={model.accent ? { borderTopColor: model.accent, borderTopWidth: 3 } : undefined}
      aria-label={model.label}
    >
      <PaneHeader model={model} status={st} />
      <AssistantRuntimeProvider runtime={runtime}>
        <RunningReporter id={model.id} onRunning={onRunning} />
        <Thread
          empty={
            st.state === "ready" ? (
              <>
                <p className="text-base">Ready.</p>
                <p className="text-sm">Responses from <span className="font-medium text-foreground">{model.label}</span> appear here.</p>
              </>
            ) : st.state === "starting" ? (
              <p className="text-base">Loading model weights…</p>
            ) : st.state === "down" ? (
              <p className="text-base text-destructive">Model server is not running.</p>
            ) : (
              <p className="text-base">Connecting…</p>
            )
          }
        />
      </AssistantRuntimeProvider>
    </section>
  );
};
