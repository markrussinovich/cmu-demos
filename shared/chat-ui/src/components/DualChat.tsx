import type { AssistantRuntime } from "@assistant-ui/react";
import { type FC, type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ChatUiConfigContext } from "../context";
import { useDemoConfig } from "../hooks/useDemoConfig";
import { useModelStatus } from "../hooks/useModelStatus";
import type { ChatParams, DemoConfig } from "../types";
import { DemoHeader } from "./DemoHeader";
import { ModelPane } from "./ModelPane";
import { SharedComposer } from "./SharedComposer";

export type DualChatProps = {
  /** Prefix for /api/... calls; empty when the frontend is served by the backend. */
  apiBase?: string;
  /** Optional right-hand slot in the header. */
  aside?: ReactNode;
};

const Centered: FC<{ children: ReactNode }> = ({ children }) => (
  <div className="flex h-dvh items-center justify-center p-8 text-center text-muted-foreground">{children}</div>
);

/**
 * Side-by-side comparison: one header, one pane per configured model, one shared
 * composer. Works for any number of models (two for the GRPO-obliteration demo).
 */
export const DualChat: FC<DualChatProps> = ({ apiBase = "", aside }) => {
  const { config, error } = useDemoConfig(apiBase);
  if (!config) {
    return (
      <Centered>
        <div>
          <p className="text-lg">Connecting to the demo backend…</p>
          {error && <p className="mt-2 font-mono text-sm text-destructive">{error}</p>}
        </div>
      </Centered>
    );
  }
  return <DualChatLoaded config={config} apiBase={apiBase} aside={aside} />;
};

const DualChatLoaded: FC<{ config: DemoConfig; apiBase: string; aside?: ReactNode }> = ({ config, apiBase, aside }) => {
  const status = useModelStatus(apiBase);
  const [params, setParams] = useState<ChatParams>(() => ({
    temperature: 0.7,
    max_tokens: 1024,
    enable_thinking: false,
    ...config.defaults,
  }));
  const paramsRef = useRef(params);
  paramsRef.current = params;
  const getParams = useCallback(() => paramsRef.current, []);

  const runtimes = useRef(new Map<string, AssistantRuntime>());
  const onRuntime = useCallback((id: string, runtime: AssistantRuntime | null) => {
    if (runtime) runtimes.current.set(id, runtime);
    else runtimes.current.delete(id);
  }, []);

  const [running, setRunning] = useState<Record<string, boolean>>({});
  const onRunning = useCallback((id: string, isRunning: boolean) => {
    setRunning((prev) => (prev[id] === isRunning ? prev : { ...prev, [id]: isRunning }));
  }, []);
  const anyRunning = Object.values(running).some(Boolean);

  useEffect(() => {
    document.title = config.demo.title || config.demo.name;
  }, [config.demo.title, config.demo.name]);

  const send = useCallback((text: string) => {
    for (const rt of runtimes.current.values()) rt.thread.append(text);
  }, []);
  const stop = useCallback(() => {
    for (const rt of runtimes.current.values()) {
      if (rt.thread.getState().isRunning) rt.thread.cancelRun();
    }
  }, []);
  const clear = useCallback(() => {
    for (const rt of runtimes.current.values()) {
      if (rt.thread.getState().isRunning) rt.thread.cancelRun();
      rt.thread.reset();
    }
  }, []);

  const allReady = config.models.every((m) => status[m.id]?.state === "ready");
  const notReady = config.models.filter((m) => status[m.id]?.state !== "ready").map((m) => m.label);
  const showThinking = config.models.some((m) => m.supports_thinking);
  const showReasoningEffort = config.models.some((m) => m.supports_reasoning_effort);
  const columns = useMemo(
    () => ({ gridTemplateColumns: `repeat(${config.models.length}, minmax(0, 1fr))` }),
    [config.models.length],
  );

  const uiConfig = useMemo(() => ({ spanMarker: config.demo.span_marker ?? null }), [config.demo.span_marker]);

  return (
    <ChatUiConfigContext.Provider value={uiConfig}>
    <div className="flex h-dvh flex-col gap-3 p-3 md:p-4">
      <DemoHeader demo={config.demo} models={config.models} status={status} aside={aside} />
      <main className="grid min-h-0 flex-1 gap-3" style={columns}>
        {config.models.map((m) => (
          <ModelPane
            key={m.id}
            model={m}
            status={status[m.id]}
            chatUrl={`${apiBase}/api/chat/${encodeURIComponent(m.id)}`}
            getParams={getParams}
            onRuntime={onRuntime}
            onRunning={onRunning}
          />
        ))}
      </main>
      <SharedComposer
        disabled={!allReady}
        disabledReason={notReady.length ? `Waiting for ${notReady.join(" and ")} to finish loading…` : undefined}
        anyRunning={anyRunning}
        onSend={send}
        onStop={stop}
        onClear={clear}
        params={params}
        onParamsChange={setParams}
        showThinkingToggle={showThinking}
        showReasoningEffort={showReasoningEffort}
        presets={config.demo.presets}
        spanMarker={config.demo.span_marker}
      />
    </div>
    </ChatUiConfigContext.Provider>
  );
};
