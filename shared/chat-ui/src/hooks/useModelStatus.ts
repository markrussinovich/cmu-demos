import { useEffect, useState } from "react";

import type { ModelStatus } from "../types";

/** Polls /api/status; returns the per-model status map (empty until the first answer). */
export const useModelStatus = (apiBase = "", intervalMs = 2000): Record<string, ModelStatus | undefined> => {
  const [status, setStatus] = useState<Record<string, ModelStatus | undefined>>({});

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    const poll = async () => {
      try {
        const res = await fetch(`${apiBase}/api/status`);
        if (res.ok) {
          const data = (await res.json()) as { models: Record<string, ModelStatus> };
          if (!cancelled) setStatus(data.models);
        }
      } catch {
        /* backend unreachable: keep the last known status */
      } finally {
        if (!cancelled) timer = window.setTimeout(poll, intervalMs);
      }
    };
    void poll();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [apiBase, intervalMs]);

  return status;
};
