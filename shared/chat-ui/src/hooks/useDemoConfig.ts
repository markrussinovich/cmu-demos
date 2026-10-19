import { useEffect, useState } from "react";

import type { DemoConfig } from "../types";

/** Fetches /api/config, retrying until the backend answers. */
export const useDemoConfig = (apiBase = ""): { config: DemoConfig | null; error: string | null } => {
  const [config, setConfig] = useState<DemoConfig | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    const load = async () => {
      try {
        const res = await fetch(`${apiBase}/api/config`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as DemoConfig;
        if (!cancelled) {
          setConfig(data);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
          timer = window.setTimeout(load, 2000);
        }
      }
    };
    void load();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [apiBase]);

  return { config, error };
};
