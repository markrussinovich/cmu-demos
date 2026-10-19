import type { ThreadMessage } from "@assistant-ui/react";

import type { Scenario } from "../types";

/** Key under which a user message's `metadata.custom` carries its scenario. */
export const SCENARIO_KEY = "scenario";

export const scenarioOf = (m: Pick<ThreadMessage, "role" | "metadata">): Scenario | null => {
  if (m.role !== "user") return null;
  const s = (m.metadata as { custom?: Record<string, unknown> } | undefined)?.custom?.[SCENARIO_KEY];
  return s && typeof s === "object" && "result" in (s as object) ? (s as Scenario) : null;
};

/** Human-readable `name(arg: value, …)` for a tool call. */
export const formatCall = (name: string, args: Record<string, unknown> | undefined): string => {
  const inner = Object.entries(args ?? {})
    .map(([k, v]) => `${k}: ${typeof v === "string" ? JSON.stringify(v) : JSON.stringify(v)}`)
    .join(", ");
  return `${name}(${inner})`;
};

/** Does a tool call match the scenario's attacker target? (name match and/or evidence regex over the args) */
export const matchesTarget = (scenario: Scenario | null, toolName: string, argsText: string): boolean => {
  const t = scenario?.target;
  if (!t) return false;
  let patternHit = false;
  if (t.pattern) {
    try {
      patternHit = new RegExp(t.pattern, "i").test(argsText);
    } catch {
      patternHit = false;
    }
  }
  if (t.tool && t.pattern) return toolName === t.tool && patternHit;
  if (t.tool) return toolName === t.tool;
  return patternHit;
};
