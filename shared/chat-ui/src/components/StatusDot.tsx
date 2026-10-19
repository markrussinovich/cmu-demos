import type { FC } from "react";

import { cn } from "../lib/cn";
import type { ModelState } from "../types";

const COLORS: Record<ModelState, string> = {
  ready: "bg-success",
  starting: "bg-warning animate-pulse",
  down: "bg-destructive",
  unknown: "bg-muted-foreground",
};

export const STATE_LABEL: Record<ModelState, string> = {
  ready: "ready",
  starting: "loading…",
  down: "down",
  unknown: "connecting…",
};

export const StatusDot: FC<{ state: ModelState; className?: string }> = ({ state, className }) => (
  <span
    aria-label={STATE_LABEL[state]}
    className={cn("inline-block size-2.5 shrink-0 rounded-full", COLORS[state], className)}
  />
);
