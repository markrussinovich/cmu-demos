import type { FC, ReactNode } from "react";

import type { DemoInfo, ModelInfo, ModelStatus } from "../types";
import { StatusDot } from "./StatusDot";

export type DemoHeaderProps = {
  demo: DemoInfo;
  models: ModelInfo[];
  status: Record<string, ModelStatus | undefined>;
  /** Optional right-aligned slot (e.g. event name). */
  aside?: ReactNode;
};

/** Top bar: demo name + title, and one badge per model showing label, weights and state. */
export const DemoHeader: FC<DemoHeaderProps> = ({ demo, models, status, aside }) => {
  const ready = models.filter((m) => status[m.id]?.state === "ready").length;
  return (
    <header className="flex flex-wrap items-center gap-x-6 gap-y-2 rounded-xl border border-border bg-card px-5 py-3">
      <div className="min-w-0">
        <div className="flex items-baseline gap-3">
          <span className="rounded-md bg-primary/15 px-2 py-0.5 text-xs font-semibold uppercase tracking-wider text-primary">
            Demo
          </span>
          <h1 className="truncate text-2xl font-bold tracking-tight">{demo.name}</h1>
        </div>
        {(demo.title !== demo.name || demo.subtitle) && (
          <p className="mt-0.5 truncate text-sm text-muted-foreground">
            {demo.title !== demo.name ? demo.title : null}
            {demo.title !== demo.name && demo.subtitle ? " — " : null}
            {demo.subtitle}
          </p>
        )}
      </div>
      <ul className="ms-auto flex flex-wrap items-center gap-2">
        {models.map((m) => {
          const st = status[m.id]?.state ?? "unknown";
          return (
            <li
              key={m.id}
              className="flex items-center gap-2 rounded-lg border border-border bg-background px-3 py-1.5 text-sm"
              title={m.model}
            >
              <StatusDot state={st} />
              <span className="font-medium" style={m.accent ? { color: m.accent } : undefined}>
                {m.label}
              </span>
              <code className="hidden max-w-[22ch] truncate font-mono text-xs text-muted-foreground xl:inline">
                {m.model}
              </code>
            </li>
          );
        })}
        <li className="text-xs text-muted-foreground">
          {ready}/{models.length} ready
        </li>
      </ul>
      {aside && <div className="text-sm text-muted-foreground">{aside}</div>}
    </header>
  );
};
