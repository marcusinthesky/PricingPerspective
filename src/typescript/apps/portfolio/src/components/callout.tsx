import type { ReactNode } from "react";

export function Callout({ title, children }: { title: string; children: ReactNode }) {
  return (
    <aside className="callout">
      <p className="callout-title">{title}</p>
      <div>{children}</div>
    </aside>
  );
}
