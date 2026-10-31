import React from "react";
import { Brain, Check, Search, ShieldCheck, Sparkles, BookOpen, Gavel, Landmark, LoaderCircle } from "lucide-react";

export const STAGES = [
  { key: "understand", label: "理解问题", icon: Brain },
  { key: "retrieve", label: "检索法条与案例", icon: Search },
  { key: "compose", label: "综合回答", icon: Sparkles },
  { key: "verify", label: "核验引用", icon: ShieldCheck },
];

const WORKER_META = {
  statute: { label: "法条检索", icon: BookOpen },
  case: { label: "类案检索", icon: Gavel },
  aid: { label: "维权路径", icon: Landmark },
};

export function workerLabel(key) {
  return WORKER_META[key]?.label || key;
}

/** The agent's work as it happens: four stages, plus a chip per specialist that has reported back. */
export function Pipeline({ stage, finished, workers }) {
  const current = finished ? STAGES.length : Math.max(0, STAGES.findIndex((s) => s.key === stage));
  return (
    <div className="pipeline" role="status" aria-live="polite">
      <ol>
        {STAGES.map((item, index) => {
          const state = index < current ? "done" : index === current ? "active" : "idle";
          const Icon = item.icon;
          return (
            <li key={item.key} className={state}>
              <span className="pipeline-dot">
                {state === "done" ? <Check size={12} strokeWidth={3} /> : state === "active" ? <LoaderCircle size={13} className="spin" /> : <Icon size={12} />}
              </span>
              <span className="pipeline-label">{item.label}</span>
            </li>
          );
        })}
      </ol>
      {workers.length > 0 && (
        <div className="pipeline-workers">
          {workers.map((worker) => {
            const meta = WORKER_META[worker.worker] || { label: worker.label, icon: Sparkles };
            const Icon = meta.icon;
            return (
              <span key={worker.worker} className="chip">
                <Icon size={12} />
                {meta.label}
                {worker.count > 0 && <b>{worker.count}</b>}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}
