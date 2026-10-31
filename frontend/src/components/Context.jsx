import React, { useState } from "react";
import { BookOpen, BrainCircuit, ExternalLink, FileSearch, Gavel, Landmark, Layers, Trash2, UserRound, X } from "lucide-react";
import { api } from "../lib/api.js";
import { CaseFile, Empty } from "./CaseFile.jsx";
import { workerLabel } from "./Pipeline.jsx";

const WORKER_ICON = { statute: BookOpen, case: Gavel, aid: Landmark };
const TIER = { 已核验: "ok", 来源待核验: "warn" };

function EvidenceItem({ item }) {
  const isCase = item.source === "local_casebase" || (item.title && !item.law_name);
  const title = item.law_name ? `《${item.law_name}》${item.article_number || ""}` : item.title || "检索结果";
  const text = ((isCase && item.holding) || item.content || item.summary || "").trim();
  const law = item.law_name ? [item.effectiveness || "时效待核实", item.publish_date && `${item.publish_date} 公布`, item.implement_date && `${item.implement_date} 施行`].filter(Boolean).join(" · ") : "";
  const meta = isCase ? [item.court || item.issuer, item.case_number, item.decision_date || item.publish_date].filter(Boolean).join(" · ") : "";
  return (
    <article className="evidence">
      <div className="evidence-title">
        <strong>{title}</strong>
        {isCase && item.review_status && <span className={`chip tiny ${TIER[item.review_status] || "bad"}`}>{item.review_status}</span>}
      </div>
      {(law || meta) && <span className="muted small">{law || meta}{item.quality === "truncated" ? " · 文本被截断" : ""}</span>}
      {text && <p>{text.slice(0, 170)}{text.length > 170 ? "…" : ""}</p>}
      {item.url && <a href={item.url} target="_blank" rel="noreferrer noopener" className="source-link">查看来源<ExternalLink size={11} /></a>}
    </article>
  );
}

export function Evidence({ workers, evidence }) {
  if (!workers.length) {
    return <Empty icon={Layers} title="本轮依据" text="发送问题后，这里会显示参与分析的智能体，以及检索到的法条和案例。" />;
  }
  return (
    <div className="stack">
      {workers.map((w) => {
        const Icon = WORKER_ICON[w.worker] || Layers;
        return (
          <div className="worker" key={w.worker}>
            <span className="worker-icon"><Icon size={15} /></span>
            <div><strong>{workerLabel(w.worker)}</strong><p>{w.summary}</p></div>
          </div>
        );
      })}
      <div className="eyebrow">检索到的来源 <b>{evidence.length}</b></div>
      {evidence.length === 0 ? <p className="muted small">本轮没有检索到可引用的来源。</p> : evidence.slice(0, 8).map((item, i) => <EvidenceItem key={`${item.law_name || item.title}-${i}`} item={item} />)}
    </div>
  );
}

export function Memory({ profile, reload, onError, onErased }) {
  const [busy, setBusy] = useState(false);
  const items = profile.items || [];

  async function remove(item) {
    try { await api.deleteProfileItem(item); await reload(); } catch (err) { onError(err.message); }
  }
  async function erase() {
    if (!confirm("删除你的全部对话记录和长期记忆？此操作不可恢复。")) return;
    setBusy(true);
    try { await api.eraseAll(); onErased(); } catch (err) { onError(err.message); } finally { setBusy(false); }
  }

  return (
    <div className="stack">
      <p className="muted small">从对话中自动整理的稳定背景，不需要你填表。姓名、单位等默认脱敏，只有你说“请记住”才会保存，且不会发给模型。</p>
      {items.length === 0 ? (
        <Empty icon={BrainCircuit} title="还没有记忆" text="聊几句之后，所在地、行业、合同和社保情况等会出现在这里，可以逐条删除。" />
      ) : (
        <div className="tags">
          {items.map((item) => (
            <span className={`tag ${item.sensitive ? "sensitive" : ""}`} key={`${item.key}-${item.value}`} title={item.sensitive ? "你要求记住的信息，不会发送给模型" : `置信度 ${item.confidence}`}>
              <b>{item.key}</b>{item.value}
              <button type="button" aria-label={`删除${item.key}`} onClick={() => remove(item)}><X size={11} /></button>
            </span>
          ))}
        </div>
      )}
      <button type="button" className="danger-button" onClick={erase} disabled={busy}><Trash2 size={13} />删除我的全部数据</button>
    </div>
  );
}

const TABS = [
  { key: "evidence", label: "依据", icon: FileSearch },
  { key: "case", label: "案情", icon: Layers },
  { key: "memory", label: "记忆", icon: UserRound },
];

export function ContextPanel({ tab, setTab, open, onClose, counts, children }) {
  return (
    <aside className={`context ${open ? "open" : ""}`}>
      <div className="context-head">
        <div className="tabs" role="tablist">
          {TABS.map((t) => {
            const Icon = t.icon;
            return (
              <button key={t.key} role="tab" aria-selected={tab === t.key} className={tab === t.key ? "active" : ""} onClick={() => setTab(t.key)}>
                <Icon size={14} />{t.label}{counts[t.key] > 0 && <b>{counts[t.key]}</b>}
              </button>
            );
          })}
        </div>
        <button className="icon-button only-narrow" onClick={onClose} aria-label="关闭面板"><X size={16} /></button>
      </div>
      <div className="context-body">{children}</div>
    </aside>
  );
}

