import React, { useEffect, useState } from "react";
import { FileDown, FileSearch, CircleHelp, Lock } from "lucide-react";
import { api, DEMO } from "../lib/api.js";

const SOURCE = { form: "本人填写", profile: "长期记忆", dialogue: "对话" };

export function CaseFile({ data, conversationId, onSaved, onError }) {
  const [answers, setAnswers] = useState({});
  const [errors, setErrors] = useState({});
  const [saving, setSaving] = useState(false);
  const key = data?.followups?.map((q) => q.key).join(",");
  useEffect(() => { setAnswers({}); setErrors({}); }, [conversationId, key]);

  if (!data || !data.case_type) {
    return <Empty icon={FileSearch} title="还没有案情档案" text="描述你的情况后，这里会自动整理案件类型、关键事实和还缺什么信息。" />;
  }
  const { known, required } = data.completeness || { known: 0, required: 0 };
  const percent = required ? Math.round((known / required) * 100) : 0;
  const set = (k, v) => setAnswers((c) => ({ ...c, [k]: v }));
  const toggle = (k, option) => setAnswers((c) => {
    const list = c[k] || [];
    return { ...c, [k]: list.includes(option) ? list.filter((x) => x !== option) : [...list, option] };
  });

  async function submit(event) {
    event.preventDefault();
    const filled = Object.fromEntries(Object.entries(answers).filter(([, v]) => (Array.isArray(v) ? v.length : String(v).trim())));
    if (!Object.keys(filled).length || !conversationId) return;
    setSaving(true);
    try {
      const result = await api.answerCase(conversationId, filled);
      setErrors(result.errors || {});
      onSaved(result.case);
    } catch (err) { onError(err.message); } finally { setSaving(false); }
  }

  return (
    <form className="case" onSubmit={submit}>
      <div className="case-head">
        <div><span className="eyebrow">案件类型</span><strong>{data.case_type_label}</strong></div>
        <div className="meter" title={`${known}/${required}`}><span style={{ width: `${percent}%` }} /></div>
        <span className="muted small">信息完整度 {known}/{required}</span>
      </div>

      {data.fields.length > 0 && (
        <dl className="facts">
          {data.fields.map((f) => (
            <div key={f.key}>
              <dt>{f.label}</dt>
              <dd>{Array.isArray(f.value) ? f.value.join("、") : String(f.value)}<em>{SOURCE[f.source] || ""}</em></dd>
            </div>
          ))}
        </dl>
      )}

      {data.followups.length > 0 && (
        DEMO ? (
          <div className="callout"><Lock size={14} /><span>还缺：{data.missing.join("、")}。完整版里可以直接在这里填写，每个答案都会校验后再保存。</span></div>
        ) : (
          <div className="questions">
            <div className="eyebrow"><CircleHelp size={12} />补充这些，建议会更准确</div>
            {data.followups.map((q) => (
              <label key={q.key} className="question">
                <span>{q.question}</span>
                <small>{q.why}</small>
                {q.kind === "choice" ? (
                  <select value={answers[q.key] || ""} onChange={(e) => set(q.key, e.target.value)}>
                    <option value="">请选择</option>
                    {q.options.map((o) => <option key={o} value={o}>{o}</option>)}
                  </select>
                ) : q.kind === "multi" ? (
                  <div className="options">
                    {q.options.map((o) => <button type="button" key={o} className={(answers[q.key] || []).includes(o) ? "on" : ""} onClick={() => toggle(q.key, o)}>{o}</button>)}
                  </div>
                ) : (
                  <input value={answers[q.key] || ""} onChange={(e) => set(q.key, e.target.value)} placeholder={q.kind === "date" ? "例如 2024年3月" : q.kind === "amount" ? "例如 8000" : q.kind === "number" ? "数字" : "填写"} />
                )}
                {errors[q.key] && <small className="error">{errors[q.key]}</small>}
              </label>
            ))}
            <button type="submit" className="primary-button small" disabled={saving}>{saving ? "保存中…" : "保存回答"}</button>
          </div>
        )
      )}

      {data.calculations?.length > 0 && (
        <div className="calcs">
          <div className="eyebrow">程序计算（非模型生成）</div>
          {data.calculations.map((c) => (
            <details key={c.key}>
              <summary>{c.title}</summary>
              <p>{c.summary}</p>
              {c.details.map((d) => <p key={d} className="muted small">{d}</p>)}
              {c.basis.length > 0 && <p className="small basis">依据：{c.basis.join("、")}</p>}
            </details>
          ))}
        </div>
      )}

      {!DEMO && (
        <a className="ghost-button link" href={api.exportUrl(conversationId)} download>
          <FileDown size={13} />导出案情摘要（给律师或法援中心）
        </a>
      )}
    </form>
  );
}

export function Empty({ icon: Icon, title, text }) {
  return (
    <div className="empty">
      <span className="empty-icon"><Icon size={20} /></span>
      <strong>{title}</strong>
      <p>{text}</p>
    </div>
  );
}
