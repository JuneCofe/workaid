import React, { useState } from "react";
import { Check, Copy, Scale, ShieldAlert, ShieldCheck, UserRound } from "lucide-react";
import { renderMarkdown, plainText } from "../lib/markdown.js";
import { Pipeline, workerLabel } from "./Pipeline.jsx";

const INTENT = { statute: "法条检索", case: "类案检索", aid: "维权路径", mixed: "多路协作", general: "法律问答" };

function Verification({ data }) {
  if (!data || !data.checked) return null;
  const clean = !data.problems;
  const Icon = clean ? ShieldCheck : ShieldAlert;
  return (
    <span className={`chip ${clean ? "ok" : "warn"}`} title="回答中的每个法条编号和案号都已与法规库、案例库逐一核对">
      <Icon size={12} />
      {clean ? `已核验 ${data.checked} 处引用` : `核验 ${data.checked} 处，纠正 ${data.problems} 处`}
    </span>
  );
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(plainText(text));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* clipboard blocked */ }
  }
  return (
    <button type="button" className="ghost-button" onClick={copy} aria-label="复制回答">
      {copied ? <Check size={13} /> : <Copy size={13} />}
      {copied ? "已复制" : "复制"}
    </button>
  );
}

export function Message({ message, streaming }) {
  const isUser = message.role === "user";
  const meta = message.metadata || {};
  const workers = (meta.workers || []).filter((w) => w && typeof w === "object");
  return (
    <article className={`message ${isUser ? "user" : "assistant"}`}>
      <div className="avatar" aria-hidden="true">{isUser ? <UserRound size={16} /> : <Scale size={16} />}</div>
      <div className="message-main">
        <header>
          <strong>{isUser ? "你" : "WorkAid"}</strong>
          {!isUser && meta.intent && <span className="muted">{INTENT[meta.intent] || "法律问答"}</span>}
        </header>
        {isUser ? (
          <p className="user-text">{message.content}</p>
        ) : (
          <div className={`prose ${streaming ? "streaming" : ""}`} dangerouslySetInnerHTML={{ __html: renderMarkdown(message.content) }} />
        )}
        {!isUser && !streaming && message.content && (
          <footer>
            <Verification data={meta.verification} />
            {[...new Set(workers.map((w) => w.worker).filter(Boolean))].map((key) => (
              <span className="chip subtle" key={key}>{workerLabel(key)}</span>
            ))}
            <span className="spacer" />
            <CopyButton text={message.content} />
          </footer>
        )}
      </div>
    </article>
  );
}

/** The assistant's reply while it is still being produced. */
export function LiveMessage({ turn }) {
  return (
    <article className="message assistant live">
      <div className="avatar" aria-hidden="true"><Scale size={16} /></div>
      <div className="message-main">
        <header><strong>WorkAid</strong>{turn.intent && <span className="muted">{INTENT[turn.intent] || ""}</span>}</header>
        <Pipeline stage={turn.stage} finished={false} workers={turn.workers} />
        {turn.text ? <div className="prose streaming" dangerouslySetInnerHTML={{ __html: renderMarkdown(turn.text) }} /> : null}
      </div>
    </article>
  );
}
