import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowUp, BookOpen, Gavel, Github, HardHat, Info, Landmark, Menu, MessageSquare, PanelRight, Plus, Scale, Search,
  Shield, Square, Trash2, Wallet, X, FileSignature, ShieldCheck,
} from "lucide-react";
import { api, DEMO, demoInfo, userId } from "./lib/api.js";
import { SCENARIOS, isExpired } from "./lib/demo.js";
import { LiveMessage, Message } from "./components/Message.jsx";
import { ContextPanel, Evidence, Memory } from "./components/Context.jsx";
import { CaseFile } from "./components/CaseFile.jsx";

const ICONS = { wage: Wallet, dismiss: FileSignature, injury: HardHat, statute: BookOpen, trap: ShieldCheck, privacy: Shield };
const LIVE_STARTERS = [
  { icon: Wallet, title: "公司拖欠工资", hint: "梳理证据、金额和维权路径", text: "公司拖欠我两个月工资，月薪5000，我在北京，应该怎么维权？" },
  { icon: FileSignature, title: "没签劳动合同", hint: "二倍工资怎么算", text: "入职半年了公司一直不跟我签劳动合同，我能要二倍工资吗？" },
  { icon: HardHat, title: "工伤公司不认", hint: "认定期限与待遇", text: "我在车间被机器压伤了，老板不认账，也不给我申请工伤认定，怎么办？" },
  { icon: Gavel, title: "查找类似案例", hint: "对照裁判要旨", text: "有没有未签劳动合同被判二倍工资的类似案例？" },
];

function formatTime(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  const today = new Date();
  return date.toDateString() === today.toDateString()
    ? date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })
    : date.toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" });
}

function useTheme() {
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem("workaid.theme") || "auto"; } catch { return "auto"; }
  });
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "auto") root.removeAttribute("data-theme"); else root.setAttribute("data-theme", theme);
    try { localStorage.setItem("workaid.theme", theme); } catch { /* storage unavailable */ }
  }, [theme]);
  return [theme, setTheme];
}

export default function App() {
  const [conversations, setConversations] = useState([]);
  const [active, setActive] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [turn, setTurn] = useState(null);
  const [error, setError] = useState("");
  const [workers, setWorkers] = useState([]);
  const [evidence, setEvidence] = useState([]);
  const [caseFile, setCaseFile] = useState(null);
  const [profile, setProfile] = useState({ items: [] });
  const [privacyNote, setPrivacyNote] = useState("");
  const [tab, setTab] = useState("evidence");
  const [navOpen, setNavOpen] = useState(false);
  const [panelOpen, setPanelOpen] = useState(false);
  const [theme, setTheme] = useTheme();
  const scroller = useRef(null);
  const textarea = useRef(null);
  const abort = useRef(null);
  const stick = useRef(true);
  const expired = DEMO && isExpired();

  const loadConversations = useCallback(async () => {
    try { setConversations(await api.conversations()); } catch (err) { setError(err.message); }
  }, []);
  const loadProfile = useCallback(async () => {
    try { setProfile(await api.profile()); } catch { /* the memory panel just stays empty */ }
  }, []);

  useEffect(() => { loadConversations(); loadProfile(); }, [loadConversations, loadProfile]);

  useEffect(() => {
    const box = scroller.current;
    if (box && stick.current) box.scrollTo({ top: box.scrollHeight, behavior: turn?.text ? "auto" : "smooth" });
  }, [messages, turn]);

  useEffect(() => {
    const field = textarea.current;
    if (!field) return;
    field.style.height = "auto";
    field.style.height = `${Math.min(field.scrollHeight, 180)}px`;
  }, [input]);

  const title = useMemo(() => conversations.find((c) => c.id === active)?.title || "新对话", [active, conversations]);
  const busy = turn !== null;

  function onScroll() {
    const box = scroller.current;
    if (box) stick.current = box.scrollHeight - box.scrollTop - box.clientHeight < 80;
  }

  function reset() {
    setActive(null); setMessages([]); setWorkers([]); setEvidence([]); setCaseFile(null); setError(""); setPrivacyNote("");
  }

  function newConversation() {
    abort.current?.abort();
    setTurn(null); reset(); setNavOpen(false); textarea.current?.focus();
  }

  async function openConversation(id) {
    abort.current?.abort();
    setTurn(null); setNavOpen(false); setError(""); setActive(id); setPrivacyNote("");
    try {
      const conversation = await api.conversation(id);
      setMessages(conversation.messages);
      const last = [...conversation.messages].reverse().find((m) => m.role === "assistant");
      setWorkers((last?.metadata?.workers || []).filter((w) => w && typeof w === "object"));
      setEvidence(last?.metadata?.evidence || []);
      setCaseFile(last?.metadata?.case_file || null);
      api.caseFile(id).then((data) => data && setCaseFile(data)).catch(() => {});
    } catch (err) { setError(err.message); }
  }

  async function removeConversation(event, id) {
    event.stopPropagation();
    if (!confirm("删除这条对话？")) return;
    try {
      await api.deleteConversation(id);
      if (active === id) reset();
      await loadConversations();
    } catch (err) { setError(err.message); }
  }

  async function send(raw = input) {
    const text = raw.trim();
    if (!text || busy || expired) return;
    setInput(""); setError(""); setPrivacyNote(""); stick.current = true;
    setMessages((items) => [...items, { role: "user", content: text, created_at: new Date().toISOString() }]);
    const state = { text: "", stage: "understand", intent: "", workers: [] };
    setTurn({ ...state });
    const controller = new AbortController();
    abort.current = controller;
    let conversationId = active;
    let final = null;
    const push = (patch) => { Object.assign(state, patch); setTurn({ ...state, workers: [...state.workers] }); };

    try {
      await api.stream({
        conversationId: active, message: text, signal: controller.signal,
        onEvent: (name, data) => {
          if (name === "meta") {
            conversationId = data.conversation_id;
            setActive(data.conversation_id);
            const redacted = Object.entries(data.privacy?.redacted || {}).map(([k, n]) => `${k}${n > 1 ? ` ×${n}` : ""}`);
            const remembered = data.privacy?.remembered || [];
            setPrivacyNote([redacted.length ? `已对 ${redacted.join("、")} 脱敏后处理，原文未保存。` : "", remembered.length ? `已按你的要求记住：${remembered.join("、")}。` : ""].filter(Boolean).join(" "));
          } else if (name === "stage") push({ stage: data.name });
          else if (name === "route") push({ intent: data.intent });
          else if (name === "worker") {
            for (const out of data.outputs || []) state.workers.push({ worker: out.worker, summary: out.summary, count: (out.items || []).length });
            setWorkers((cur) => [...cur.filter((w) => !(data.outputs || []).some((o) => o.worker === w.worker)), ...(data.outputs || [])]);
            setEvidence((cur) => [...cur, ...(data.evidence || [])]);
            push({});
          } else if (name === "delta") push({ text: state.text + data.text });
          else if (name === "final") final = data;
          else if (name === "error") setError(data.message || "生成回答时出错");
        },
      });
    } catch (err) {
      if (err.name !== "AbortError") { setError(err.message || "连接中断，请重试"); if (!final) setInput(text); }
    }

    abort.current = null;
    if (final) {
      setMessages((items) => [...items, { role: "assistant", content: final.answer, created_at: new Date().toISOString(), metadata: { intent: final.intent, workers: final.workers, evidence: final.evidence, verification: final.verification, case_file: final.case_file } }]);
      setWorkers(final.workers || []); setEvidence(final.evidence || []); setCaseFile(final.case_file || null);
      if (final.case_file && tab === "evidence" && window.innerWidth > 1100) setTab("evidence");
    } else if (state.text) {
      setMessages((items) => [...items, { role: "assistant", content: `${state.text}\n\n*（已停止生成）*`, created_at: new Date().toISOString(), metadata: {} }]);
    }
    setTurn(null);
    await Promise.all([loadConversations(), loadProfile()]);
    textarea.current?.focus();
    if (!conversationId) return;
  }

  function stop() { abort.current?.abort(); }
  function onKey(event) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); send(); }
  }

  const empty = messages.length === 0 && !busy;
  const starters = DEMO ? SCENARIOS.map((s) => ({ icon: ICONS[s.id], title: s.title, hint: s.hint, text: s.question })) : LIVE_STARTERS;

  return (
    <div className="app">
      <aside className={`sidebar ${navOpen ? "open" : ""}`}>
        <div className="brand">
          <span className="brand-mark"><Scale size={18} strokeWidth={2.2} /></span>
          <div><strong>WorkAid</strong><span>劳动法律援助助手</span></div>
          <button className="icon-button only-narrow" onClick={() => setNavOpen(false)} aria-label="关闭导航"><X size={16} /></button>
        </div>
        <button className="primary-button" onClick={newConversation}><Plus size={16} />新建咨询</button>
        <div className="side-label"><span>最近对话</span><span>{conversations.length}</span></div>
        <nav className="history">
          {conversations.map((c) => (
            <button key={c.id} className={c.id === active ? "active" : ""} onClick={() => openConversation(c.id)}>
              <MessageSquare size={14} />
              <span className="history-title">{c.title}</span>
              <span className="history-time">{formatTime(c.updated_at)}</span>
              <span role="button" tabIndex={0} className="history-delete" aria-label="删除对话" onClick={(e) => removeConversation(e, c.id)} onKeyDown={(e) => e.key === "Enter" && removeConversation(e, c.id)}><Trash2 size={13} /></span>
            </button>
          ))}
          {conversations.length === 0 && <p className="muted small pad">新的咨询会出现在这里</p>}
        </nav>
        <div className="side-foot">
          {DEMO && demoInfo.repo && <a href={demoInfo.repo} target="_blank" rel="noreferrer noopener"><Github size={14} />查看源码与部署说明</a>}
          <div className="theme-switch" role="group" aria-label="主题">
            {[["auto", "跟随系统"], ["light", "浅色"], ["dark", "深色"]].map(([key, label]) => (
              <button key={key} className={theme === key ? "on" : ""} onClick={() => setTheme(key)}>{label}</button>
            ))}
          </div>
          <p className="muted tiny">提供一般法律信息，不构成律师意见。</p>
        </div>
      </aside>
      {navOpen && <button className="scrim" aria-label="关闭导航" onClick={() => setNavOpen(false)} />}

      <main className="main">
        <header className="topbar">
          <button className="icon-button only-narrow" onClick={() => setNavOpen(true)} aria-label="打开导航"><Menu size={18} /></button>
          <div className="crumbs"><span>WorkAid</span><i>/</i><strong>{title}</strong></div>
          {DEMO && <span className="badge demo"><Info size={12} />在线演示{demoInfo.until ? ` · 至 ${demoInfo.until}` : ""}</span>}
          <button className="icon-button only-medium" onClick={() => setPanelOpen(true)} aria-label="打开依据面板"><PanelRight size={18} /></button>
        </header>

        <div className="workspace">
          <section className="chat">
            {DEMO && !expired && (
              <div className="demo-banner">
                <Info size={14} />
                <span>这是<b>回放演示</b>：下面的场景都是事先从真实系统录制的，没有后端、不会调用任何模型；输入其他问题只会得到说明。完整系统请见源码仓库。</span>
              </div>
            )}
            {expired && <div className="demo-banner expired"><Info size={14} /><span>在线演示已于 {demoInfo.until} 结束。请按 README 在本地或 Codespaces 运行完整系统。</span></div>}

            <div className="scroll" ref={scroller} onScroll={onScroll}>
              {empty ? (
                <div className="hero">
                  <span className="hero-mark"><Scale size={26} /></span>
                  <h1>把劳动纠纷说清楚，<br /><em>下一步一起理顺。</em></h1>
                  <p>描述你的情况，WorkAid 会检索法条与类案，核验每一处引用，并按你所在地给出维权路径。</p>
                  <div className="starters">
                    {starters.map((s) => {
                      const Icon = s.icon || Search;
                      return (
                        <button key={s.title} onClick={() => send(s.text)} disabled={expired}>
                          <span className="starter-icon"><Icon size={16} /></span>
                          <span><strong>{s.title}</strong><small>{s.hint}</small></span>
                        </button>
                      );
                    })}
                  </div>
                  <ul className="trust">
                    <li><ShieldCheck size={14} />回答里的法条和案号逐条核验</li>
                    <li><Shield size={14} />手机号、证件号、姓名自动脱敏</li>
                    <li><Landmark size={14} />按所在地给出机构与材料清单</li>
                  </ul>
                </div>
              ) : (
                <div className="thread">
                  {messages.map((m, i) => <Message key={`${m.created_at || i}-${i}`} message={m} />)}
                  {turn && <LiveMessage turn={turn} />}
                </div>
              )}
            </div>

            {error && <div className="alert" role="alert">{error}<button onClick={() => setError("")} aria-label="关闭"><X size={14} /></button></div>}
            {privacyNote && <div className="privacy-note"><ShieldCheck size={13} />{privacyNote}</div>}

            <div className="composer-wrap">
              <div className="composer">
                <textarea ref={textarea} rows={1} value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={onKey} disabled={expired}
                  placeholder={DEMO ? "点选上面的示例，或试着输入类似的问题…" : "描述你遇到的情况，例如：公司拖欠工资、被辞退、工伤…"} />
                {busy ? (
                  <button className="send stop" onClick={stop} aria-label="停止生成"><Square size={14} fill="currentColor" /></button>
                ) : (
                  <button className="send" onClick={() => send()} disabled={!input.trim() || expired} aria-label="发送"><ArrowUp size={17} /></button>
                )}
              </div>
              <p className="legal">请勿输入身份证号等敏感信息（系统会自动脱敏，但仍建议避免）。回答仅供参考，重大事项请咨询律师或法律援助中心。</p>
            </div>
          </section>

          <ContextPanel tab={tab} setTab={setTab} open={panelOpen} onClose={() => setPanelOpen(false)}
            counts={{ evidence: evidence.length, case: caseFile?.fields?.length || 0, memory: (profile.items || []).length }}>
            {tab === "evidence" && <Evidence workers={workers} evidence={evidence} />}
            {tab === "case" && <CaseFile data={caseFile} conversationId={active} onSaved={setCaseFile} onError={setError} />}
            {tab === "memory" && <Memory profile={profile} reload={loadProfile} onError={setError} onErased={() => { reset(); loadConversations(); loadProfile(); }} />}
          </ContextPanel>
          {panelOpen && <button className="scrim wide" aria-label="关闭面板" onClick={() => setPanelOpen(false)} />}
        </div>
      </main>
    </div>
  );
}
