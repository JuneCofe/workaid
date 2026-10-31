/** Demo mode for static hosting (GitHub Pages).
 *
 * There is no server: each scenario is a recording of a real run of the backend (scripts/record_demo.py),
 * replayed with the same events and pacing the live API produces. Everything outside the scenarios
 * is answered with an explanation instead of an invented reply.
 */
const BASE = import.meta.env.BASE_URL || "/";
const STORE_KEY = "workaid.demo.v1";

export const info = {
  until: import.meta.env.VITE_DEMO_UNTIL || "",
  repo: import.meta.env.VITE_REPO_URL || "",
};

export const SCENARIOS = [
  { id: "wage", title: "欠薪与未签合同", hint: "拖欠三个月工资，没签劳动合同", question: "我在北京一家餐厅做服务员，月薪5000，2025年8月入职，公司拖欠我三个月工资，一直没跟我签劳动合同，怎么办？", match: /拖欠|欠薪|工资|合同/, followup: { id: "wage-2", match: /多少|金额|赔|要回|最多/ } },
  { id: "dismiss", title: "被口头辞退", hint: "没给补偿，怎么主张赔偿", question: "我2020年3月入职，月薪6000，昨天被公司口头辞退了，没有给任何补偿，我在北京。", match: /辞退|开除|解除|辞职|补偿/ },
  { id: "injury", title: "工地受伤不认账", hint: "老板不申请工伤认定", question: "我在北京一个工地干活，2026年9月20日摔伤了，老板不认账，也不给我申请工伤认定，怎么办？", match: /工伤|摔伤|受伤|事故/ },
  { id: "statute", title: "查一条法律", hint: "精确定位条文原文", question: "劳动合同法第八十二条是怎么规定的？", match: /第八十二条|二倍工资|双倍工资/ },
  { id: "trap", title: "诱导编造法条", hint: "看系统如何拦截不存在的条号", question: "劳动合同法第一百九十九条是怎么规定的？", match: /第一百九十九条|一百九十九|199/ },
  { id: "privacy", title: "含个人信息的提问", hint: "手机号、姓名自动脱敏", question: "我叫王小明，手机13812345678，老板赵永康拖欠我两个月工资，月薪7000，我在上海。", match: /我叫|手机|身份证|赵永康/ },
];

// More specific scenarios are tried first, so a question about 第八十二条 is not taken for a wage question.
const ORDER = ["trap", "privacy", "statute", "injury", "dismiss", "wage"];

export function isExpired() {
  if (!info.until) return false;
  const end = new Date(`${info.until}T23:59:59`);
  return !Number.isNaN(end.getTime()) && Date.now() > end.getTime();
}

const sleep = (ms, signal) =>
  new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("Aborted", "AbortError"));
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => { clearTimeout(timer); reject(new DOMException("Aborted", "AbortError")); }, { once: true });
  });

const cache = new Map();
async function fixture(name) {
  if (!cache.has(name)) {
    const response = await fetch(`${BASE}demo/${name}.json`);
    if (!response.ok) throw new Error("演示数据加载失败");
    cache.set(name, await response.json());
  }
  return cache.get(name);
}

function load() {
  try { return JSON.parse(localStorage.getItem(STORE_KEY)) || { conversations: [], profile: [] }; }
  catch { return { conversations: [], profile: [] }; }
}

function save(state) {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(state)); } catch { /* storage unavailable */ }
}

function pick(message, conversation) {
  const text = message.trim();
  if (conversation?.scenario) {
    const scenario = SCENARIOS.find((s) => s.id === conversation.scenario);
    if (scenario?.followup && !conversation.followedUp && scenario.followup.match.test(text)) {
      return { fixture: scenario.followup.id, scenario, followup: true };
    }
  }
  for (const id of ORDER) {
    const scenario = SCENARIOS.find((s) => s.id === id);
    if (scenario.match.test(text)) return { fixture: scenario.id, scenario };
  }
  return null;
}

function notSupported() {
  return (
    "## 这是在线演示\n\n演示站没有后端，只能回放几个**事先从真实系统录制下来**的场景，没法回答新的问题：\n\n" +
    SCENARIOS.map((s) => `- ${s.title}：${s.hint}`).join("\n") +
    "\n\n点击左侧或上方的示例即可体验。想用自己的问题，请按 README 在本地或 Codespaces 里运行完整系统（需要配置自己的模型密钥）。"
  );
}

async function replay({ conversationId, message, signal, onEvent }) {
  if (isExpired()) throw new Error("在线演示已结束，请按 README 在本地或 Codespaces 运行完整系统。");
  const state = load();
  let conversation = state.conversations.find((c) => c.id === conversationId);
  if (!conversation) {
    conversation = { id: `demo-${Date.now().toString(36)}`, title: message.slice(0, 40), updated_at: new Date().toISOString(), messages: [], scenario: "", followedUp: false, case_file: null };
    state.conversations.unshift(conversation);
  }
  const stamp = () => new Date().toISOString();
  conversation.messages.push({ role: "user", content: message, metadata: {}, created_at: stamp() });

  const chosen = pick(message, conversation);
  onEvent("meta", { conversation_id: conversation.id, privacy: { redacted: {}, remembered: [] } });
  await sleep(180, signal);

  if (!chosen) {
    const answer = notSupported();
    onEvent("stage", { name: "understand" });
    await sleep(300, signal);
    for (let at = 0; at < answer.length; at += 14) {
      onEvent("delta", { text: answer.slice(at, at + 14) });
      await sleep(16, signal);
    }
    onEvent("final", { conversation_id: conversation.id, intent: "general", answer, workers: [], evidence: [], verification: { checked: 0, problems: 0 }, case_file: conversation.case_file, scripts: [], privacy: { redacted: {}, remembered: [] } });
    conversation.messages.push({ role: "assistant", content: answer, metadata: { intent: "general", workers: [], evidence: [] }, created_at: stamp() });
    conversation.updated_at = stamp();
    save(state);
    onEvent("done", {});
    return;
  }

  const recording = await fixture(chosen.fixture);
  if (!conversation.scenario) conversation.scenario = chosen.scenario.id;
  if (chosen.followup) conversation.followedUp = true;
  let finalPayload = null;
  for (const { e: name, d: data } of recording.events) {
    if (name === "meta") {
      onEvent("meta", { ...data, conversation_id: conversation.id });
    } else if (name === "delta") {
      onEvent("delta", data);
      await sleep(Math.min(70, 8 + data.text.length * 5), signal);
    } else if (name === "final") {
      finalPayload = { ...data, conversation_id: conversation.id };
      onEvent("final", finalPayload);
    } else {
      if (name === "stage") await sleep(320, signal);
      if (name === "worker") await sleep(420, signal);
      onEvent(name, data);
    }
  }
  if (finalPayload) {
    conversation.messages.push({ role: "assistant", content: finalPayload.answer, metadata: { intent: finalPayload.intent, workers: finalPayload.workers, evidence: finalPayload.evidence, verification: finalPayload.verification, case_file: finalPayload.case_file }, created_at: stamp() });
    conversation.case_file = finalPayload.case_file || conversation.case_file;
    conversation.updated_at = stamp();
    if (recording.profile_items?.length) state.profile = recording.profile_items;
    save(state);
  }
  onEvent("done", {});
}

export const api = {
  stream: replay,
  async conversations() {
    return load().conversations.map((c) => ({ id: c.id, title: c.title, updated_at: c.updated_at, message_count: c.messages.length }));
  },
  async conversation(id) {
    const found = load().conversations.find((c) => c.id === id);
    if (!found) throw new Error("会话不存在");
    return { id: found.id, title: found.title, updated_at: found.updated_at, messages: found.messages };
  },
  async deleteConversation(id) {
    const state = load();
    state.conversations = state.conversations.filter((c) => c.id !== id);
    save(state);
    return { deleted: true };
  },
  async profile() {
    return { facts: [], preferences: {}, items: load().profile || [] };
  },
  async deleteProfileItem(item) {
    const state = load();
    state.profile = (state.profile || []).filter((p) => !(p.key === item.key && p.value === item.value));
    save(state);
    return { deleted: true };
  },
  async eraseAll() {
    save({ conversations: [], profile: [] });
    return { deleted: {} };
  },
  async caseFile(id) {
    const found = load().conversations.find((c) => c.id === id);
    return found?.case_file || null;
  },
  async answerCase() {
    throw new Error("演示模式不支持填写问卷，请在完整版里体验。");
  },
  exportUrl: () => "",
  async sources() {
    return fixture("sources");
  },
  async health() {
    return fixture("health");
  },
  async searchStatutes(q) {
    const table = await fixture("statutes");
    const norm = (text) => text.replace(/\s+/g, "");
    const query = norm(q);
    // A demo key matches only when the whole key appears in the query, so "第八十二条" cannot pick "第九十条".
    const key = Object.keys(table).find((k) => query.includes(norm(k)));
    if (!key) throw new Error("演示模式只提供几个示例检索（如“劳动合同法 第八十二条”），完整版可检索全部法规。");
    return table[key];
  },
};
