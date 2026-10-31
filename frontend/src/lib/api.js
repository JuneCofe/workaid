import * as demo from "./demo.js";

export const DEMO = import.meta.env.VITE_DEMO === "1";
const API = "/api";
const USER_KEY = "workaid.user-id";

function makeUserId() {
  const id = typeof crypto?.randomUUID === "function" ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(16).slice(2);
  return `guest-${id}`;
}

export const userId = (() => {
  try {
    const stored = localStorage.getItem(USER_KEY);
    if (stored) return stored;
    const fresh = makeUserId();
    localStorage.setItem(USER_KEY, fresh);
    return fresh;
  } catch {
    return makeUserId();
  }
})();

const enc = encodeURIComponent;

async function request(path, init) {
  const response = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "请求失败，请稍后重试");
  }
  return response.json();
}

/** Parse a text/event-stream body into (event, data) calls. */
async function readEvents(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let name = "message";
      const data = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) name = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
      }
      if (data.length) onEvent(name, JSON.parse(data.join("\n")));
    }
  }
}

async function streamChat({ conversationId, message, signal, onEvent }) {
  const response = await fetch(`${API}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ user_id: userId, conversation_id: conversationId, message }),
    signal,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "请求失败，请稍后重试");
  }
  await readEvents(response, onEvent);
}

const live = {
  stream: streamChat,
  conversations: () => request(`/conversations?user_id=${enc(userId)}`),
  conversation: (id) => request(`/conversations/${id}?user_id=${enc(userId)}`),
  deleteConversation: (id) => request(`/conversations/${id}?user_id=${enc(userId)}`, { method: "DELETE" }),
  profile: () => request(`/profile?user_id=${enc(userId)}`),
  deleteProfileItem: (item) => request(`/profile/items?user_id=${enc(userId)}&key=${enc(item.key)}&value=${enc(item.value)}`, { method: "DELETE" }),
  eraseAll: () => request(`/user-data?user_id=${enc(userId)}`, { method: "DELETE" }),
  caseFile: (id) => request(`/conversations/${id}/case?user_id=${enc(userId)}`),
  answerCase: (id, answers) => request(`/conversations/${id}/case`, { method: "PUT", body: JSON.stringify({ user_id: userId, answers }) }),
  exportUrl: (id) => `${API}/conversations/${id}/case/export?user_id=${enc(userId)}`,
  sources: () => request("/sources"),
  health: () => request("/health"),
  searchStatutes: (q) => request(`/statutes/search?q=${enc(q)}`),
};

export const api = DEMO ? demo.api : live;
export const demoInfo = demo.info;
