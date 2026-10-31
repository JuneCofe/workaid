/** A small, safe Markdown renderer for assistant answers.
 *  All text is HTML-escaped first; only the tags produced here reach the page. */

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const escapeHtml = (value) => value.replace(/[&<>"']/g, (ch) => ESCAPES[ch]);

function inline(text) {
  let out = escapeHtml(text);
  out = out.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  out = out.replace(/\*\*([^*\n]+?)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/(^|[^*])\*([^*\s][^*\n]*?)\*(?!\*)/g, "$1<em>$2</em>");
  out = out.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer noopener">$1</a>');
  out = out.replace(/(^|[\s（(：:])(https?:\/\/[^\s<）)，。；]+)/g, (match, lead, url) => {
    if (lead && url.includes("&quot;")) return match;
    const label = url.length > 56 ? `${url.slice(0, 54)}…` : url;
    return `${lead}<a href="${url}" target="_blank" rel="noreferrer noopener">${label}</a>`;
  });
  return out;
}

const isTableRule = (line) => /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/.test(line);
const cells = (line) => line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());

export function renderMarkdown(source) {
  const lines = (source || "").replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) { index += 1; continue; }

    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      const level = Math.min(heading[1].length + 1, 5);
      html.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      index += 1;
      continue;
    }
    if (/^\s*(---+|\*\*\*+)\s*$/.test(line)) { html.push("<hr />"); index += 1; continue; }

    if (line.trim().startsWith("|") && index + 1 < lines.length && isTableRule(lines[index + 1])) {
      const head = cells(line);
      index += 2;
      const rows = [];
      while (index < lines.length && lines[index].trim().startsWith("|")) { rows.push(cells(lines[index])); index += 1; }
      html.push(`<div class="table-wrap"><table><thead><tr>${head.map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
      continue;
    }

    if (/^\s*>/.test(line)) {
      const quote = [];
      while (index < lines.length && /^\s*>/.test(lines[index])) { quote.push(lines[index].replace(/^\s*>\s?/, "")); index += 1; }
      html.push(`<blockquote>${quote.map(inline).join("<br />")}</blockquote>`);
      continue;
    }

    const bullet = /^\s*[-*•]\s+/;
    const numbered = /^\s*\d+[.、)]\s+/;
    if (bullet.test(line) || numbered.test(line)) {
      const ordered = numbered.test(line);
      const pattern = ordered ? numbered : bullet;
      const items = [];
      while (index < lines.length && pattern.test(lines[index])) {
        let text = lines[index].replace(pattern, "");
        index += 1;
        while (index < lines.length && lines[index].trim() && /^\s{2,}\S/.test(lines[index]) && !pattern.test(lines[index])) {
          text += ` ${lines[index].trim()}`;
          index += 1;
        }
        items.push(`<li>${inline(text)}</li>`);
      }
      html.push(`<${ordered ? "ol" : "ul"}>${items.join("")}</${ordered ? "ol" : "ul"}>`);
      continue;
    }

    const paragraph = [];
    while (index < lines.length && lines[index].trim() && !/^(#{1,4}\s|\s*>|\s*[-*•]\s+|\s*\d+[.、)]\s+|\s*---+\s*$|\s*\|)/.test(lines[index])) {
      paragraph.push(lines[index]);
      index += 1;
    }
    if (paragraph.length) html.push(`<p>${paragraph.map(inline).join("<br />")}</p>`);
    else index += 1;
  }
  return html.join("");
}

export function plainText(source) {
  return (source || "").replace(/[*`>#]/g, "").replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
}
