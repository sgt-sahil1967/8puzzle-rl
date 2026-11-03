// Drives headless Chrome through the Chrome DevTools Protocol and saves screenshots of the demo site.
import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
const PORT = 9333, OUT = decodeURIComponent(new URL("./shots/", import.meta.url).pathname);
const chrome = spawn("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  ["--headless=new", `--remote-debugging-port=${PORT}`, "--no-first-run", "--user-data-dir=/tmp/chrome-shots", "--hide-scrollbars", "about:blank"], { stdio: "ignore" });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let ws, id = 0; const pending = new Map();
async function connect() {
  for (let i = 0; i < 50; i++) { try { const t = await (await fetch(`http://127.0.0.1:${PORT}/json`)).json();
      const page = t.find(x => x.type === "page"); if (page) { ws = new WebSocket(page.webSocketDebuggerUrl); break; } } catch {} await sleep(200); }
  await new Promise(r => ws.addEventListener("open", r));
  ws.addEventListener("message", e => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } });
}
const send = (method, params = {}) => new Promise(res => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
const js = async (expr) => (await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;
const shot = async (name, sel) => {
  let clip;
  if (sel) { const r = await js(`(()=>{const e=document.querySelector(${JSON.stringify(sel)});const b=e.getBoundingClientRect();return [b.x+scrollX,b.y+scrollY,b.width,b.height]})()`); clip = { x: r[0] - 8, y: r[1] - 8, width: r[2] + 16, height: r[3] + 16, scale: 1 }; }
  else { const h = await js("document.documentElement.scrollHeight"); const w = await js("document.documentElement.clientWidth"); clip = { x: 0, y: 0, width: w, height: h, scale: 1 }; }
  const r = await send("Page.captureScreenshot", { format: "png", clip, captureBeyondViewport: true });
  writeFileSync(OUT + name, Buffer.from(r.result.data, "base64")); console.log("saved", name);
};
const sec = id => `(document.getElementById('${id}').closest('section')||document.getElementById('${id}').parentElement)`;
const region = async (name, fromId, toId) => {
  const r = await js(`(()=>{const a=${sec(fromId)}.getBoundingClientRect(), b=${sec(toId)}.getBoundingClientRect();return [Math.min(a.x,b.x)+scrollX, a.y+scrollY, Math.max(a.width,b.width), b.bottom-a.top]})()`);
  const clip = { x: r[0] - 6, y: r[1] - 6, width: r[2] + 12, height: r[3] + 12, scale: 1 };
  const d = await send("Page.captureScreenshot", { format: "png", clip, captureBeyondViewport: true });
  writeFileSync(OUT + name, Buffer.from(d.result.data, "base64")); console.log("saved", name);
};
const click = sel => js(`document.querySelector(${JSON.stringify(sel)}).click()`);
const waitText = async (sel, re, ms = 20000) => { const t0 = Date.now(); while (Date.now() - t0 < ms) { const t = await js(`document.querySelector(${JSON.stringify(sel)})?.innerText||""`); if (new RegExp(re, "i").test(t)) return t; await sleep(300); } return null; };

await connect();
await send("Page.enable"); await send("DOM.enable"); await send("Runtime.enable");
await send("Emulation.setDeviceMetricsOverride", { width: 1100, height: 900, deviceScaleFactor: 1.5, mobile: false });
await send("Page.navigate", { url: "http://127.0.0.1:5000/" }); await sleep(2500);
await region("01_home.png", "h-image", "h-play");

// upload a picture through the real file input
const doc = await send("DOM.getDocument"); const inp = await send("DOM.querySelector", { nodeId: doc.result.root.nodeId, selector: "#imageInput" });
await send("DOM.setFileInputFiles", { nodeId: inp.result.nodeId, files: [OUT + "_picture.png"] }); await sleep(2000);
await js(`document.getElementById('depth').value=20; document.getElementById('depth').dispatchEvent(new Event('input',{bubbles:true}))`);
await click("#scrambleBtn"); await sleep(1500);
await region("02_upload_scramble.png", "h-image", "h-scramble");

await js(`document.getElementById('speed').value=10; document.getElementById('speed').dispatchEvent(new Event('input',{bubbles:true}))`);
// Watch AI: A*
await click('[data-method="astar"]'); await sleep(1200); await region("03_watch_mid.png", "h-play", "h-play");
console.log("astar:", await waitText("#statusMsg", "solved|failed|moves")); await sleep(800); await region("04_watch_astar_solved.png", "h-play", "h-play");
// try to catch a DQN v1 failure on a deep board for the "failed honestly" screenshot
let failed = false;
for (let i = 0; i < 12 && !failed; i++) {
  await js(`document.getElementById('depth').value=30; document.getElementById('depth').dispatchEvent(new Event('input',{bubbles:true}))`);
  await click("#scrambleBtn"); await sleep(700);
  await click('[data-method="dqn"]');
  const t = await waitText("#statusMsg", "solved|failed", 25000); console.log("dqn try", i, (t || "").slice(0, 80));
  if (t && /fail/i.test(t)) { failed = true; await sleep(500); await region("05_watch_dqn_failed.png", "h-play", "h-play"); }
}
// DQN v2 on the current board
for (let i = 0; i < 15; i++) {
  await js(`document.getElementById('depth').value=14; document.getElementById('depth').dispatchEvent(new Event('input',{bubbles:true}))`);
  await click("#scrambleBtn"); await sleep(700); await click('[data-method="dqn2"]');
  const t = await waitText("#statusMsg", "solved|failed", 25000); console.log("dqn2 try", i, (t || "").slice(0, 60));
  if (t && /solved/i.test(t) && !/fail/i.test(t)) { await sleep(500); await region("06_watch_dqn2.png", "h-play", "h-play"); break; }
}

// Compare mode
await click('[data-mode="compare"]'); await js(`document.getElementById('depth').value=25; document.getElementById('depth').dispatchEvent(new Event('input',{bubbles:true}))`);
await click("#scrambleBtn"); await sleep(800); await click("#compareBtn"); await sleep(1500); await waitText("#compareTable", "A\\*", 25000); await sleep(6000);
await region("07_compare.png", "h-play", "h-play");

// Play mode with a hint
await click('[data-mode="play"]'); await js(`document.getElementById('depth').value=8; document.getElementById('depth').dispatchEvent(new Event('input',{bubbles:true}))`);
await click("#scrambleBtn"); await sleep(800); await click("#hintBtn"); await sleep(800); await region("08_play_hint.png", "h-play", "h-play");

// results panel
await region("09_results_panel.png", "h-results", "h-results");
await js("document.getElementById('h-results').parentElement.id");
// mobile
await send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 2, mobile: true });
await send("Page.navigate", { url: "http://127.0.0.1:5000/" }); await sleep(2500); await click("#scrambleBtn"); await sleep(800);
await shot("10_mobile.png");
chrome.kill(); process.exit(0);
