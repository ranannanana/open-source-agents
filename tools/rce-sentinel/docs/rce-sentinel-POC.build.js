const pptxgen = require("pptxgenjs");
const fs = require("fs");
const S = JSON.parse(fs.readFileSync(__dirname + "/cave_summary.json", "utf8"));
const LOGO = process.argv[2];
const MARK = process.argv[3];

// Impact Analytics palette
const BLUE = "264CD7", OFF = "F4F4F6", INK = "1C1B1B", G1 = "E4E4E4", G2 = "D7D6D2",
  G3 = "B3B2AD", WHITE = "FFFFFF", ORANGE = "FF6F1C", MUTED = "5F5E5A", TINT = "E9EEFC";
const H = "Calibri", B = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: "Calibri", bodyFontFace: "Calibri" };
pres.author = "Prakhar Khurana";
pres.company = "Impact Analytics";
pres.title = "rce-sentinel POC: cost-aware model routing";

const W = 13.33, M = 0.6;

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: OFF },
  objects: [
    { image: { path: MARK, x: W - M - 0.42, y: 0.42, w: 0.42, h: 0.36 } },
    { text: { text: "Internal | rce-sentinel POC | Nov 2026", options: { x: M, y: 7.0, w: 6, h: 0.3, fontFace: B, fontSize: 10, color: G3, margin: 0 } } },
  ],
  slideNumber: { x: W - M - 0.5, y: 7.0, w: 0.5, h: 0.3, fontFace: B, fontSize: 10, color: G3, align: "right" },
});

function pre(s, text) {
  s.addText(text, { x: M, y: 0.45, w: 9, h: 0.3, fontFace: B, fontSize: 12, bold: true, color: BLUE, margin: 0, isTextBox: true });
}
function title(s, text, sub) {
  s.addText(text, { x: M, y: 0.8, w: W - 2 * M - 0.6, h: 0.95, fontFace: H, fontSize: 26, valign: "top", bold: true, color: INK, margin: 0, isTextBox: true });
  if (sub) s.addText(sub, { x: M, y: 1.8, w: W - 2 * M, h: 0.35, fontFace: B, fontSize: 14, color: MUTED, margin: 0, isTextBox: true });
}
function card(s, x, y, w, h, fill = WHITE) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: G1, width: 1 }, rectRadius: 0.12 });
}
function num(s, n, x, y) {
  s.addShape(pres.shapes.OVAL, { x, y, w: 0.42, h: 0.42, fill: { color: TINT }, line: { color: TINT } });
  s.addText(String(n), { x, y, w: 0.42, h: 0.42, align: "center", valign: "middle", fontFace: B, fontSize: 13, bold: true, color: BLUE, margin: 0, isTextBox: true });
}
function pill(s, text, x, y, w, color, fill) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.32, fill: { color: fill }, line: { color: fill }, rectRadius: 0.16 });
  s.addText(text, { x, y, w, h: 0.32, align: "center", valign: "middle", fontFace: B, fontSize: 11, bold: true, color, margin: 0, isTextBox: true });
}
function banner(s, label, text, y = 5.95) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y, w: W - 2 * M, h: 0.85, fill: { color: BLUE }, line: { color: BLUE }, rectRadius: 0.12 });
  s.addText([
    { text: label, options: { fontSize: 11, color: "C9D4F7", breakLine: true } },
    { text, options: { fontSize: 15, bold: true, color: WHITE } },
  ], { x: M + 0.35, y, w: W - 2 * M - 0.7, h: 0.85, valign: "middle", fontFace: B, margin: 0, isTextBox: true });
}

// 1. Title
{
  const s = pres.addSlide();
  s.background = { color: OFF };
  s.addImage({ path: LOGO, x: M, y: 0.55, w: 2.35, h: 0.8 });
  s.addText("Prakhar Khurana | POC readout for engineering", { x: M, y: 1.85, w: 7, h: 0.35, fontFace: B, fontSize: 14, bold: true, color: BLUE, margin: 0, isTextBox: true });
  s.addText("Send each dev prompt to the cheapest model that gets it right", { x: M, y: 2.3, w: 7.6, h: 1.75, fontFace: H, fontSize: 34, bold: true, color: INK, margin: 0, valign: "top", isTextBox: true });
  s.addText("rce-sentinel: a cost-aware model router with context handoff. What is built, what the benchmarks show, and what caveman mode saves.", { x: M, y: 3.85, w: 7.2, h: 0.9, fontFace: B, fontSize: 15, color: MUTED, margin: 0, valign: "top", isTextBox: true });
  // stat column
  card(s, 8.6, 0.8, 4.13, 5.6, WHITE);
  const stats = [["98%", "pass rate with the router on held-out prompts"], ["55%", "less spend than always using Sonnet"], ["894", "graded benchmark runs across 6 models"]];
  stats.forEach(([n, l], i) => {
    const y = 1.15 + i * 1.75;
    s.addText(n, { x: 8.95, y, w: 3.5, h: 0.85, fontFace: H, fontSize: 48, bold: true, color: BLUE, margin: 0, isTextBox: true });
    s.addText(l, { x: 8.95, y: y + 0.85, w: 3.5, h: 0.6, fontFace: B, fontSize: 13, color: INK, margin: 0, valign: "top", isTextBox: true });
  });
  s.addText("Internal | 17 Nov 2026 | Figures are API-equivalent cost on a Claude Code seat, not billed spend", { x: M, y: 6.9, w: 8, h: 0.3, fontFace: B, fontSize: 10, color: G3, margin: 0, isTextBox: true });
}

// 2. How it works
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  pre(s, "How it works");
  title(s, "Tag the prompt, route on measured pass rate, then learn from the grade", "The same loop runs for every prompt. Nothing the models write is executed.");
  const steps = [
    ["Tag", "Pre-eval reads the prompt: task type (build, debug, design), language, difficulty."],
    ["Route", "Pick the cheapest model whose calibrated pass rate clears the bar, counting the cost of a failed try."],
    ["Run", "Claude through the Claude Code seat (Haiku, Sonnet, Opus) or local models through Ollama. Code is sandboxed."],
    ["Grade and learn", "Checks plus a judge from another model family. Each result updates the calibration."],
  ];
  const cw = (W - 2 * M - 3 * 0.3) / 4;
  steps.forEach(([h, t], i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, 2.35, cw, 2.75);
    num(s, i + 1, x + 0.3, 2.6);
    s.addText(h, { x: x + 0.3, y: 3.2, w: cw - 0.6, h: 0.4, fontFace: H, fontSize: 18, bold: true, color: INK, margin: 0, isTextBox: true });
    s.addText(t, { x: x + 0.3, y: 3.65, w: cw - 0.6, h: 1.4, fontFace: B, fontSize: 13, color: MUTED, margin: 0, valign: "top", isTextBox: true });
  });
  card(s, M, 5.3, W - 2 * M, 1.4, TINT);
  s.addText([
    { text: "Handoff between models: ", options: { bold: true, color: BLUE } },
    { text: "when a task moves to another model, a signed context pack carries the goal, your requests word for word, decisions, plan, done, next steps and files, instead of the whole transcript.", options: { color: INK } },
  ], { x: M + 0.35, y: 5.3, w: W - 2 * M - 0.7, h: 1.4, valign: "middle", fontFace: B, fontSize: 15, margin: 0, isTextBox: true });
}

// 3. Which model for which work
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  pre(s, "Which model, which work");
  title(s, "Haiku handles most debugging and design; Sonnet earns its price on multi-file builds", "Learned from 39 hard prompts, graded with checks plus a judge from another model family");
  const hdr = (t) => ({ text: t, options: { bold: true, color: BLUE, fill: { color: TINT }, fontSize: 13 } });
  const rows = [
    [hdr("Work"), hdr("Model"), hdr("Evidence"), hdr("Spend")],
    ["Debugging: stack traces, wrong output, flaky tests", "Haiku", "15/16 hard debug prompts passed", "$0.005 per prompt"],
    ["Design: APIs, schemas, trade-off write-ups", "Haiku", "7/7 hard design prompts passed", "$0.005 per prompt"],
    ["Multi-file builds: services, UI and backend together", "Sonnet", "14/15 builds, against Haiku's 7/15", "$0.045 per prompt"],
    ["Small, single-language snippets (Rust, Go, Java)", "Local qwen-coder 7B (candidate)", "22/42 on varied prompts, 1/39 on hard ones", "$0 (runs locally)"],
    ["Anything at all", "Opus: not chosen", "Cheapest correct model on only 1 of 39 hard prompts", "$0.158 per prompt"],
  ];
  s.addTable(rows, {
    x: M, y: 2.35, w: W - 2 * M, colW: [4.4, 2.6, 3.6, 1.53], rowH: 0.62,
    fontFace: B, fontSize: 13, color: INK, valign: "middle",
    fill: { color: WHITE }, border: { type: "solid", pt: 0.75, color: G1 }, margin: [0, 0.12, 0, 0.12],
  });
  s.addText("Spend is the API-equivalent cost reported by the Claude Code CLI. The seat itself is flat-rate.", { x: M, y: 6.3, w: W - 2 * M, h: 0.35, fontFace: B, fontSize: 12, italic: true, color: MUTED, margin: 0, isTextBox: true });
}

// 4. Hard benchmark chart
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  pre(s, "Evidence: hard prompts");
  title(s, "On hard prompts the gap between models shows up only in builds", "Pass rate (%) by task type, 39 hard prompts. Local models passed 1 of 39 and are left off.");
  const labels = ["Build (15)", "Debug (16)", "Design (7)"];
  s.addChart(pres.charts.BAR, [
    { name: "Haiku", labels, values: [47, 94, 100] },
    { name: "Sonnet", labels, values: [93, 94, 100] },
    { name: "Opus", labels, values: [73, 100, 100] },
  ], {
    x: M, y: 2.3, w: 7.6, h: 4.45, barDir: "col", barGrouping: "clustered",
    chartColors: ["B3C9F7", BLUE, "1C1B1B"],
    showLegend: true, legendPos: "b", legendFontFace: "+mn-lt", legendFontSize: 12, legendColor: INK,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: INK, dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: MUTED, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt",
    valAxisLabelColor: MUTED, valAxisLabelFontSize: 11, valAxisLabelFontFace: "+mn-lt",
    valAxisMaxVal: 100, valAxisMinVal: 0, valGridLine: { color: "E4E4E4", size: 0.5 }, catGridLine: { style: "none" },
  });
  const cx = 8.6, cw = W - M - cx;
  const facts = [["26 of 39", "hard prompts where Haiku was the cheapest model that passed"], ["9x", "Sonnet's cost per prompt compared with Haiku's"], ["3.5x", "Opus's cost compared with Sonnet's, with a lower pass rate (34 vs 36 of 39)"]];
  facts.forEach(([n, l], i) => {
    const y = 2.3 + i * 1.5;
    card(s, cx, y, cw, 1.32);
    s.addText(n, { x: cx + 0.3, y: y + 0.15, w: cw - 0.6, h: 0.55, fontFace: H, fontSize: 26, bold: true, color: BLUE, margin: 0, isTextBox: true });
    s.addText(l, { x: cx + 0.3, y: y + 0.7, w: cw - 0.6, h: 0.55, fontFace: B, fontSize: 12, color: INK, margin: 0, valign: "top", isTextBox: true });
  });
}

// 5. Router vs single model
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  pre(s, "Evidence: router vs one model");
  title(s, "The router keeps the top pass rate at less than half of Sonnet's spend", "46 held-out prompts. Extra cost = spend above the cheapest correct model, including review and rework after a failure.");
  const strat = [
    ["Router (calibrated)", "98%", "$0.98", "$0.39", true],
    ["Always Sonnet", "96%", "$1.15", "$0.87", false],
    ["Always Opus", "98%", "$1.16", "$2.42", false],
    ["Always Haiku", "98%", "$1.24", "$0.09", false],
  ];
  const cw = (W - 2 * M - 3 * 0.3) / 4;
  strat.forEach(([n, p, r, sp, hi], i) => {
    const x = M + i * (cw + 0.3), y = 2.35;
    card(s, x, y, cw, 3.3, hi ? BLUE : WHITE);
    const fg = hi ? WHITE : INK, sub = hi ? "C9D4F7" : MUTED;
    s.addText(n, { x: x + 0.3, y: y + 0.25, w: cw - 0.6, h: 0.4, fontFace: H, fontSize: 16, bold: true, color: fg, margin: 0, isTextBox: true });
    [["Pass", p], ["Extra cost per prompt", r], ["Total spend", sp]].forEach(([k, v], j) => {
      const yy = y + 0.85 + j * 0.82;
      s.addText(v, { x: x + 0.3, y: yy, w: cw - 0.6, h: 0.45, fontFace: H, fontSize: 24, bold: true, color: hi ? WHITE : (j === 1 ? INK : INK), margin: 0, isTextBox: true });
      s.addText(k, { x: x + 0.3, y: yy + 0.42, w: cw - 0.6, h: 0.3, fontFace: B, fontSize: 11, color: sub, margin: 0, isTextBox: true });
    });
  });
  banner(s, "Caveat", "Always-Haiku is cheapest but fails more often on builds. The router's one miss was the websocket-presence build.");
}

// 6. Shared memory
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  pre(s, "Shared memory: context pack");
  title(s, "The context pack keeps 62% of your decisions using a fifth of the handoff tokens", "Live 4-turn build passed from Haiku to Sonnet. Seven decisions from turn 1 were checked in later files.");
  const opts = [
    ["Nothing passed on", "0%", "0 tokens", "$0.05", "The next model starts blind."],
    ["Full conversation", "100%", "16.2K tokens", "$0.62", "Keeps everything, costs the most."],
    ["Context pack v2", "62%", "3.4K tokens", "$0.45", "Was 12% before your requests were kept word for word."],
  ];
  const cw = (W - 2 * M - 2 * 0.3) / 3;
  opts.forEach(([n, ret, tok, usd, note], i) => {
    const x = M + i * (cw + 0.3), y = 2.35;
    card(s, x, y, cw, 2.75, i === 2 ? TINT : WHITE);
    s.addText(n, { x: x + 0.3, y: y + 0.2, w: cw - 0.6, h: 0.4, fontFace: H, fontSize: 16, bold: true, color: INK, margin: 0, isTextBox: true });
    s.addText(ret, { x: x + 0.3, y: y + 0.65, w: cw - 0.6, h: 0.75, fontFace: H, fontSize: 40, bold: true, color: BLUE, margin: 0, isTextBox: true });
    s.addText(`decisions kept | ${tok} | ${usd}`, { x: x + 0.3, y: y + 1.4, w: cw - 0.6, h: 0.35, fontFace: B, fontSize: 12, color: MUTED, margin: 0, isTextBox: true });
    s.addText(note, { x: x + 0.3, y: y + 1.85, w: cw - 0.6, h: 0.7, fontFace: B, fontSize: 13, color: INK, margin: 0, valign: "top", isTextBox: true });
  });
  card(s, M, 5.3, W - 2 * M, 1.45);
  s.addText([
    { text: "Why it isn't 100% yet: ", options: { bold: true, color: ORANGE } },
    { text: "in turn 1 Haiku made up its own rule, \"split work into small pieces\". The pack passed it on as binding, so Sonnet built only part of the UI. Fix next: mark rules the model invented as advisory. One run per mode, so treat as directional.", options: { color: INK } },
  ], { x: M + 0.35, y: 5.3, w: W - 2 * M - 0.7, h: 1.45, valign: "middle", fontFace: B, fontSize: 14, margin: 0, isTextBox: true });
}

// 7. Goals scorecard
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  pre(s, "Where the POC stands");
  title(s, "Most of the machinery is built; proof from real day-to-day use is still thin", "Seven goals set at the start, 27 Oct 2026");
  const G = [
    ["Save money and use tokens well", "Built", "Router: 55% less spend than always Sonnet, on benchmark prompts"],
    ["Try similar model families first", "Built", "Escalation stays at equal or higher quality and prefers another family"],
    ["Configurable, swappable cache", "Partial", "Context pack v2: 62% of decisions kept on 21% of the tokens"],
    ["Personal insights and session stats", "Partial", "Cost report exists; only 6 live usage events so far"],
    ["Route on reliability, save the metrics", "Built", "894 benchmark runs and 972 outcomes feed the calibration"],
    ["Analyse and tag the prompt first", "Partial", "Task tagging is 55% accurate on the varied prompt set, kept unseen"],
    ["Predict the workload at runtime", "Not started", "No model of session length or cost yet"],
  ];
  const col = { Built: ["1C1B1B", "D7F5E9"], Partial: ["1C1B1B", "FFE2D1"], "Not started": [WHITE, "5F5E5A"] };
  G.forEach(([g, st, ev], i) => {
    const y = 2.35 + i * 0.6;
    card(s, M, y, W - 2 * M, 0.5);
    s.addText(g, { x: M + 0.25, y, w: 4.3, h: 0.5, fontFace: B, fontSize: 13, bold: true, color: INK, valign: "middle", margin: 0, isTextBox: true });
    pill(s, st, M + 4.65, y + 0.09, 1.3, col[st][0], col[st][1]);
    s.addText(ev, { x: M + 6.2, y, w: W - 2 * M - 6.4, h: 0.5, fontFace: B, fontSize: 13, color: MUTED, valign: "middle", margin: 0, isTextBox: true });
  });
}

// 8. Caveman: usage stats
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  pre(s, "/caveman-mode skill: usage");
  title(s, "Caveman mode is on in every session, but the replies still read like normal English", "Measured over all Claude Code transcripts on this machine, 5 Oct to 17 Nov 2026");
  const tiles = [
    [String(S.sessions), "sessions with caveman mode loaded (a SessionStart hook forces it)"],
    [S.messages.toLocaleString("en-US"), "chat replies analysed, " + Math.round(S.words / 1000) + "K words"],
    [String(S.med_words), "words in the median reply"],
    [S.art_rate.toFixed(1), "articles (a, an, the) per 100 words, the rate of ordinary English"],
  ];
  const cw = (W - 2 * M - 3 * 0.3) / 4;
  tiles.forEach(([n, l], i) => {
    const x = M + i * (cw + 0.3), y = 2.35;
    card(s, x, y, cw, 1.9);
    s.addText(n, { x: x + 0.3, y: y + 0.2, w: cw - 0.6, h: 0.75, fontFace: H, fontSize: 38, bold: true, color: i === 3 ? ORANGE : BLUE, margin: 0, isTextBox: true });
    s.addText(l, { x: x + 0.3, y: y + 0.95, w: cw - 0.6, h: 0.85, fontFace: B, fontSize: 12, color: INK, margin: 0, valign: "top", isTextBox: true });
  });
  card(s, M, 4.45, W - 2 * M, 2.3);
  s.addText([
    { text: "What the transcripts show", options: { bold: true, color: BLUE, breakLine: true } },
    { text: "Every session loads the skill, but the rule against articles is mostly not followed: about 9 per 100 words remain, the same as ordinary writing.", options: { bullet: true, breakLine: true } },
    { text: "Replies stay short (median " + S.med_words + " words), so word count was never the main cost; tool calls and code are.", options: { bullet: true, breakLine: true } },
    { text: "There is no session without the skill to compare against, so the controlled test on the next slide is the real measure.", options: { bullet: true } },
  ], { x: M + 0.35, y: 4.6, w: W - 2 * M - 0.7, h: 2.05, valign: "top", fontFace: B, fontSize: 14, color: INK, margin: 0, paraSpaceAfter: 6, isTextBox: true });
}

// 9. Caveman: A/B findings
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  pre(s, "/caveman-mode skill: controlled test");
  title(s, S.ab_title, "12 developer questions, each asked with and without the skill. Sonnet judged whether the shorter answer kept the key points.");
  const labels = ["Haiku", "Sonnet"];
  s.addChart(pres.charts.BAR, [
    { name: "Normal", labels, values: [S.haiku.normal_tok, S.sonnet.normal_tok] },
    { name: "Caveman", labels, values: [S.haiku.cave_tok, S.sonnet.cave_tok] },
  ], {
    x: M, y: 2.3, w: 6.4, h: 4.45, barDir: "col", barGrouping: "clustered",
    chartColors: ["B3B2AD", BLUE], showTitle: true, title: "Median output tokens per answer", titleFontSize: 13, titleColor: INK, titleFontFace: "+mn-lt",
    showLegend: true, legendPos: "b", legendFontFace: "+mn-lt", legendFontSize: 12, legendColor: INK,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: INK, dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: MUTED, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt",
    valAxisLabelColor: MUTED, valAxisLabelFontSize: 11, valAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: "E4E4E4", size: 0.5 }, catGridLine: { style: "none" },
  });
  const cx = 7.4, cw = W - M - cx;
  const facts = [
    [S.cut_pct + "%", "fewer output tokens in total (Haiku 54%, Sonnet 44%)"],
    [S.kept_pct + "%", "of the normal answer's key points kept in the caveman answer"],
    [S.faster_pct + "%", "faster to the final answer (median, both models)"],
    [S.errors + " of " + S.judged, "caveman answers the judge flagged for a minor error the normal one avoided"],
  ];
  facts.forEach(([n, l], i) => {
    const y = 2.3 + i * 1.12;
    card(s, cx, y, cw, 1.04);
    s.addText(n, { x: cx + 0.3, y: y + 0.08, w: cw - 0.6, h: 0.5, fontFace: H, fontSize: 24, bold: true, color: BLUE, margin: 0, isTextBox: true });
    s.addText(l, { x: cx + 0.3, y: y + 0.54, w: cw - 0.6, h: 0.42, fontFace: B, fontSize: 12, color: INK, margin: 0, valign: "top", isTextBox: true });
  });
}

// 10. Habits / don'ts
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  pre(s, "What we learned");
  title(s, "Habits that keep quality up and spend down");
  const cw = (W - 2 * M - 0.3) / 2;
  const L = [
    "Start on Haiku for debugging and design. Move up only on a failed check.",
    "Score a wrong answer at the full task's size. A short wrong reply is not cheap.",
    "Pass your requests on word for word at each handoff, not the model's summary of them.",
    "Grade with a model from another family, and loosen checks that fail correct answers.",
    "Keep caveman mode for chat; it cuts reply tokens with little loss of content.",
  ];
  const R = [
    "Don't default to Opus. It was the cheapest correct model on 1 of 39 hard prompts.",
    "Don't trust a single-run comparison. Rerun before acting on it.",
    "Don't read only the CLI's result field: long replies get cut to their last part.",
    "Don't let a failed task retry on a weaker model.",
    "Don't route hard multi-file builds to local 3B-7B models yet.",
  ];
  [["Do", L, WHITE], ["Don't", R, WHITE]].forEach(([h, items, f], i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, 1.75, cw, 4.0, f);
    s.addText(h, { x: x + 0.35, y: 1.95, w: cw - 0.7, h: 0.45, fontFace: H, fontSize: 20, bold: true, color: i ? ORANGE : BLUE, margin: 0, isTextBox: true });
    s.addText(items.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < items.length - 1 } })),
      { x: x + 0.35, y: 2.5, w: cw - 0.7, h: 4.0, valign: "top", fontFace: B, fontSize: 14, color: INK, margin: 0, paraSpaceAfter: 10, isTextBox: true });
  });
}

// 11. Next steps
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  pre(s, "Next steps");
  title(s, "Next: move from benchmark proof to daily use");
  const N = [
    ["Fix the context pack", "Mark rules the model made up as advisory, then rerun the 4-turn handoff test."],
    ["Test splitting work across cheaper models", "Compare one Sonnet call, a Haiku planner with parallel cheap workers, and Sonnet planning for Haiku workers."],
    ["Learn per language", "So the router can send Rust, Go and Java snippets to the free local qwen-coder."],
    ["Use it every day", "Run the proxy in daily work to collect real usage, then build workload prediction on that data."],
  ];
  N.forEach(([h, t], i) => {
    const y = 1.85 + i * 0.98;
    card(s, M, y, W - 2 * M, 0.82);
    num(s, i + 1, M + 0.25, y + 0.2);
    s.addText(h, { x: M + 0.95, y, w: 3.9, h: 0.82, fontFace: H, fontSize: 15, bold: true, color: INK, valign: "middle", margin: 0, isTextBox: true });
    s.addText(t, { x: M + 4.95, y, w: W - 2 * M - 5.2, h: 0.82, fontFace: B, fontSize: 13, color: MUTED, valign: "middle", margin: 0, isTextBox: true });
  });
  banner(s, "One-liner", "Tag the prompt, send it to the cheapest model that passes, grade it, learn.");
}

pres.writeFile({ fileName: __dirname + "/rce-sentinel-POC.pptx" }).then(f => console.log("wrote", f));
