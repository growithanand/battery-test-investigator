"use strict";

const byId = id => document.getElementById(id);
const presets = {
  capacity: "Compare the reported capacities for these discharges. Give the change in Ah and percent, with the earlier record as the reference.",
  extrema: "Give the minimum and maximum recorded current, voltage and temperature for both discharges, including their sample numbers and times. Also give the final voltage for each recording.",
  conditions: "Compare the reported capacities and inspect the recorded test conditions. Explain what the measurements and applicable protocol can establish and what remains unresolved."
};
let session = null;
let running = false;
let savedReport = null;

function text(id, value) { byId(id).textContent = value; }
function element(tag, value) { const node = document.createElement(tag); node.textContent = value; return node; }
function message(value, error = false) { text("message", value); byId("message").classList.toggle("error", error); }
function number(value, digits = 6) { return Number(value).toFixed(digits); }

function table(container, headings, rows) {
  const node = document.createElement("table");
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const value of headings) { const cell = element("th", value); cell.scope = "col"; headRow.append(cell); }
  head.append(headRow);
  const body = document.createElement("tbody");
  for (const values of rows) { const row = document.createElement("tr"); for (const value of values) row.append(element("td", value)); body.append(row); }
  node.append(head, body);
  byId(container).replaceChildren(node);
}

async function refreshSession() {
  try {
    const response = await fetch("/api/session", { cache: "no-store" });
    if (!response.ok) throw new Error();
    session = await response.json();
    const wait = session.retry_after_seconds;
    text("availability", !session.enabled ? "Live requests are off for this server session." :
      session.busy ? "An investigation is running. Please wait." :
      session.remaining === 0 ? "This server session has reached its five-run limit." :
      `${session.remaining} of ${session.limit} runs remaining${wait ? ` · ready in ${wait}s` : ""}`);
    byId("run-button").disabled = running || !session.enabled || session.busy || !session.remaining || wait > 0;
  } catch {
    text("availability", "The local server is unavailable. Start it and reload this page.");
    byId("run-button").disabled = true;
  }
}

function renderTraces(evidence) {
  byId("traces-result").hidden = !evidence;
  byId("load-windows").replaceChildren();
  if (!evidence) return;
  const rows = [];
  const signals = [["current_A", "Current (A)"], ["voltage_V", "Voltage (V)"], ["temperature_C", "Temperature (°C)"]];
  for (const side of ["earlier", "later"]) {
    const record = evidence[side];
    for (const [key, label] of signals) {
      for (const kind of ["minimum", "maximum"]) {
        const point = record.extrema_samples[key][kind];
        rows.push([record.discharge_index, label, kind, number(point[key]), point.sample_index, number(point.elapsed_s, 3)]);
      }
    }
    const point = record.last_sample;
    rows.push([record.discharge_index, "Voltage (V)", "final", number(point.voltage_V), point.sample_index, number(point.elapsed_s, 3)]);
  }
  table("trace-table", ["Discharge", "Signal", "Point", "Value", "Sample", "Time (s)"], rows);
  if (evidence.active_load) {
    const target = byId("load-windows");
    target.append(element("h4", "Sampled load windows"));
    for (const side of ["earlier", "later"]) {
      const window = evidence.active_load[side];
      const index = evidence[side].discharge_index;
      if (window.status !== "observed") { target.append(element("p", `Discharge ${index}: load window unresolved.`)); continue; }
      const start = window.load_on_bracket, end = window.load_off_bracket;
      target.append(element("p", `Discharge ${index}: ${window.on_load_sample_count} on-load samples over ${number(window.observed_span_s, 3)} s. Load-on bracket: samples ${start.before.sample_index}–${start.after.sample_index}; load-off bracket: samples ${end.before.sample_index}–${end.after.sample_index}.`));
    }
  }
}

function renderResult(result) {
  savedReport = result;
  byId("empty-state").hidden = true;
  byId("result").hidden = false;
  text("result-status", result.findings ? "Report ready · broader assessment open" : "Investigation stopped");
  text("run-time", `${number(result.seconds, 2)} s`);
  text("result-question", result.request.question);
  const comparison = result.comparison?.comparison;
  byId("capacity-result").hidden = !comparison;
  if (comparison) {
    text("capacity-change", `${number(comparison.capacity_change_Ah, 9)} Ah (${number(comparison.capacity_change_percent)}%)`);
    text("capacity-basis", `Relative to discharge ${comparison.earlier.discharge_index}, using its reported capacity as the denominator.`);
    table("capacity-table", ["Discharge", "Source operation", "Capacity (Ah)"],
      [comparison.earlier, comparison.later].map(record => [record.discharge_index, record.operation_index, number(record.reported_capacity_Ah, 9)]));
  }
  byId("follow-up-result").hidden = !result.follow_up;
  if (result.follow_up) text("follow-up", `${result.trace_evidence ? "Selected recordings inspected. " : "No additional trace inspection ran. "}${result.follow_up.reason}`);
  renderTraces(result.trace_evidence);
  text("protocol-status", ({ available: "Applicable experiment passages retrieved and source-verified.", partial: "Some applicable protocol evidence could not be verified.", unavailable: "Applicable protocol evidence is unavailable or could not be verified." })[result.protocol_status] || "No completed protocol lookup.");
  byId("claims").replaceChildren();
  for (const claim of result.findings?.protocol_claims || []) {
    const article = document.createElement("article"); article.className = "claim";
    const citation = element("span", claim.evidence_id); citation.className = "citation";
    article.append(element("p", claim.statement), element("blockquote", claim.quote), citation);
    byId("claims").append(article);
  }
  const limits = ["Comparability of test conditions is not established. Physical cause is not determined.", ...(result.findings?.limitations || [])];
  if (result.findings?.protocol_evidence_limitation) limits.push(result.findings.protocol_evidence_limitation);
  if (!result.findings) limits.push("This run has no accepted final findings or recommendation. Available measurements are shown above.");
  byId("limitations").replaceChildren(...limits.map(value => element("li", value)));
  byId("recommendation-result").hidden = !result.findings;
  if (result.findings) text("recommendation", result.findings.next_check);
  const usage = result.usage;
  text("usage", `${usage.input_tokens.toLocaleString()} input · ${usage.output_tokens.toLocaleString()} output tokens${usage.complete ? "" : " · usage incomplete"}`);
  text("full-report", result.report);
}

document.querySelectorAll("[data-preset]").forEach(button => {
  button.addEventListener("click", () => { byId("question").value = presets[button.dataset.preset]; byId("question").focus(); });
});

byId("investigation-form").addEventListener("submit", async event => {
  event.preventDefault();
  if (running || !session || byId("run-button").disabled) return;
  const early = Number(byId("early").value), late = Number(byId("late").value);
  const question = byId("question").value.trim();
  if (!question || !Number.isInteger(early) || !Number.isInteger(late) || early < 1 || early >= late || late > 168) {
    message("Enter a question and choose an earlier discharge before the later one, from 1 to 168.", true); return;
  }
  running = true;
  byId("run-button").disabled = true;
  byId("run-button").firstChild.textContent = "Investigating… ";
  document.querySelector(".output-panel").setAttribute("aria-busy", "true");
  byId("result").hidden = true;
  message("Reading the selected records and checking the evidence. The report will appear here when the run finishes.");
  try {
    const response = await fetch("/api/investigations", { method: "POST", headers: { "Content-Type": "application/json", "X-Investigation-Token": session.token }, body: JSON.stringify({ question, early, late }) });
    const data = await response.json();
    if (!response.ok) { message(typeof data.detail === "string" ? data.detail : "The investigation could not start.", true); return; }
    renderResult(data.result);
    message(data.result.findings ? "" : "The run stopped before final findings could be accepted. Review the available evidence below.", !data.result.findings);
  } catch {
    message("The response could not be read. The server may still be processing this run; check its status before submitting again.", true);
  } finally {
    running = false;
    byId("run-button").firstChild.textContent = "Run investigation ";
    document.querySelector(".output-panel").setAttribute("aria-busy", "false");
    await refreshSession();
  }
});

byId("download").addEventListener("click", () => {
  if (!savedReport) return;
  const url = URL.createObjectURL(new Blob([savedReport.report], { type: "text/markdown;charset=utf-8" }));
  const link = document.createElement("a"); link.href = url; link.download = `battery-investigation-${savedReport.run_id}.md`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

refreshSession();
setInterval(() => { if (!running) refreshSession(); }, 3000);
