const API = "";

const state = {
  datasetId: null,
  datasetFilename: null,
  columns: [],
  targetColumn: null,
  jobId: null,
  featureColumns: [],
  taskType: null,
  modality: "tabular",
  pipelineType: "tabular_automl",
  textColumn: null,
  datetimeColumn: null,
  completedSteps: new Set(),
  currentStep: 0,
  mleFeature: "quality",
};

const MLE_FEATURE_META = {
  quality: { title: "Dataset Quality", subtitle: "Score completeness, consistency, validity, and uniqueness." },
  leakage: { title: "Data Leakage", subtitle: "Find columns that leak target information before training." },
  review: { title: "AI Model Review", subtitle: "Overfitting, calibration, and class balance audit." },
  business: { title: "Business Insights", subtitle: "Executive summary and top feature drivers." },
  compare: { title: "Compare Runs", subtitle: "Side-by-side metrics for two training jobs." },
  modelcard: { title: "Model Card", subtitle: "Documentation, metrics, limits, and ethics." },
  active: { title: "Active Learning", subtitle: "Flag low-confidence rows from unlabeled data." },
};

const WORKFLOW_STEPS = [
  { step: 0, icon: "🏠", label: "Dashboard" },
  { step: 1, icon: "📁", label: "Upload" },
  { step: 2, icon: "📊", label: "Profile" },
  { step: 3, icon: "📈", label: "EDA" },
  { step: 4, icon: "🤖", label: "Train" },
  { step: 5, icon: "🎯", label: "Predict" },
  { step: 6, icon: "📉", label: "Drift" },
  { step: 7, icon: "🧠", label: "AI Advisor" },
  { step: 8, icon: "💬", label: "ML Chat" },
  { step: 9, icon: "⚙️", label: "ML Suite" },
];

const TRAIN_STAGES = [
  { key: "profiling", label: "Profiling dataset" },
  { key: "leakage", label: "Checking data leakage" },
  { key: "preprocessing", label: "Cleaning & feature engineering" },
  { key: "feature_selection", label: "Selecting best features" },
  { key: "model_selection", label: "Finding best model" },
  { key: "tuning", label: "Running Optuna tuning" },
  { key: "training", label: "Fitting final pipeline" },
  { key: "evaluation", label: "Evaluating on holdout" },
  { key: "explainability", label: "Building SHAP explanations" },
  { key: "review", label: "AI model review" },
  { key: "advisor", label: "Generating AI report" },
  { key: "model_card", label: "Creating model card" },
  { key: "tracking", label: "Logging to MLflow" },
  { key: "complete", label: "Complete" },
];

const stepsNav = document.getElementById("steps-nav");
const panels = document.querySelectorAll(".panel");
const apiStatus = document.getElementById("api-status");

const uploadZone = document.getElementById("upload-zone");
const fileInput = document.getElementById("file-input");
const browseBtn = document.getElementById("browse-btn");
const uploadResult = document.getElementById("upload-result");

const profileContent = document.getElementById("profile-content");

const edaTarget = document.getElementById("eda-target");
const edaBtn = document.getElementById("eda-btn");
const edaContent = document.getElementById("eda-content");

const pipelineForm = document.getElementById("pipeline-form");
const datasetIdInput = document.getElementById("dataset-id");
const targetSelect = document.getElementById("target-column");
const nTrialsInput = document.getElementById("n-trials");
const runPipelineBtn = document.getElementById("run-pipeline-btn");
const progressBar = document.getElementById("progress-bar");
const progressFill = document.getElementById("progress-fill");
const pipelineStatus = document.getElementById("pipeline-status");
const pipelineMetrics = document.getElementById("pipeline-metrics");
const featureImportance = document.getElementById("feature-importance");

const predictJobId = document.getElementById("predict-job-id");
const explainShap = document.getElementById("explain-shap");

function updateExplainCheckbox() {
  const row = explainShap?.closest("label");
  if (!row) return;
  let labelText = " Include SHAP explanations (single predict)";
  if (state.modality === "text") labelText = " Include keyword explanations (text model)";
  if (state.modality === "image") labelText = " Include class probabilities (image model)";
  for (const node of row.childNodes) {
    if (node.nodeType === Node.TEXT_NODE) node.textContent = labelText;
  }
}

const predictRecords = document.getElementById("predict-records");
const predictImagePreview = document.getElementById("predict-image-preview");
const predictBtn = document.getElementById("predict-btn");
const predictResult = document.getElementById("predict-result");
const batchZone = document.getElementById("batch-upload-zone");
const batchFileInput = document.getElementById("batch-file-input");
const batchBrowseBtn = document.getElementById("batch-browse-btn");

const driftJobId = document.getElementById("drift-job-id");
const autoRetrain = document.getElementById("auto-retrain");
const driftZone = document.getElementById("drift-upload-zone");
const driftFileInput = document.getElementById("drift-file-input");
const driftBrowseBtn = document.getElementById("drift-browse-btn");
const driftResult = document.getElementById("drift-result");

const advisorPostTrain = document.getElementById("advisor-post-train");
const advisorBtn = document.getElementById("advisor-btn");
const advisorInsights = document.getElementById("advisor-insights");

const chatStatus = document.getElementById("chat-status");
const chatLog = document.getElementById("chat-log");
const chatInput = document.getElementById("chat-input");
const chatBtn = document.getElementById("chat-btn");
let chatThreadId = null;

const profileIntelligence = document.getElementById("profile-intelligence");
const pipelineIntelligence = document.getElementById("pipeline-intelligence");
const counterfactualBtn = document.getElementById("counterfactual-btn");
const featureDriftMode = document.getElementById("feature-drift-mode");
const mleQualityBtn = document.getElementById("mle-quality-btn");
const mleLeakageBtn = document.getElementById("mle-leakage-btn");
const mleQualityResult = document.getElementById("mle-quality-result");
const mleLeakageResult = document.getElementById("mle-leakage-result");
const mleReviewBtn = document.getElementById("mle-review-btn");
const mleReviewResult = document.getElementById("mle-review-result");
const mleBusinessBtn = document.getElementById("mle-business-btn");
const mleBusinessResult = document.getElementById("mle-business-result");
const mleCompareBtn = document.getElementById("mle-compare-btn");
const mleCompareResult = document.getElementById("mle-compare-result");
const compareJobA = document.getElementById("compare-job-a");
const compareJobB = document.getElementById("compare-job-b");
const mleModelcardBtn = document.getElementById("mle-modelcard-btn");
const mleModelcardResult = document.getElementById("mle-modelcard-result");
const mleModelcardDownload = document.getElementById("mle-modelcard-download");
const activeLearningZone = document.getElementById("active-learning-zone");
const activeLearningInput = document.getElementById("active-learning-input");
const activeLearningBrowse = document.getElementById("active-learning-browse");
const mleActiveResult = document.getElementById("mle-active-result");
const workflowTracker = document.getElementById("workflow-tracker");
const trainingStagesEl = document.getElementById("training-stages");
const trainingProgressWrap = document.getElementById("training-progress-wrap");
const refreshDashboardBtn = document.getElementById("refresh-dashboard-btn");

const API_TIMEOUT = {
  health: 8000,
  default: 30000,
  eda: 120000,
  trainPoll: 30000,
  trainWatch: 90 * 60 * 1000,
  drift: 90000,
  mle: 60000,
  chat: 90000,
};

const TRAIN_POLL_INTERVAL_MS = 2000;

async function api(path, options = {}, timeoutMs = API_TIMEOUT.default) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${API}${path}`, { ...options, signal: controller.signal });
    const text = await res.text();
    let data;
    try { data = JSON.parse(text); } catch { data = text; }
    if (!res.ok) {
      const msg = data?.detail ? JSON.stringify(data.detail) : text;
      throw new Error(msg || res.statusText);
    }
    return data;
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error(
        `Request timed out after ${Math.round(timeoutMs / 1000)}s. ` +
        "Large datasets (EDA, training, drift) can take longer — wait and retry, or use a smaller CSV."
      );
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

function renderWorkflowTracker() {
  if (!workflowTracker) return;
  workflowTracker.innerHTML = WORKFLOW_STEPS.map((s) => {
    let icon = "○";
    let cls = "pending";
    if (state.completedSteps.has(s.step)) {
      icon = "✔";
      cls = "done";
    } else if (s.step === state.currentStep) {
      icon = "⏳";
      cls = "active";
    }
    return `<div class="wf-item ${cls}" data-goto-step="${s.step}"><span class="wf-icon">${icon}</span><span class="wf-label">${s.icon} ${s.label}</span></div>`;
  }).join("");
}

function setStep(n) {
  state.currentStep = n;
  stepsNav.querySelectorAll(".step").forEach((btn) => {
    btn.classList.toggle("active", Number(btn.dataset.step) === n);
  });
  panels.forEach((p) => p.classList.toggle("active", p.id === `step-${n}`));
  document.getElementById("mle-step-group")?.classList.toggle("expanded", n === 9);
  renderWorkflowTracker();
  if (n === 0) loadDashboard();
  if (n === 9) {
    updateMleContext();
    setMleFeature(state.mleFeature || "quality");
  } else {
    document.querySelectorAll(".mle-subnav-btn").forEach((btn) => btn.classList.remove("active"));
  }
}

function setMleFeature(feature) {
  if (!MLE_FEATURE_META[feature]) feature = "quality";
  state.mleFeature = feature;
  const meta = MLE_FEATURE_META[feature];
  const titleEl = document.getElementById("mle-pane-title");
  const subtitleEl = document.getElementById("mle-pane-subtitle");
  if (titleEl) titleEl.textContent = meta.title;
  if (subtitleEl) subtitleEl.textContent = meta.subtitle;
  document.querySelectorAll(".mle-subnav-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mleFeature === feature);
  });
  document.querySelectorAll(".mle-pane").forEach((pane) => {
    pane.classList.toggle("hidden", pane.dataset.mleFeature !== feature);
  });
}

function markStepDone(n) {
  state.completedSteps.add(n);
  const btn = stepsNav.querySelector(`[data-step="${n}"]`);
  if (btn) btn.classList.add("done");
  renderWorkflowTracker();
}
function renderDatasetCard(data, profile) {
  const p = profile || {};
  const missingCount = Object.values(p.missing_values || {}).filter((v) => v.n_missing > 0).length;
  const modality = data.modality || state.modality || "tabular";
  const pipeline = data.pipeline_type || state.pipelineType || "tabular_automl";
  return `
    <div class="dataset-card">
      <h3>Dataset</h3>
      <div class="dataset-name">${escapeHtml(data.filename || state.datasetFilename || "Uploaded file")}</div>
      <p><span class="badge badge-muted">${escapeHtml(modality)}</span> <span class="badge">${escapeHtml(pipeline)}</span></p>
      ${data.detection_reason ? `<p class="hint">${escapeHtml(data.detection_reason)}</p>` : ""}
      <div class="dataset-stats">
        <div class="dataset-stat"><div class="val">${(data.n_rows || p.n_rows || 0).toLocaleString()}</div><div class="lbl">Rows</div></div>
        <div class="dataset-stat"><div class="val">${data.n_columns || p.n_columns || 0}</div><div class="lbl">Columns</div></div>
        <div class="dataset-stat"><div class="val">${escapeHtml(modality)}</div><div class="lbl">Modality</div></div>
        <div class="dataset-stat"><div class="val">${missingCount}</div><div class="lbl">Cols w/ Missing</div></div>
        <div class="dataset-stat"><div class="val">${p.n_duplicate_rows ?? 0}</div><div class="lbl">Duplicates</div></div>
      </div>
      <p style="margin-top:1rem;font-size:0.82rem;color:var(--muted)">ID: <code>${escapeHtml(data.dataset_id)}</code></p>
      <button type="button" class="btn primary" style="margin-top:1rem" data-goto-step="2">📊 View profile →</button>
    </div>`;
}

function renderLeaderboardMedals(leaderboard, winner, taskType) {
  const entries = (Array.isArray(leaderboard)
    ? leaderboard
    : Object.entries(leaderboard || {}).sort((a, b) => b[1] - a[1])
  ).slice(0, 8);
  if (!entries.length) return "";
  const medals = ["🥇", "🥈", "🥉"];
  const scoreLabel = taskType === "regression" ? "CV R²" : "CV Score";
  const rows = entries.map((item, i) => {
    const name = Array.isArray(item) ? item[0] : item.model || item[0];
    const score = Array.isArray(item) ? item[1] : item.score;
    const medal = i < 3 ? medals[i] : `${i + 1}`;
    const isWinner = name === winner;
    const pct = typeof score === "number" ? (score <= 1 ? (score * 100).toFixed(1) + "%" : score.toFixed(4)) : score;
    return `<tr class="${isWinner ? "lb-winner" : ""}"><td><span class="lb-medal">${medal}</span></td><td>${escapeHtml(formatModelName(name))}</td><td>${pct}</td></tr>`;
  }).join("");
  return `
    <div class="leaderboard">
      <h3>Model Leaderboard</h3>
      <table class="leaderboard-table">
        <thead><tr><th>Rank</th><th>Model</th><th>${scoreLabel}</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

function renderTrainingStages(currentStage) {
  if (!trainingStagesEl) return;
  const idx = TRAIN_STAGES.findIndex((s) => s.key === currentStage);
  trainingStagesEl.innerHTML = TRAIN_STAGES.map((s, i) => {
    let cls = "pending";
    if (idx >= 0 && i < idx) cls = "done";
    else if (s.key === currentStage) cls = "active";
    const icon = cls === "done" ? "✔" : cls === "active" ? "⏳" : "○";
    return `<div class="train-stage ${cls}"><span>${icon}</span><span>${escapeHtml(s.label)}</span></div>`;
  }).join("");
}

function formatRelativeTime(iso) {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    const diff = Date.now() - d.getTime();
    const days = Math.floor(diff / 86400000);
    if (days === 0) return "Today";
    if (days === 1) return "Yesterday";
    if (days < 7) return `${days} days ago`;
    return d.toLocaleDateString();
  } catch { return "—"; }
}

async function loadDashboard() {
  try {
    const d = await api("/pipeline/dashboard", {}, 10000);
    document.getElementById("dash-models").textContent = d.models_trained ?? 0;
    document.getElementById("dash-datasets").textContent = d.datasets_uploaded ?? 0;
    const acc = d.best_accuracy ?? 0;
    document.getElementById("dash-accuracy").textContent = acc ? `${acc}%` : "—";
    document.getElementById("dash-latest").textContent = d.latest_model || "—";
    document.getElementById("dash-drift").textContent = d.active_drift_checks ?? 0;
    document.getElementById("dash-ai").textContent = d.ai_suggestions ?? 0;

    const histEl = document.getElementById("job-history");
    const history = d.job_history || [];
    if (!history.length) {
      histEl.innerHTML = '<p class="hint">No runs yet — upload a dataset and train your first model.</p>';
      return;
    }
    histEl.innerHTML = history.map((j) => {
      const score = j.score != null ? `${(j.score <= 1 ? j.score * 100 : j.score).toFixed(1)}%` : "—";
      const model = j.model_name ? formatModelName(j.model_name) : "—";
      const statusCls = j.status === "success" ? "success" : j.status === "failed" ? "failed" : "running";
      return `
        <div class="job-history-card" data-job-id="${escapeHtml(j.job_id)}">
          <div>
            <div class="jh-title">Run ${escapeHtml(j.job_id?.slice(-8) || "")} · ${escapeHtml(model)}</div>
            <div class="jh-meta">${escapeHtml(j.target_column || "—")} · ${escapeHtml(j.dataset_id || "")} · ${formatRelativeTime(j.created_at)}</div>
          </div>
          <div class="jh-score">${score}</div>
          <span class="jh-status ${statusCls}">${escapeHtml(j.status || "—")}</span>
        </div>`;
    }).join("");
    histEl.querySelectorAll(".job-history-card").forEach((card) => {
      card.addEventListener("click", () => {
        state.jobId = card.dataset.jobId;
        predictJobId.value = state.jobId;
        driftJobId.value = state.jobId;
        setStep(4);
      });
    });
  } catch {
    document.getElementById("job-history").innerHTML = '<p class="hint">Connect to API to load dashboard stats.</p>';
  }
}

refreshDashboardBtn?.addEventListener("click", () => loadDashboard());

function renderJson(el, obj) {
  el.innerHTML = `<pre>${escapeHtml(JSON.stringify(obj, null, 2))}</pre>`;
}
function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function dedupeDocs(docs) {
  const seen = new Set();
  return (docs || []).filter((d) => {
    const key = `${d.source}|${d.title}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function formatModelName(name) {
  return String(name || "").replace(/_/g, " ");
}

function renderMarkdown(md) {
  if (!md) return "";
  const lines = md.split("\n");
  const out = [];
  let inList = false;
  let inTable = false;
  let tableRows = [];

  function flushList() {
    if (inList) { out.push("</ul>"); inList = false; }
  }
  function flushTable() {
    if (!inTable || !tableRows.length) return;
    const rows = tableRows.filter((r) => !/^\|[\s\-:|]+\|$/.test(r.trim()));
    if (rows.length) {
      out.push("<table>");
      rows.forEach((row, i) => {
        const cells = row.split("|").slice(1, -1).map((c) => c.trim());
        const tag = i === 0 ? "th" : "td";
        out.push(`<tr>${cells.map((c) => `<${tag}>${c.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")}</${tag}>`).join("")}</tr>`);
      });
      out.push("</table>");
    }
    inTable = false;
    tableRows = [];
  }

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (line.startsWith("|") && line.includes("|")) {
      flushList();
      if (!inTable) inTable = true;
      tableRows.push(line);
      continue;
    }
    flushTable();
    if (/^### (.+)$/.test(line)) {
      flushList();
      out.push(`<h4>${line.replace(/^### /, "").replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")}</h4>`);
    } else if (/^## (.+)$/.test(line)) {
      flushList();
      out.push(`<h3>${line.replace(/^## /, "").replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")}</h3>`);
    } else if (/^# (.+)$/.test(line)) {
      flushList();
      out.push(`<h2>${line.replace(/^# /, "").replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")}</h2>`);
    } else if (/^[-*] (.+)$/.test(line)) {
      if (!inList) { out.push("<ul>"); inList = true; }
      out.push(`<li>${line.replace(/^[-*] /, "").replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")}</li>`);
    } else if (line === "") {
      flushList();
    } else if (line.startsWith("---")) {
      flushList();
      out.push("<hr>");
    } else {
      flushList();
      const text = line.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\*(.+?)\*/g, "<em>$1</em>");
      out.push(`<p>${text}</p>`);
    }
  }
  flushList();
  flushTable();
  return out.join("\n");
}

function renderLeaderboard(leaderboard, winner) {
  const entries = Object.entries(leaderboard || {}).sort((a, b) => b[1] - a[1]);
  if (!entries.length) return '<p class="hint">No CV leaderboard available.</p>';
  const max = entries[0][1] || 1;
  return entries.map(([name, score]) => `
    <div class="lb-row${name === winner ? " lb-winner" : ""}">
      <span class="lb-name" title="${escapeHtml(name)}">${escapeHtml(formatModelName(name))}</span>
      <div class="lb-bar-wrap"><div class="lb-bar" style="width:${Math.max(2, (score / max) * 100).toFixed(1)}%"></div></div>
      <span class="lb-score">${Number(score).toFixed(4)}</span>
    </div>
  `).join("");
}

function renderFeatureImportance(items) {
  if (!items?.length) return "";
  const max = Math.max(...items.map((f) => f.importance || f.value || 0), 0.001);
  return items.slice(0, 5).map((f) => {
    const val = f.importance ?? f.value ?? 0;
    return `
      <div class="fi-mini-row">
        <span>${escapeHtml(f.feature)}</span>
        <div class="fi-mini-bar-wrap"><div class="fi-mini-bar" style="width:${((val / max) * 100).toFixed(1)}%"></div></div>
        <span>${Number(val).toFixed(3)}</span>
      </div>`;
  }).join("");
}

function groupRagDocs(docs) {
  const groups = { runs: [], playbooks: [], other: [] };
  for (const d of docs) {
    if (d.chunk_type === "run_memory" || d.category === "similar_runs" || d.category === "models") {
      groups.runs.push(d);
    } else if (d.chunk_type === "playbook" || d.category === "tuning" || d.category === "drift") {
      groups.playbooks.push(d);
    } else {
      groups.other.push(d);
    }
  }
  return groups;
}

function renderAdvisorDashboard(data) {
  const fp = data.fingerprint || {};
  const metrics = fp.metrics || {};
  const leaderboard = fp.leaderboard || {};
  const winner = fp.winner_model || (data.model_recommendations_detail?.[0]?.model) || data.model_recommendations?.[0];
  const winnerDetail = (data.model_recommendations_detail || []).find((m) => m.model === winner) || data.model_recommendations_detail?.[0];
  const lbEntries = Object.entries(leaderboard).sort((a, b) => b[1] - a[1]);
  const cvScore = lbEntries[0]?.[1];
  const runnerUp = lbEntries[1];
  const confidencePct = ((data.confidence || 0) * 100).toFixed(0);
  const modeLabel = data.mode === "post_train" ? "Post-train" : "Pre-train";
  const taskLabel = fp.task_type || data.task_type || "—";
  const datasetLabel = fp.target_column
    ? `${fp.target_column} · ${taskLabel} · ${fp.row_bucket || "—"} (${(fp.n_rows || 0).toLocaleString()} rows)`
    : `${taskLabel} · ${fp.row_bucket || "—"}`;

  const llmLabel = data.llm_used
    ? (data.llm_provider || "LLM").toUpperCase()
    : null;

  const badges = [
    `<span class="advisor-badge mode">${escapeHtml(modeLabel)}</span>`,
    `<span class="advisor-badge ${data.critic_passed ? "ok" : "warn"}">${data.critic_passed ? "✓ Critic passed" : "⚠ Critic expanded search"}</span>`,
    data.web_evidence_used ? `<span class="advisor-badge warn">Web RAG</span>` : "",
    llmLabel ? `<span class="advisor-badge llm">${escapeHtml(llmLabel)} narrative</span>` : `<span class="advisor-badge">Rules report</span>`,
  ].filter(Boolean).join("");

  const kpis = [];
  if (cvScore != null) {
    kpis.push(`<div class="advisor-kpi highlight"><div class="kpi-value">${Number(cvScore).toFixed(4)}</div><div class="kpi-label">CV score</div></div>`);
  }
  if (metrics.roc_auc != null) {
    kpis.push(`<div class="advisor-kpi"><div class="kpi-value">${Number(metrics.roc_auc).toFixed(4)}</div><div class="kpi-label">Holdout ROC-AUC</div></div>`);
  }
  if (metrics.f1_weighted != null) {
    kpis.push(`<div class="advisor-kpi"><div class="kpi-value">${Number(metrics.f1_weighted).toFixed(4)}</div><div class="kpi-label">Holdout F1</div></div>`);
  } else if (metrics.r2 != null) {
    kpis.push(`<div class="advisor-kpi"><div class="kpi-value">${Number(metrics.r2).toFixed(4)}</div><div class="kpi-label">Holdout R²</div></div>`);
  }
  kpis.push(`<div class="advisor-kpi"><div class="kpi-value">${confidencePct}%</div><div class="kpi-label">Confidence</div></div>`);

  const params = fp.best_params || {};
  const paramChips = Object.entries(params).slice(0, 6).map(([k, v]) =>
    `<span class="param-chip">${escapeHtml(k)}=${escapeHtml(v)}</span>`
  ).join("");

  const insights = (data.data_insights || [])
    .map((i) => `<li>${escapeHtml(i.replace(/\*\*(.+?)\*\*/g, "$1"))}</li>`)
    .join("");

  const actions = (data.top_actions || [])
    .map((a) => `<li>${escapeHtml(typeof a === "string" ? a : a.action)}</li>`)
    .join("");

  const tips = (data.preprocessing_tips || []).slice(0, 5)
    .map((t) => `<li>${escapeHtml(t)}</li>`)
    .join("");

  const docs = dedupeDocs(data.retrieved_docs || []);
  const ragGroups = groupRagDocs(docs);
  let ragHtml = "";
  if (ragGroups.runs.length) {
    ragHtml += `<div class="rag-group"><div class="rag-group-title">Similar past runs</div>${ragGroups.runs.slice(0, 4).map((d) =>
      `<span class="rag-chip run" title="score ${d.score}">${escapeHtml(d.title)}</span>`
    ).join("")}</div>`;
  }
  if (ragGroups.playbooks.length) {
    ragHtml += `<div class="rag-group"><div class="rag-group-title">Playbooks</div>${ragGroups.playbooks.slice(0, 4).map((d) =>
      `<span class="rag-chip playbook" title="score ${d.score}">${escapeHtml(d.title)}</span>`
    ).join("")}</div>`;
  }

  const fiHtml = renderFeatureImportance(fp.feature_importance);
  const postTrainBlock = data.mode === "post_train" && winner ? `
    <div class="advisor-kpi-row">${kpis.join("")}</div>
    <div class="advisor-grid-2">
      <div class="advisor-card winner-card">
        <h4><span class="icon">🏆</span> Winning model</h4>
        <div class="winner-model">${escapeHtml(formatModelName(winner))}</div>
        <div class="winner-confidence">${confidencePct}<small>% confidence</small></div>
        ${runnerUp ? `<p class="winner-rationale">Beat ${escapeHtml(formatModelName(runnerUp[0]))} by ${(lbEntries[0][1] - runnerUp[1]).toFixed(4)} CV</p>` : ""}
        ${winnerDetail?.rationale ? `<p class="winner-rationale">${escapeHtml(winnerDetail.rationale)}</p>` : ""}
        ${paramChips ? `<div class="params-grid" style="margin-top:0.75rem">${paramChips}</div>` : ""}
      </div>
      <div class="advisor-card">
        <h4><span class="icon">📊</span> CV leaderboard</h4>
        ${renderLeaderboard(leaderboard, winner)}
      </div>
    </div>
    ${fiHtml ? `<div class="advisor-card"><h4><span class="icon">⚡</span> Top feature drivers</h4>${fiHtml}</div>` : ""}
  ` : `
    <div class="advisor-kpi-row">${kpis.join("")}</div>
    <div class="advisor-card">
      <h4><span class="icon">🎯</span> Recommended models</h4>
      <p>${(data.model_recommendations_detail || []).map((m) =>
        `<span class="tag" title="${escapeHtml(m.rationale || "")}">${escapeHtml(formatModelName(m.model))} (${((m.confidence || 0) * 100).toFixed(0)}%)</span>`
      ).join(" ") || "—"}</p>
    </div>
  `;

  const narrativeBlock = data.narrative_report ? `
    <div class="advisor-narrative">
      <button type="button" class="advisor-narrative-toggle" data-advisor-toggle aria-expanded="true">
        ${llmLabel ? `${escapeHtml(llmLabel)} executive summary` : "Full advisory report"} ▾
      </button>
      <div class="advisor-narrative-body" data-advisor-body>${renderMarkdown(data.narrative_report)}</div>
    </div>
  ` : "";

  const recChecks = [];
  if (winner) recChecks.push({ ok: true, text: `${formatModelName(winner)} selected as winning model` });
  if (!fp.is_imbalanced) recChecks.push({ ok: true, text: "Dataset target is reasonably balanced" });
  else recChecks.push({ ok: false, text: "Class imbalance detected — monitor F1/AUC" });
  if (!fp.has_missing) recChecks.push({ ok: true, text: "Missing values handled in pipeline" });
  (data.risks || []).slice(0, 2).forEach((r) => recChecks.push({ ok: false, text: r }));
  const nextStep = (data.top_actions || [])[0];
  const recCard = `
    <div class="advisor-rec-card">
      <h4>🧠 AI Advisor — Recommendation</h4>
      <ul class="advisor-checklist">
        ${recChecks.map((c) => `<li class="${c.ok ? "ok" : "warn"}">${escapeHtml(c.text)}</li>`).join("")}
      </ul>
      ${nextStep ? `<div class="advisor-next-step"><strong>Suggested next step:</strong> ${escapeHtml(typeof nextStep === "string" ? nextStep : nextStep.action)}</div>` : ""}
    </div>`;

  return `
    <div class="advisor-hero">
      <div class="advisor-hero-main">
        <h3>ML Advisor Report</h3>
        <p class="advisor-hero-sub">${escapeHtml(datasetLabel)}</p>
      </div>
      <div class="advisor-badges">${badges}</div>
    </div>
    ${recCard}
    ${postTrainBlock}
    ${data.risks?.length ? `<div class="advisor-risks"><strong>⚠ Risks</strong><ul>${data.risks.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul></div>` : ""}
    <div class="advisor-grid-2">
      <div class="advisor-card">
        <h4><span class="icon">💡</span> Key insights</h4>
        <ul class="advisor-insight-list">${insights || "<li>No insights generated.</li>"}</ul>
      </div>
      <div class="advisor-card">
        <h4><span class="icon">✅</span> Recommended actions</h4>
        ${actions ? `<ul class="advisor-actions">${actions}</ul>` : '<p class="hint">No actions suggested.</p>'}
        ${tips ? `<h4 style="margin-top:1rem"><span class="icon">📘</span> Playbook tips</h4><ul class="advisor-tips">${tips}</ul>` : ""}
      </div>
    </div>
    ${ragHtml ? `<div class="advisor-card"><h4><span class="icon">🔍</span> Evidence sources</h4>${ragHtml}</div>` : ""}
    ${narrativeBlock}
  `;
}

function sampleRecord(columns, target) {
  if (state.modality === "text" || state.textColumn) {
    const col = state.textColumn || "review_text";
    return { [col]: "This product is absolutely wonderful and exceeded all my expectations" };
  }
  if (state.modality === "image") {
    return {
      image_path: "sample_data/image_samples/ok/ok_000.png",
    };
  }
  const features = columns.filter((c) => c !== target);
  const row = {};
  for (const col of features) {
    if (/age|count|rooms|tenure|charges|calls|income|value|median/i.test(col)) row[col] = 0;
    else if (/date/i.test(col)) row[col] = "2024-01-01";
    else if (/id/i.test(col)) row[col] = "SAMPLE001";
    else if (/churn|contract|service/i.test(col)) row[col] = "example";
    else row[col] = 1.0;
  }
  return row;
}

function defaultTargets() {
  if (state.modality === "image" || state.modality === "documents") return ["label"];
  return [];
}

function updateEdaButtonState() {
  if (!edaBtn) return;
  edaBtn.disabled = state.modality !== "tabular" || !state.datasetId || !state.columns.length;
}

function syncTargetSelects(suggested) {
  const cols = suggested?.length ? suggested : (state.columns.length ? state.columns : defaultTargets());
  const opts = cols.map((c) => `<option value="${c}">${c}</option>`).join("");
  targetSelect.innerHTML = opts;
  edaTarget.innerHTML = state.columns.length ? state.columns.map((c) => `<option value="${c}">${c}</option>`).join("") : opts;
  targetSelect.disabled = !cols.length;
  edaTarget.disabled = !state.columns.length;
}

async function checkHealth() {
  apiStatus.textContent = "Checking API…";
  apiStatus.className = "badge badge-muted";
  try {
    await api("/health", {}, API_TIMEOUT.health);
    apiStatus.textContent = "API online";
    apiStatus.className = "badge badge-ok";
  } catch (err) {
    apiStatus.textContent = "API offline — start server";
    apiStatus.className = "badge badge-err";
    apiStatus.title = err.message || "";
  }
}

stepsNav.addEventListener("click", (e) => {
  const btn = e.target.closest(".step");
  if (btn) setStep(Number(btn.dataset.step));
});

document.getElementById("mle-subnav")?.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-mle-feature]");
  if (!btn) return;
  setStep(9);
  setMleFeature(btn.dataset.mleFeature);
});

document.addEventListener("click", (e) => {
  const goto = e.target.closest("[data-goto-step]");
  if (goto) setStep(Number(goto.dataset.gotoStep));
});

function setupDropZone(zone, input, onFile) {
  zone.addEventListener("click", (e) => { if (!e.target.closest(".text-btn")) input.click(); });
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("dragover"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
    if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => { if (input.files[0]) onFile(input.files[0]); });
}

async function uploadFile(file) {
  const allowed = [".csv", ".tsv", ".xlsx", ".xls", ".pdf", ".txt", ".jsonl", ".zip", ".jpg", ".jpeg", ".png"];
  const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (!allowed.includes(ext)) {
    alert(`Unsupported file type. Allowed: ${allowed.join(", ")}`);
    return;
  }
  uploadResult.innerHTML = '<span class="spinner"></span> Uploading and detecting data type…';
  show(uploadResult);
  const form = new FormData();
  form.append("file", file);
  try {
    const endpoint = ext === ".csv" ? "/datasets/upload" : "/multimodal/upload";
    const data = await api(endpoint, { method: "POST", body: form }, API_TIMEOUT.eda);
    state.datasetId = data.dataset_id;
    state.datasetFilename = data.filename || file.name;
    state.columns = data.columns || [];
    state.modality = data.modality || "tabular";
    state.pipelineType = data.pipeline_type || "tabular_automl";
    state.textColumn = data.text_column || null;
    state.datetimeColumn = data.datetime_column || null;
    updateExplainCheckbox();
    datasetIdInput.value = data.dataset_id;
    syncTargetSelects(data.suggested_targets);
    runPipelineBtn.disabled = false;
    updateEdaButtonState();
    advisorBtn.disabled = false;
    let profile = null;
    if (state.modality === "tabular" && data.columns?.length) {
      const profileData = await api(`/datasets/${data.dataset_id}/profile`);
      profile = profileData.profile;
      await loadProfile(data.dataset_id);
    } else {
      profileContent.innerHTML = `<p class="hint">Modality: <strong>${escapeHtml(state.modality)}</strong> — pipeline <strong>${escapeHtml(state.pipelineType)}</strong> will run automatically on train.</p>
        <p style="margin-top:1rem">
          <button type="button" class="btn" id="profile-quality-btn">Dataset quality score</button>
          <button type="button" class="btn" id="profile-leakage-btn">Check leakage</button>
        </p>`;
      document.getElementById("profile-quality-btn")?.addEventListener("click", () => runDatasetQuality(profileIntelligence));
      document.getElementById("profile-leakage-btn")?.addEventListener("click", () => runLeakageCheck(profileIntelligence));
    }
    uploadResult.innerHTML = renderDatasetCard(data, profile);
    show(uploadResult);
    markStepDone(1);
    setStep(state.modality === "tabular" ? 2 : 4);
  } catch (err) {
    uploadResult.innerHTML = `<span class="status-failed">Error: ${escapeHtml(err.message)}</span>`;
  }
}

browseBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
setupDropZone(uploadZone, fileInput, uploadFile);

async function loadProfile(datasetId) {
  profileContent.innerHTML = '<span class="spinner"></span> Loading profile…';
  try {
    const data = await api(`/datasets/${datasetId}/profile`);
    const p = data.profile;
    const missing = Object.entries(p.missing_values || {}).filter(([, v]) => v.n_missing > 0).slice(0, 8);
    profileContent.innerHTML = `
      <table class="profile-table">
        <tr><th>Rows</th><td>${p.n_rows}</td><th>Columns</th><td>${p.n_columns}</td></tr>
        <tr><th>Numeric</th><td colspan="3">${(p.numeric_columns || []).join(", ") || "—"}</td></tr>
        <tr><th>Categorical</th><td colspan="3">${(p.categorical_columns || []).join(", ") || "—"}</td></tr>
        <tr><th>Duplicates</th><td colspan="3">${p.n_duplicate_rows}</td></tr>
      </table>
      ${missing.length ? `<p style="margin-top:1rem;color:var(--muted)">Missing: ${missing.map(([c, v]) => `${c} (${(v.pct_missing * 100).toFixed(1)}%)`).join(", ")}</p>` : ""}
      <p style="margin-top:1rem">
        <button type="button" class="btn" data-goto-step="3">Run EDA next</button>
        <button type="button" class="btn" id="profile-quality-btn">Dataset quality score</button>
        <button type="button" class="btn" id="profile-leakage-btn">Check leakage</button>
      </p>
    `;
    document.getElementById("profile-quality-btn")?.addEventListener("click", () => runDatasetQuality(profileIntelligence));
    document.getElementById("profile-leakage-btn")?.addEventListener("click", () => runLeakageCheck(profileIntelligence));
    enableMleButtons();
    markStepDone(2);
  } catch (err) {
    profileContent.innerHTML = `<span class="status-failed">${escapeHtml(err.message)}</span>`;
  }
}

edaBtn.addEventListener("click", async () => {
  const target = edaTarget.value || targetSelect.value;
  if (!state.datasetId || !target) return;
  state.targetColumn = target;
  edaBtn.disabled = true;
  edaContent.innerHTML = '<span class="spinner"></span> Running EDA (charts for top features — may take 30–60s on large CSVs)…';
  show(edaContent);
  try {
    const data = await api(`/datasets/${state.datasetId}/eda?target_column=${encodeURIComponent(target)}`, {}, API_TIMEOUT.eda);
    edaContent.innerHTML = renderEDA(data.eda);
    markStepDone(3);
  } catch (err) {
    edaContent.innerHTML = `<span class="status-failed">${escapeHtml(err.message)}</span>`;
  }
  updateEdaButtonState();
});

function renderEDA(eda) {
  let html = `<p><strong>Task:</strong> ${eda.task_type} · <strong>Target:</strong> ${eda.target_column}</p>`;
  const ta = eda.target_analysis;
  if (ta.type === "classification") {
    if (ta.is_imbalanced) {
      html += `<p class="status-running">⚠ Imbalanced target (${(ta.imbalance_ratio * 100).toFixed(0)}% majority class)</p>`;
    }
  } else {
    html += `<div class="eda-card"><h4>Target stats</h4><p>Mean: ${ta.mean?.toFixed(2)} · Std: ${ta.std?.toFixed(2)} · Range: [${ta.min?.toFixed(2)}, ${ta.max?.toFixed(2)}]</p></div>`;
  }

  if (eda.charts?.length) {
    html += `<div class="eda-charts-section"><h3>Charts (features vs target)</h3><div class="eda-charts-grid">`;
    for (const ch of eda.charts) {
      const badge = ch.chart_type ? `<span class="chart-badge">${escapeHtml(ch.chart_type)}</span>` : "";
      html += `<figure class="eda-chart-card">${badge}<img src="${ch.url}" alt="${escapeHtml(ch.title)}" loading="lazy" /><figcaption>${escapeHtml(ch.title)}</figcaption></figure>`;
    }
    html += `</div></div>`;
  }

  if (eda.correlation_with_target?.length) {
    html += `<div class="eda-card"><h4>Top correlations with target</h4><table class="profile-table">`;
    for (const c of eda.correlation_with_target.slice(0, 8)) {
      html += `<tr><td>${escapeHtml(c.feature)}</td><td>${c.correlation}</td></tr>`;
    }
    html += `</table></div>`;
  }

  html += `<div class="eda-grid">`;
  for (const [col, info] of Object.entries(eda.numeric_features || {})) {
    html += `<div class="eda-card"><h4>${escapeHtml(col)} (numeric)</h4>`;
    html += `<p>mean=${info.mean} · skew=${info.skew} · missing ${(info.missing_pct * 100).toFixed(1)}%</p>`;
    if (info.mean_by_class) {
      for (const [cls, v] of Object.entries(info.mean_by_class)) {
        html += `<p>${escapeHtml(cls)}: avg ${v}</p>`;
      }
    }
    html += `</div>`;
  }
  for (const [col, info] of Object.entries(eda.categorical_features || {})) {
    html += `<div class="eda-card"><h4>${escapeHtml(col)} (categorical)</h4>`;
    html += `<p>${info.n_unique} unique values</p>`;
    if (info.target_rate_by_category) {
      const top = Object.entries(info.target_rate_by_category).slice(0, 3);
      for (const [cat, rates] of top) {
        const rateStr = Object.entries(rates).map(([k, v]) => `${k}:${(v * 100).toFixed(0)}%`).join(", ");
        html += `<p><small>${escapeHtml(cat)} → ${rateStr}</small></p>`;
      }
    }
    html += `</div>`;
  }
  html += `</div>`;
  return html;
}

function formatProbability(prob, labelClasses) {
  if (prob == null) return "";
  if (Array.isArray(prob)) {
    if (labelClasses?.length === prob.length) {
      return labelClasses.map((name, i) => `${name}: ${(prob[i] * 100).toFixed(1)}%`).join(", ");
    }
    return prob.map((v, i) => `${i}: ${(v * 100).toFixed(1)}%`).join(", ");
  }
  if (typeof prob === "object") {
    return Object.entries(prob)
      .sort((a, b) => b[1] - a[1])
      .map(([k, v]) => `${k}: ${(v * 100).toFixed(1)}%`)
      .join(", ");
  }
  return String(prob);
}

function extractImagePathsFromRecords(records) {
  if (!Array.isArray(records)) return [];
  const paths = [];
  for (const row of records) {
    if (!row || typeof row !== "object") continue;
    for (const key of ["image_path", "path", "file", "filepath", "image"]) {
      if (row[key]) {
        paths.push(String(row[key]));
        break;
      }
    }
  }
  return paths;
}

function imagePreviewUrl(path) {
  return `/predict/preview-image?path=${encodeURIComponent(path)}`;
}

function renderInlineImagePreview(paths) {
  if (!predictImagePreview) return;
  if (!paths.length || state.modality !== "image") {
    hide(predictImagePreview);
    predictImagePreview.innerHTML = "";
    return;
  }
  predictImagePreview.innerHTML = paths.map((path, i) => `
    <div class="predict-image-card">
      <img src="${imagePreviewUrl(path)}" alt="Preview ${i + 1}" onerror="this.alt='Could not load image'; this.style.opacity='0.4';" />
      <div class="path">${escapeHtml(path)}</div>
    </div>
  `).join("");
  show(predictImagePreview);
}

function updatePredictImagePreview() {
  if (state.modality !== "image") {
    renderInlineImagePreview([]);
    return;
  }
  try {
    const records = JSON.parse(predictRecords.value || "[]");
    renderInlineImagePreview(extractImagePathsFromRecords(records));
  } catch {
    renderInlineImagePreview([]);
  }
}

function renderPredictResult(data) {
  const labelClasses = data.label_classes || [];
  const imageRows = data.image_rows || [];

  if (imageRows.length) {
    const cards = imageRows.map((row, i) => {
      const probStr = formatProbability(row.probabilities, labelClasses);
      const preview = row.preview_url || imagePreviewUrl(row.image_path || "");
      return `
        <div class="predict-result-card">
          <img src="${preview}" alt="Input image ${i + 1}" />
          <div class="pred-label">Prediction: <span class="tag">${escapeHtml(String(row.prediction ?? data.predictions?.[i] ?? "—"))}</span></div>
          ${probStr ? `<p class="muted">${escapeHtml(probStr)}</p>` : ""}
          <div class="path">${escapeHtml(row.image_path || "")}</div>
        </div>`;
    }).join("");
    let html = `
      <div class="predict-summary">
        <p><strong>Model:</strong> <span class="tag">${escapeHtml(data.model_name || "unknown")}</span></p>
        <p><strong>Task:</strong> ${escapeHtml(data.task_type || "—")} · <strong>Job:</strong> <code>${escapeHtml(data.job_id)}</code></p>
        ${labelClasses.length ? `<p><strong>Classes:</strong> ${labelClasses.map((c) => `<span class="tag">${escapeHtml(c)}</span>`).join(" ")}</p>` : ""}
        <div class="predict-result-images">${cards}</div>
      </div>`;
    if (data.explanations?.length) {
      html += `<details><summary>Explanations</summary><pre>${escapeHtml(JSON.stringify(data.explanations, null, 2))}</pre></details>`;
    }
    return html;
  }

  const preds = (data.predictions || []).map((p, i) => {
    const probStr = formatProbability(data.probabilities?.[i], labelClasses);
    return `<li><strong>Row ${i + 1}:</strong> ${escapeHtml(String(p))}${probStr ? ` <span class="muted">(${escapeHtml(probStr)})</span>` : ""}</li>`;
  }).join("");
  let html = `
    <div class="predict-summary">
      <p><strong>Model:</strong> <span class="tag">${escapeHtml(data.model_name || "unknown")}</span></p>
      <p><strong>Task:</strong> ${escapeHtml(data.task_type || "—")} · <strong>Job:</strong> <code>${escapeHtml(data.job_id)}</code></p>
      ${labelClasses.length ? `<p><strong>Classes:</strong> ${labelClasses.map((c) => `<span class="tag">${escapeHtml(c)}</span>`).join(" ")}</p>` : ""}
      <ul class="predict-list">${preds}</ul>
    </div>`;
  if (data.explanations?.length) {
    html += `<details><summary>SHAP explanations</summary><pre>${escapeHtml(JSON.stringify(data.explanations, null, 2))}</pre></details>`;
  }
  return html;
}

pipelineForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const target = targetSelect.value;
  if (!state.datasetId || !target) return;
  state.targetColumn = target;
  runPipelineBtn.disabled = true;
  show(trainingProgressWrap);
  progressFill.style.width = "0%";
  renderTrainingStages("profiling");
  pipelineStatus.innerHTML = '<span class="spinner"></span> <span class="status-running">Starting pipeline… large CSVs may take 10–30 minutes.</span>';
  show(pipelineStatus);
  hide(pipelineMetrics);
  hide(featureImportance);
  try {
    const payload = {
      dataset_id: state.datasetId,
      target_column: target,
      n_trials: Number(nTrialsInput.value) || 15,
    };
    if (state.modality && state.modality !== "tabular") payload.modality = state.modality;
    if (state.textColumn) payload.text_column = state.textColumn;
    if (state.datetimeColumn) payload.datetime_column = state.datetimeColumn;
    const data = await api("/pipeline/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.jobId = data.job_id;
    predictJobId.value = data.job_id;
    driftJobId.value = data.job_id;
    await pollJob(data.job_id);
  } catch (err) {
    pipelineStatus.innerHTML = `<span class="status-failed">Error: ${escapeHtml(err.message)}</span>`;
    runPipelineBtn.disabled = false;
  }
});

function formatElapsed(seconds) {
  if (seconds < 60) return `${seconds}s`;
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}m ${secs}s`;
}

async function pollJob(jobId, watchMs = API_TIMEOUT.trainWatch) {
  const started = Date.now();
  runPipelineBtn.disabled = true;
  show(trainingProgressWrap);

  while (Date.now() - started < watchMs) {
    try {
      const job = await api(`/pipeline/jobs/${jobId}`, {}, API_TIMEOUT.trainPoll);
      const prog = job.progress || {};
      const pct = prog.pct || 0;
      const msg = prog.message || job.status;
      const stage = prog.stage || job.status;
      const elapsed = formatElapsed(Math.round((Date.now() - started) / 1000));
      progressFill.style.width = `${pct}%`;
      renderTrainingStages(stage);
      pipelineStatus.innerHTML =
        `<span class="spinner"></span> <span class="status-running"><strong>${escapeHtml(stage)}</strong> — ${escapeHtml(msg)}</span> ` +
        `<span class="hint">(${elapsed} elapsed)</span>`;

      if (job.status === "success") {
        const r = job.result || {};
        state.taskType = r.task_type;
        if (r.modality) state.modality = r.modality;
        if (r.text_column) state.textColumn = r.text_column;
        updateExplainCheckbox();
        progressFill.style.width = "100%";
        renderTrainingStages("complete");
        pipelineStatus.innerHTML = `<span class="status-success">✔ Training complete!</span> Model: <strong>${escapeHtml(formatModelName(r.model_name))}</strong> · Job: <code>${jobId}</code>`;

        if (r.metrics) {
          show(pipelineMetrics);
          let metricsHtml = Object.entries(r.metrics)
            .map(([k, v]) => `<div class="metric-card"><div class="value">${typeof v === "number" ? v.toFixed(4) : v}</div><div class="label">${k.replace(/_/g, " ")}</div></div>`)
            .join("");
          metricsHtml += renderLeaderboardMedals(r.model_leaderboard || r.baseline_scores, r.model_name, r.task_type);
          pipelineMetrics.innerHTML = metricsHtml;
        }

        await loadFeatureImportance(jobId);
        renderPipelineIntelligence(r);
        predictRecords.value = JSON.stringify([sampleRecord(state.columns, state.targetColumn)], null, 2);
        updatePredictImagePreview();
        predictBtn.disabled = false;
        counterfactualBtn.disabled = state.modality !== "tabular";
        counterfactualBtn.title = state.modality === "tabular"
          ? "What small feature changes would flip this prediction?"
          : "Tabular models only (CSV features)";
        enableMleButtons();
        markStepDone(4);
        setStep(5);
        runPipelineBtn.disabled = false;
        return;
      }
      if (job.status === "failed") {
        pipelineStatus.innerHTML = `<span class="status-failed">Failed: ${escapeHtml(job.error || "Unknown")}</span>`;
        runPipelineBtn.disabled = false;
        return;
      }
    } catch (err) {
      const elapsed = formatElapsed(Math.round((Date.now() - started) / 1000));
      pipelineStatus.innerHTML =
        `<span class="status-running">Connection hiccup (${escapeHtml(err.message)}). Retrying…</span> ` +
        `<span class="hint">(${elapsed} elapsed)</span>`;
    }
    await new Promise((r) => setTimeout(r, TRAIN_POLL_INTERVAL_MS));
  }

  const elapsed = formatElapsed(Math.round((Date.now() - started) / 1000));
  pipelineStatus.innerHTML = `
    <span class="status-running">Training is still running on the server (${elapsed} watched).</span>
    <p class="hint">Large datasets (7k+ rows, many features) often need <strong>10–30 minutes</strong>. Optuna tuning is the slowest step — this is normal.</p>
    <button type="button" class="btn primary" id="resume-poll-btn">Keep watching this job</button>
    <span class="hint">Job ID: <code>${escapeHtml(jobId)}</code></span>
  `;
  document.getElementById("resume-poll-btn")?.addEventListener("click", () => pollJob(jobId));
  runPipelineBtn.disabled = false;
}

async function loadFeatureImportance(jobId) {
  try {
    const data = await api(`/predict/${jobId}/feature-importance`);
    const max = data.features[0]?.importance || 1;
    let html = `<h3>⚡ Feature Importance</h3><div class="fi-chart">`;
    for (const f of data.features.slice(0, 10)) {
      const w = (f.importance / max) * 100;
      html += `
        <div class="fi-chart-row">
          <span>${escapeHtml(f.feature)}</span>
          <div class="fi-chart-bar-wrap"><div class="fi-chart-bar" style="width:${w}%"></div></div>
          <span>${f.importance.toFixed(3)}</span>
        </div>`;
    }
    html += `</div>`;
    featureImportance.innerHTML = html;
    show(featureImportance);
  } catch { /* optional */ }
}

predictRecords?.addEventListener("input", () => {
  clearTimeout(predictRecords._previewTimer);
  predictRecords._previewTimer = setTimeout(updatePredictImagePreview, 300);
});

predictBtn.addEventListener("click", async () => {
  if (!state.jobId) return;
  let records;
  try {
    records = JSON.parse(predictRecords.value);
    if (!Array.isArray(records)) throw new Error("Must be JSON array");
  } catch (err) { alert("Invalid JSON: " + err.message); return; }

  predictBtn.disabled = true;
  predictResult.innerHTML = '<span class="spinner"></span> Predicting…';
  show(predictResult);
  try {
    const data = await api(`/predict/${state.jobId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ records, explain: explainShap.checked }),
    });
    predictResult.innerHTML = renderPredictResult(data);
    markStepDone(5);
  } catch (err) {
    predictResult.innerHTML = `<span class="status-failed">${escapeHtml(err.message)}</span>`;
  }
  predictBtn.disabled = false;
});

batchBrowseBtn.addEventListener("click", (e) => { e.stopPropagation(); batchFileInput.click(); });
async function batchPredict(file) {
  if (!state.jobId) { alert("Train a model first"); return; }
  predictResult.innerHTML = '<span class="spinner"></span> Batch predicting…';
  show(predictResult);
  const form = new FormData();
  form.append("file", file);
  try {
    const data = await api(`/predict/${state.jobId}/batch`, { method: "POST", body: form });
    predictResult.innerHTML = `
      <p class="status-success">Predicted <strong>${data.n_rows}</strong> rows using <span class="tag">${escapeHtml(data.model_name || "unknown")}</span> (${escapeHtml(data.task_type || "")}).</p>
      <p><a href="${data.download_path}" class="link-btn" download>Download predictions CSV →</a></p>
      <pre>${escapeHtml(JSON.stringify(data.preview, null, 2))}</pre>
    `;
    markStepDone(5);
  } catch (err) {
    predictResult.innerHTML = `<span class="status-failed">${escapeHtml(err.message)}</span>`;
  }
}
setupDropZone(batchZone, batchFileInput, batchPredict);

driftBrowseBtn.addEventListener("click", (e) => { e.stopPropagation(); driftFileInput.click(); });
async function checkDrift(file) {
  if (!state.jobId) { alert("Complete training first"); return; }
  driftResult.innerHTML = '<span class="spinner"></span> Running drift analysis…';
  show(driftResult);
  const form = new FormData();
  form.append("file", file);
  const retrain = autoRetrain.checked ? "&auto_retrain=true" : "";
  const endpoint = featureDriftMode?.checked
    ? `/mle/monitor/drift/${state.jobId}/features`
    : `/monitor/drift/${state.jobId}?n_trials=15${retrain}`;
  try {
    const data = await api(endpoint, { method: "POST", body: form }, API_TIMEOUT.drift);
    if (featureDriftMode?.checked) {
      driftResult.innerHTML = renderFeatureDrift(data);
    } else {
      const driftClass = data.dataset_drift_detected ? "status-failed" : "status-success";
      let html = `
        <p class="${driftClass}"><strong>Drift detected: ${data.dataset_drift_detected}</strong></p>
        <p>Drift share: ${(data.drift_share * 100).toFixed(1)}% · Columns: ${data.number_of_drifted_columns}/${data.number_of_columns}</p>
        <p><a href="/monitor/drift/${state.jobId}/report" target="_blank" class="link-btn">Open HTML report →</a></p>
      `;
      if (data.retrain_message) html += `<p class="status-running">${escapeHtml(data.retrain_message)}</p>`;
      if (data.retrain_job_id) {
        html += `<p>New retrain job: <code>${data.retrain_job_id}</code></p>`;
        state.jobId = data.retrain_job_id;
        predictJobId.value = data.retrain_job_id;
        driftJobId.value = data.retrain_job_id;
      }
      driftResult.innerHTML = html;
    }
    markStepDone(6);
  } catch (err) {
    driftResult.innerHTML = `<span class="status-failed">${escapeHtml(err.message)}</span>`;
  }
}
setupDropZone(driftZone, driftFileInput, checkDrift);

advisorBtn.addEventListener("click", async () => {
  advisorBtn.disabled = true;
  advisorInsights.innerHTML = '<span class="spinner"></span> Running evidence-grounded advisor…';
  show(advisorInsights);

  const body = {};
  if (advisorPostTrain.checked && state.jobId) {
    body.job_id = state.jobId;
  } else if (state.datasetId) {
    body.dataset_id = state.datasetId;
    body.target_column = edaTarget.value || targetSelect.value;
  } else return;

  try {
    const data = await api("/agent/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    advisorInsights.innerHTML = renderAdvisorDashboard(data);
    advisorInsights.querySelector("[data-advisor-toggle]")?.addEventListener("click", (e) => {
      const btn = e.currentTarget;
      const bodyEl = advisorInsights.querySelector("[data-advisor-body]");
      const open = btn.getAttribute("aria-expanded") === "true";
      btn.setAttribute("aria-expanded", open ? "false" : "true");
      bodyEl.style.display = open ? "none" : "block";
      btn.textContent = btn.textContent.replace(/▾|▸/, open ? "▸" : "▾");
    });
    markStepDone(7);
  } catch (err) {
    advisorInsights.innerHTML = `<span class="status-failed">${escapeHtml(err.message)}</span>`;
  }
  advisorBtn.disabled = false;
});

function appendChat(role, text, meta = null) {
  removeChatTyping();
  const wrap = document.createElement("div");
  if (role === "user") {
    wrap.className = "chat-bubble user";
    wrap.textContent = text;
  } else if (role === "error") {
    wrap.className = "chat-bubble error";
    wrap.textContent = text;
  } else {
    wrap.className = "chat-bubble bot";
    if (meta) {
      const metaEl = document.createElement("div");
      metaEl.className = "chat-bubble-meta";
      metaEl.textContent = meta;
      wrap.appendChild(metaEl);
    }
    const body = document.createElement("div");
    body.className = "chat-bubble-body";
    body.innerHTML = renderMarkdown(text);
    wrap.appendChild(body);
  }
  chatLog.appendChild(wrap);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function appendChatFallbackNotice(reason) {
  const div = document.createElement("div");
  div.className = "chat-fallback-notice";
  div.textContent = reason
    ? `LLM unavailable — showing formatted rule-based answer. (${reason.slice(0, 120)})`
    : "LLM unavailable — showing formatted rule-based answer.";
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function showChatTyping() {
  removeChatTyping();
  const div = document.createElement("div");
  div.className = "chat-typing";
  div.id = "chat-typing-indicator";
  div.innerHTML = '<span class="spinner"></span> Thinking…';
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function removeChatTyping() {
  document.getElementById("chat-typing-indicator")?.remove();
}

function resolveChatPrompt(template) {
  let msg = template;
  if (state.jobId && /last job|this job|my job/i.test(msg)) {
    msg = msg.replace(/last job|this job|my job/gi, state.jobId);
  }
  return msg;
}

async function sendChatMessage(rawMsg) {
  const msg = rawMsg.trim();
  if (!msg) return;
  chatBtn.disabled = true;
  appendChat("user", msg);
  chatInput.value = "";
  showChatTyping();
  try {
    const data = await api("/chat/message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg, thread_id: chatThreadId }),
    }, 120000);
    chatThreadId = data.thread_id;
    removeChatTyping();
    if (data.fallback && data.provider === "rules") {
      appendChatFallbackNotice(data.fallback_reason);
      appendChat("bot", data.response || "(no response)", "Rule-based assistant");
    } else {
      const meta = `${(data.provider || "assistant").toUpperCase()} · ${chatMetaModel || "LLM"}`;
      appendChat("bot", data.response || "(no response)", meta);
    }
    markStepDone(8);
  } catch (err) {
    removeChatTyping();
    appendChat("error", "Error: " + err.message);
  }
  chatBtn.disabled = false;
}

let chatMetaModel = "";

async function initChat() {
  try {
    const st = await api("/chat/status");
    chatMetaModel = st.model || "";
    chatStatus.className = "chat-status-pill";
    if (st.available) {
      if (st.provider === "rules") {
        chatStatus.textContent = "Rule-based chat — no LLM key configured";
        chatStatus.classList.add("rules");
      } else if (st.ready === false) {
        chatStatus.textContent = `LLM not ready — ${st.hint || "check API key"}`;
        chatStatus.classList.add("error");
      } else {
        const advisorNote = st.advisor_model && st.advisor_model !== st.model
          ? ` · advisor: ${st.advisor_model}`
          : "";
        chatStatus.textContent = `Ready — ${st.provider} / ${st.model}${advisorNote}`;
        chatStatus.classList.add("ready");
      }
      chatInput.disabled = false;
      chatBtn.disabled = false;
      chatInput.placeholder = st.provider === "rules"
        ? "Try: list jobs · status job_xxx · features job_xxx"
        : "What was the best model for my last job?";
    } else {
      chatStatus.textContent = st.hint || "Chat unavailable.";
      chatStatus.classList.add("error");
    }
  } catch {
    chatStatus.textContent = "Chat service unavailable.";
    chatStatus.className = "chat-status-pill error";
  }
}

document.getElementById("chat-quick-prompts")?.addEventListener("click", (e) => {
  const chip = e.target.closest("[data-chat-prompt]");
  if (!chip) return;
  const prompt = resolveChatPrompt(chip.dataset.chatPrompt);
  chatInput.value = prompt;
  sendChatMessage(prompt);
});

chatBtn.addEventListener("click", () => sendChatMessage(chatInput.value));
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendChatMessage(chatInput.value);
  }
});

function getTargetColumn() {
  return state.targetColumn || targetSelect.value || edaTarget.value;
}

function enableMleButtons() {
  const hasDs = !!state.datasetId;
  const hasJob = !!state.jobId;
  mleQualityBtn.disabled = !hasDs;
  mleLeakageBtn.disabled = !hasDs;
  mleReviewBtn.disabled = !hasJob;
  mleBusinessBtn.disabled = !hasJob;
  mleModelcardBtn.disabled = !hasJob;
  if (hasJob) {
    compareJobA.value = compareJobA.value || state.jobId;
  }
  updateMleContext();
}

function updateMleContext() {
  const dsEl = document.getElementById("mle-context-dataset");
  const jobEl = document.getElementById("mle-context-job");
  if (dsEl) dsEl.textContent = state.datasetId || "—";
  if (jobEl) jobEl.textContent = state.jobId || "—";
}

function renderQualityScore(q) {
  const dims = Object.entries(q.dimensions || {}).map(([k, v]) =>
    `<div class="quality-dim"><strong>${escapeHtml(k.replace(/_/g, " "))}</strong><br>${v.score}/100 · ${escapeHtml(v.grade)}</div>`
  ).join("");
  const tips = (q.suggestions || []).map((s) => `<li>${escapeHtml(s)}</li>`).join("");
  return `
    <div class="mle-result-title">Quality score</div>
    <div class="quality-score-ring">${q.overall_score}/100</div>
    <p style="text-align:center;color:var(--muted);margin-bottom:0.75rem">Grade: <strong>${escapeHtml(q.grade)}</strong></p>
    <div class="quality-dim-grid">${dims}</div>
    ${tips ? `<h4>Suggestions</h4><ul>${tips}</ul>` : ""}`;
}

function renderLeakageReport(report) {
  if (!report.leakage_detected) {
    return `<div class="mle-result-title">Leakage scan</div><p class="status-success">✓ No leakage detected</p>`;
  }
  const issues = (report.issues || []).map((i) =>
    `<div class="leakage-issue"><strong>${escapeHtml(i.column)}</strong> — ${escapeHtml(i.reason)} <em>(${escapeHtml(i.type)})</em></div>`
  ).join("");
  const drops = report.recommended_drop?.length
    ? `<p style="margin-top:0.75rem"><strong>Recommended removal:</strong> ${report.recommended_drop.map((c) => `<code>${escapeHtml(c)}</code>`).join(", ")}</p>`
    : "";
  return `
    <div class="mle-result-title">Leakage scan</div>
    <p class="status-failed">⚠ Leakage detected (${report.n_issues} issue${report.n_issues === 1 ? "" : "s"})</p>
    ${issues}${drops}`;
}

function renderModelReview(review) {
  return `
    <div class="mle-review-verdict">
      <span class="tag">${escapeHtml(review.overall_verdict)}</span>
      <span class="muted">${escapeHtml(review.model_name || "")}</span>
    </div>
    <div class="mle-review-grid">
      <div class="mle-review-card">
        <h4>Strengths</h4>
        <ul>${(review.strengths || []).map((s) => `<li>${escapeHtml(s)}</li>`).join("") || "<li>None listed</li>"}</ul>
      </div>
      <div class="mle-review-card">
        <h4>Weaknesses</h4>
        <ul>${(review.weaknesses || []).map((s) => `<li>${escapeHtml(s)}</li>`).join("") || "<li>None listed</li>"}</ul>
      </div>
      <div class="mle-review-card" style="grid-column:1/-1">
        <h4>Recommendations</h4>
        <ul>${(review.recommendations || []).map((s) => `<li>${escapeHtml(s)}</li>`).join("") || "<li>None listed</li>"}</ul>
      </div>
    </div>
    ${review.narrative ? `<div class="advisor-narrative-body" style="margin-top:1rem">${renderMarkdown(review.narrative)}</div>` : ""}`;
}

async function runDatasetQuality(el, emptyEl) {
  const target = getTargetColumn();
  if (!state.datasetId || !target) { alert("Select target column first (Step 3 or 4)"); return; }
  el.innerHTML = '<span class="spinner"></span> Scoring dataset…';
  mleShowResult(el, emptyEl);
  try {
    const data = await api(`/mle/datasets/${state.datasetId}/quality?target_column=${encodeURIComponent(target)}`, {}, API_TIMEOUT.mle);
    el.innerHTML = renderQualityScore(data.quality);
    markStepDone(9);
  } catch (err) {
    el.innerHTML = `<span class="status-failed">${escapeHtml(err.message)}</span>`;
  }
}

async function runLeakageCheck(el, emptyEl) {
  const target = getTargetColumn();
  if (!state.datasetId || !target) { alert("Select target column first"); return; }
  el.innerHTML = '<span class="spinner"></span> Scanning for leakage…';
  mleShowResult(el, emptyEl);
  try {
    const data = await api(`/mle/datasets/${state.datasetId}/leakage?target_column=${encodeURIComponent(target)}`, {}, API_TIMEOUT.mle);
    el.innerHTML = renderLeakageReport(data.report);
    markStepDone(9);
  } catch (err) {
    el.innerHTML = `<span class="status-failed">${escapeHtml(err.message)}</span>`;
  }
}

function renderPipelineIntelligence(r) {
  if (!pipelineIntelligence) return;
  let html = "";
  if (r.leakage_report) {
    html += `<h3>Leakage Scan</h3>${renderLeakageReport(r.leakage_report)}`;
  }
  if (r.feature_selection?.removed_count > 0) {
    const fs = r.feature_selection;
    html += `<h3>Feature Selection</h3><p>Original: <strong>${fs.original_count}</strong> → Selected: <strong>${fs.selected_count}</strong> (removed ${fs.removed_count})</p>`;
  }
  if (r.model_review) {
    const rev = r.model_review;
    html += `<h3>AI Model Review</h3><p>Verdict: <strong>${escapeHtml(rev.overall_verdict)}</strong></p>`;
    html += `<p><strong>Strengths:</strong> ${(rev.strengths || []).slice(0, 3).map(escapeHtml).join(" · ")}</p>`;
    if (rev.weaknesses?.length) html += `<p><strong>Weaknesses:</strong> ${rev.weaknesses.slice(0, 2).map(escapeHtml).join(" · ")}</p>`;
  }
  if (r.model_card_path) {
    html += `<p><a href="/mle/jobs/${state.jobId}/model-card/download?format=md" class="link-btn" target="_blank">Download model card →</a></p>`;
  }
  if (html) {
    pipelineIntelligence.innerHTML = html;
    show(pipelineIntelligence);
  }
}

function renderFeatureDrift(data) {
  const rows = (data.feature_drift || []).map((f) => {
    const cls = f.drift_level === "high" ? "drift-high" : f.drift_level === "low" ? "drift-low" : "drift-none";
    return `<div class="drift-feature-row"><span>${escapeHtml(f.feature)}</span><span class="${cls}">${escapeHtml(f.drift_level)}</span><span>${f.score}</span></div>`;
  }).join("");
  const ds = data.dataset_drift || {};
  return `
    <p><strong>Dataset drift:</strong> ${ds.dataset_drift_detected ? "Yes" : "No"} · share ${((ds.drift_share || 0) * 100).toFixed(1)}%</p>
    <h3>Per-feature drift</h3>${rows || "<p>No features analyzed.</p>"}`;
}

counterfactualBtn?.addEventListener("click", async () => {
  if (!state.jobId) return;
  let record;
  try {
    record = JSON.parse(predictRecords.value)[0];
  } catch { alert("Enter valid JSON record first"); return; }
  counterfactualBtn.disabled = true;
  try {
    const data = await api(`/mle/jobs/${state.jobId}/counterfactual`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ record }),
    });
    const cfs = (data.result.counterfactuals || []).map((c) =>
      `<li>${escapeHtml(c.direction)} <strong>${escapeHtml(c.feature)}</strong> from ${c.original_value} → ${c.suggested_value} → prediction <strong>${escapeHtml(String(c.new_prediction))}</strong></li>`
    ).join("");
    predictResult.innerHTML += `
      <div class="advisor-card" style="margin-top:1rem">
        <h4>Counterfactual explanations</h4>
        <p>${renderMarkdown(data.result.summary || "")}</p>
        ${cfs ? `<ul>${cfs}</ul>` : ""}
      </div>`;
    show(predictResult);
  } catch (err) {
    alert(err.message);
  }
  counterfactualBtn.disabled = false;
});

function mleShowResult(resultEl, emptyEl) {
  if (resultEl) show(resultEl);
  if (emptyEl) hide(emptyEl);
}

mleQualityBtn?.addEventListener("click", () => runDatasetQuality(mleQualityResult, document.getElementById("mle-quality-empty")));
mleLeakageBtn?.addEventListener("click", () => runLeakageCheck(mleLeakageResult, document.getElementById("mle-leakage-empty")));
mleReviewBtn?.addEventListener("click", async () => {
  const emptyEl = document.getElementById("mle-review-empty");
  mleReviewResult.innerHTML = '<span class="spinner"></span> Reviewing model…';
  mleShowResult(mleReviewResult, emptyEl);
  try {
    const data = await api(`/mle/jobs/${state.jobId}/review`);
    mleReviewResult.innerHTML = renderModelReview(data.review);
    markStepDone(9);
  } catch (err) {
    mleReviewResult.innerHTML = `<span class="status-failed">${escapeHtml(err.message)}</span>`;
  }
});
mleBusinessBtn?.addEventListener("click", async () => {
  const emptyEl = document.getElementById("mle-business-empty");
  mleBusinessResult.innerHTML = '<span class="spinner"></span> Generating insights…';
  mleShowResult(mleBusinessResult, emptyEl);
  try {
    const data = await api(`/mle/jobs/${state.jobId}/business-insights`);
    const ins = data.insights;
    mleBusinessResult.innerHTML = `
      <h4>Top drivers</h4>
      <div class="mle-business-tags">${(ins.top_drivers || []).map((d) => `<span class="tag">${escapeHtml(d.feature)}</span>`).join("")}</div>
      <div class="advisor-narrative-body">${renderMarkdown(ins.executive_summary || "")}</div>`;
    markStepDone(9);
  } catch (err) {
    mleBusinessResult.innerHTML = `<span class="status-failed">${escapeHtml(err.message)}</span>`;
  }
});
mleCompareBtn?.addEventListener("click", async () => {
  const emptyEl = document.getElementById("mle-compare-empty");
  const a = compareJobA.value.trim();
  const b = compareJobB.value.trim();
  if (!a || !b) { alert("Enter both job IDs"); return; }
  mleCompareResult.innerHTML = '<span class="spinner"></span> Comparing…';
  mleShowResult(mleCompareResult, emptyEl);
  try {
    const data = await api(`/mle/experiments/compare?job_a=${encodeURIComponent(a)}&job_b=${encodeURIComponent(b)}`);
    const c = data.comparison;
    const metrics = (c.metric_diffs || []).map((d) =>
      `<tr><td>${escapeHtml(d.metric)}</td><td>${d.run_a}</td><td>${d.run_b}</td><td>${d.delta >= 0 ? "+" : ""}${d.delta}${d.pct_change != null ? ` (${d.pct_change}%)` : ""}</td></tr>`
    ).join("");
    mleCompareResult.innerHTML = `
      <div class="mle-result-title">Comparison</div>
      <p>${escapeHtml(c.summary || "")}</p>
      <div style="overflow-x:auto;margin-top:0.75rem">
        <table class="profile-table"><tr><th>Metric</th><th>Run A</th><th>Run B</th><th>Delta</th></tr>${metrics}</table>
      </div>`;
    markStepDone(9);
  } catch (err) {
    mleCompareResult.innerHTML = `<span class="status-failed">${escapeHtml(err.message)}</span>`;
  }
});
mleModelcardBtn?.addEventListener("click", async () => {
  const emptyEl = document.getElementById("mle-modelcard-empty");
  mleModelcardResult.innerHTML = '<span class="spinner"></span> Loading model card…';
  mleShowResult(mleModelcardResult, emptyEl);
  try {
    const data = await api(`/mle/jobs/${state.jobId}/model-card`);
    mleModelcardResult.innerHTML = `<div class="advisor-narrative-body">${renderMarkdown(data.markdown)}</div>`;
    mleModelcardDownload.href = `/mle/jobs/${state.jobId}/model-card/download?format=md`;
    mleModelcardDownload.classList.remove("hidden");
    markStepDone(9);
  } catch (err) {
    mleModelcardResult.innerHTML = `<span class="status-failed">${escapeHtml(err.message)}</span>`;
  }
});
activeLearningBrowse?.addEventListener("click", (e) => { e.stopPropagation(); activeLearningInput.click(); });
async function runActiveLearning(file) {
  if (!state.jobId) { alert("Train a model first"); return; }
  const emptyEl = document.getElementById("mle-active-empty");
  mleActiveResult.innerHTML = '<span class="spinner"></span> Scoring uncertainty…';
  mleShowResult(mleActiveResult, emptyEl);
  const form = new FormData();
  form.append("file", file);
  try {
    const data = await api(`/mle/jobs/${state.jobId}/active-learning`, { method: "POST", body: form }, API_TIMEOUT.mle);
    const queue = (data.result.review_queue || []).map((r) =>
      `<tr><td>${r.row_index}</td><td>${escapeHtml(String(r.prediction))}</td><td>${r.confidence ?? "—"}</td><td>${escapeHtml(r.action)}</td></tr>`
    ).join("");
    mleActiveResult.innerHTML = `
      <div class="mle-result-title">Review queue</div>
      <p><strong>${escapeHtml(data.result.summary)}</strong></p>
      <div style="overflow-x:auto;margin-top:0.75rem">
        <table class="profile-table"><tr><th>Row</th><th>Prediction</th><th>Confidence</th><th>Action</th></tr>${queue || "<tr><td colspan=4>No uncertain rows</td></tr>"}</table>
      </div>`;
    markStepDone(9);
  } catch (err) {
    mleActiveResult.innerHTML = `<span class="status-failed">${escapeHtml(err.message)}</span>`;
  }
}
setupDropZone(activeLearningZone, activeLearningInput, runActiveLearning);

checkHealth();
initChat();
renderWorkflowTracker();
loadDashboard();
updateMleContext();
setMleFeature("quality");
setStep(0);
setInterval(checkHealth, 30000);
