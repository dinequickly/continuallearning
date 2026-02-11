const state = {
  sessions: [],
  jobs: [],
  activeSessionId: null,
  currentJobId: null,
  pollToken: 0,
};

const MODEL_CATALOG = [
  { id: "Qwen/Qwen3-VL-235B-A22B-Instruct", family: "Vision", architecture: "MoE", size: "Large" },
  { id: "Qwen/Qwen3-VL-30B-A3B-Instruct", family: "Vision", architecture: "MoE", size: "Medium" },
  { id: "Qwen/Qwen3-235B-A22B-Instruct-2507", family: "Instruction", architecture: "MoE", size: "Large" },
  { id: "Qwen/Qwen3-30B-A3B-Instruct-2507", family: "Instruction", architecture: "MoE", size: "Medium" },
  { id: "Qwen/Qwen3-30B-A3B", family: "Hybrid", architecture: "MoE", size: "Medium" },
  { id: "Qwen/Qwen3-30B-A3B-Base", family: "Base", architecture: "MoE", size: "Medium" },
  { id: "Qwen/Qwen3-32B", family: "Hybrid", architecture: "Dense", size: "Medium" },
  { id: "Qwen/Qwen3-8B", family: "Hybrid", architecture: "Dense", size: "Small" },
  { id: "Qwen/Qwen3-8B-Base", family: "Base", architecture: "Dense", size: "Small" },
  { id: "Qwen/Qwen3-4B-Instruct-2507", family: "Instruction", architecture: "Dense", size: "Compact" },
  { id: "openai/gpt-oss-120b", family: "Reasoning", architecture: "MoE", size: "Medium" },
  { id: "openai/gpt-oss-20b", family: "Reasoning", architecture: "MoE", size: "Small" },
  { id: "deepseek-ai/DeepSeek-V3.1", family: "Hybrid", architecture: "MoE", size: "Large" },
  { id: "deepseek-ai/DeepSeek-V3.1-Base", family: "Base", architecture: "MoE", size: "Large" },
  { id: "meta-llama/Llama-3.1-70B", family: "Base", architecture: "Dense", size: "Large" },
  { id: "meta-llama/Llama-3.3-70B-Instruct", family: "Instruction", architecture: "Dense", size: "Large" },
  { id: "meta-llama/Llama-3.1-8B", family: "Base", architecture: "Dense", size: "Small" },
  { id: "meta-llama/Llama-3.1-8B-Instruct", family: "Instruction", architecture: "Dense", size: "Small" },
  { id: "meta-llama/Llama-3.2-3B", family: "Base", architecture: "Dense", size: "Compact" },
  { id: "meta-llama/Llama-3.2-1B", family: "Base", architecture: "Dense", size: "Compact" },
  { id: "moonshotai/Kimi-K2-Thinking", family: "Reasoning", architecture: "MoE", size: "Large" },
  { id: "moonshotai/Kimi-K2.5", family: "Reasoning", architecture: "MoE", size: "Large" },
];

function byId(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function fmtDate(value) {
  if (!value) return "-";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleString();
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function floatOrNull(value) {
  const raw = String(value ?? "").trim();
  if (!raw) return null;
  const n = Number.parseFloat(raw);
  return Number.isFinite(n) ? n : null;
}

function intOrNull(value) {
  const raw = String(value ?? "").trim();
  if (!raw) return null;
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) ? n : null;
}

function showToast(message, level = "info") {
  const toast = byId("toast");
  toast.textContent = message;
  toast.className = `toast show ${level}`.trim();
  window.clearTimeout(showToast._timer);
  showToast._timer = window.setTimeout(() => {
    toast.className = "toast";
  }, 3400);
}

async function api(path, method = "GET", body = null) {
  const res = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : null,
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const payload = await res.json();
      if (payload && payload.detail) {
        detail = payload.detail;
      } else {
        detail = JSON.stringify(payload);
      }
    } catch (_err) {
      const text = await res.text();
      detail = text || detail;
    }
    throw new Error(detail);
  }

  return await res.json();
}

function switchView(view) {
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.viewTarget === view);
  });
  document.querySelectorAll(".view").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `view-${view}`);
  });
}

function activeSession() {
  return state.sessions.find((s) => s.session_id === state.activeSessionId) || null;
}

function setupModelCatalog() {
  const list = byId("model_catalog");
  if (!list) return;
  list.innerHTML = MODEL_CATALOG.map((m) => {
    const label = `${m.id} [${m.family} | ${m.architecture} | ${m.size}]`;
    return `<option value="${escapeHtml(m.id)}" label="${escapeHtml(label)}"></option>`;
  }).join("");
  updateModelMetaHint();
}

function updateModelMetaHint() {
  const modelName = byId("session_model_name")?.value?.trim() || "";
  const meta = byId("session_model_meta");
  if (!meta) return;
  const found = MODEL_CATALOG.find((m) => m.id === modelName);
  if (!modelName) {
    meta.textContent = "Select from catalog or type any model id.";
    return;
  }
  if (!found) {
    meta.textContent = "Custom model id.";
    return;
  }
  meta.textContent = `${found.family} · ${found.architecture} · ${found.size}`;
}

function updateSessionBanner() {
  const current = activeSession();
  const banner = byId("active_session_banner");
  const chip = byId("active_session_chip");

  if (!current) {
    banner.textContent = "No active session. Create one to start prompting.";
    chip.textContent = "No active session";
    return;
  }

  banner.textContent = `Active: ${current.session_id} | requested=${current.requested_model} | resolved=${current.resolved_model} | mode=${current.mode} | messages=${current.messages}`;
  chip.textContent = current.session_id;
}

function renderSessionMeta() {
  const current = activeSession();
  byId("meta_session_id").textContent = current?.session_id ?? "-";
  byId("meta_requested_model").textContent = current?.requested_model ?? "-";
  byId("meta_resolved_model").textContent = current?.resolved_model ?? "-";
  byId("meta_mode").textContent = current?.mode ?? "-";
  byId("meta_messages").textContent = String(current?.messages ?? 0);
  byId("meta_assistant_messages").textContent = String(current?.assistant_messages ?? 0);
  byId("meta_created").textContent = current ? fmtDate(current.created_at) : "-";

  const badgeWrap = byId("meta_training_badge");
  if (!current) {
    badgeWrap.innerHTML = "";
    return;
  }
  badgeWrap.innerHTML = current.has_last_completion
    ? '<span class="badge success">Trainable Completion Available</span>'
    : '<span class="badge ghost">No Completion Buffered</span>';
}

function renderSessionSelect() {
  const select = byId("session_select");
  if (!state.sessions.length) {
    select.innerHTML = "";
    return;
  }

  select.innerHTML = state.sessions
    .map((s) => {
      const label = `${s.session_id.slice(0, 8)} | req=${s.requested_model} -> ${s.resolved_model} | ${s.mode} | m=${s.messages}`;
      return `<option value="${escapeHtml(s.session_id)}">${escapeHtml(label)}</option>`;
    })
    .join("");

  if (!state.activeSessionId || !state.sessions.some((s) => s.session_id === state.activeSessionId)) {
    state.activeSessionId = state.sessions[0].session_id;
  }
  select.value = state.activeSessionId;
}

function renderSessionsTable() {
  const tbody = byId("sessions_table_body");
  const query = (byId("session_search").value || byId("global_search").value || "").toLowerCase().trim();
  const rows = state.sessions.filter((s) => {
    if (!query) return true;
    return (
      s.session_id.toLowerCase().includes(query) ||
      s.requested_model.toLowerCase().includes(query) ||
      s.resolved_model.toLowerCase().includes(query)
    );
  });

  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="7">No sessions found.</td></tr>';
    return;
  }

  tbody.innerHTML = rows
    .map((s) => {
      const active = s.session_id === state.activeSessionId;
      return `
        <tr>
          <td class="mono">${escapeHtml(s.session_id)}</td>
          <td>${escapeHtml(s.requested_model)}</td>
          <td>${escapeHtml(s.resolved_model)}</td>
          <td><span class="badge ghost">${escapeHtml(s.mode)}</span></td>
          <td>${s.messages}</td>
          <td>${escapeHtml(fmtDate(s.created_at))}</td>
          <td>
            <div class="inline-actions" style="margin-top:0;">
              <button class="shell-btn" data-session-action="switch" data-session-id="${escapeHtml(s.session_id)}">${active ? "Active" : "Switch"}</button>
              <button class="danger-btn" data-session-action="delete" data-session-id="${escapeHtml(s.session_id)}">Delete</button>
            </div>
          </td>
        </tr>
      `;
    })
    .join("");
}

function renderSessions() {
  renderSessionSelect();
  renderSessionsTable();
  updateSessionBanner();
  renderSessionMeta();
}

async function refreshSessions({ quiet = false } = {}) {
  const data = await api("/api/sessions");
  const sessions = data.sessions || [];
  sessions.sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
  state.sessions = sessions;
  if (!state.activeSessionId || !sessions.some((s) => s.session_id === state.activeSessionId)) {
    state.activeSessionId = sessions.length ? sessions[0].session_id : null;
  }
  renderSessions();
  if (!quiet) {
    showToast("Sessions refreshed", "success");
  }
}

async function refreshActiveSessionDetails() {
  if (!state.activeSessionId) {
    renderSessionMeta();
    return;
  }
  const detail = await api(`/api/sessions/${state.activeSessionId}`);
  const idx = state.sessions.findIndex((s) => s.session_id === detail.session_id);
  if (idx >= 0) {
    state.sessions[idx] = detail;
  } else {
    state.sessions.unshift(detail);
  }
  renderSessions();
}

function sessionPayloadFromForm() {
  return {
    model_name: byId("session_model_name").value.trim() || null,
    mode: byId("session_mode").value,
    system_prompt: byId("session_system_prompt").value.trim() || null,
    temperature: floatOrNull(byId("session_temperature").value),
    max_tokens: intOrNull(byId("session_max_tokens").value),
    load_checkpoint_path: byId("session_checkpoint").value.trim() || null,
    sample_timeout_seconds: floatOrNull(byId("session_sample_timeout").value),
    train_timeout_seconds: floatOrNull(byId("session_train_timeout").value),
    enable_training: byId("session_enable_training").value === "true",
  };
}

async function createSession() {
  const payload = sessionPayloadFromForm();
  const created = await api("/api/sessions", "POST", payload);
  state.activeSessionId = created.session_id;
  await refreshSessions({ quiet: true });
  byId("active_response").textContent = "Response will appear here.";
  byId("active_metrics").textContent = "{}";
  switchView("active");
  showToast(`Session ${created.session_id.slice(0, 8)} created`, "success");
}

async function switchSelectedSession() {
  const selected = byId("session_select").value;
  if (!selected) {
    throw new Error("No session selected.");
  }
  state.activeSessionId = selected;
  await refreshActiveSessionDetails();
  switchView("active");
  showToast("Switched active session", "success");
}

async function deleteSelectedSession(sessionId = null) {
  const target = sessionId || byId("session_select").value;
  if (!target) {
    throw new Error("No session selected.");
  }
  await api(`/api/sessions/${target}`, "DELETE");
  if (state.activeSessionId === target) {
    state.activeSessionId = null;
    byId("active_response").textContent = "Response will appear here.";
    byId("active_metrics").textContent = "{}";
  }
  await refreshSessions({ quiet: true });
  showToast("Session deleted", "success");
}

function requireActiveSession() {
  if (!state.activeSessionId) {
    throw new Error("No active session. Create one first.");
  }
}

async function sendPrompt() {
  requireActiveSession();
  const prompt = byId("active_prompt").value.trim();
  if (!prompt) {
    throw new Error("Prompt text is required.");
  }

  const btn = byId("send_prompt_btn");
  btn.disabled = true;
  btn.textContent = "Generating...";
  try {
    const data = await api(`/api/sessions/${state.activeSessionId}/prompt`, "POST", { text: prompt });
    byId("active_response").textContent = data.response || "";
    byId("active_metrics").textContent = "{}";
    await refreshSessions({ quiet: true });
    await refreshActiveSessionDetails();
  } finally {
    btn.disabled = false;
    btn.textContent = "Generate";
  }
}

async function sendFeedback(kind) {
  requireActiveSession();
  const text = byId("active_feedback").value.trim();
  if ((kind === "revise" || kind === "ideal") && !text) {
    throw new Error("Feedback text is required for revise/ideal.");
  }

  const data = await api(`/api/sessions/${state.activeSessionId}/feedback`, "POST", {
    kind,
    text: text || null,
  });

  if (data.response) {
    byId("active_response").textContent = data.response;
  }
  byId("active_metrics").textContent = JSON.stringify(data.metrics || {}, null, 2);
  await refreshSessions({ quiet: true });
  await refreshActiveSessionDetails();
}

async function clearActiveSession() {
  requireActiveSession();
  await api(`/api/sessions/${state.activeSessionId}/clear`, "POST");
  byId("active_response").textContent = "Response will appear here.";
  byId("active_metrics").textContent = "{}";
  await refreshSessions({ quiet: true });
  await refreshActiveSessionDetails();
  showToast("Session cleared", "success");
}

function jobStatusBadge(status) {
  const normalized = String(status || "unknown").toLowerCase();
  if (normalized === "succeeded") return '<span class="badge success job-pill">succeeded</span>';
  if (normalized === "failed") return '<span class="badge error job-pill">failed</span>';
  if (normalized === "running") return '<span class="badge warn job-pill">running</span>';
  return `<span class="badge ghost job-pill">${escapeHtml(normalized)}</span>`;
}

function renderJobsTable() {
  const body = byId("jobs_table_body");
  if (!state.jobs.length) {
    body.innerHTML = '<tr><td colspan="5">No jobs yet.</td></tr>';
    return;
  }

  body.innerHTML = state.jobs
    .map((job) => `
      <tr>
        <td class="mono">${escapeHtml(job.id)}</td>
        <td>${escapeHtml(job.kind)}</td>
        <td>${jobStatusBadge(job.status)}</td>
        <td>${escapeHtml(fmtDate(job.created_at))}</td>
        <td><button class="shell-btn" data-job-action="view" data-job-id="${escapeHtml(job.id)}">Inspect</button></td>
      </tr>
    `)
    .join("");
}

async function refreshJobs({ quiet = true } = {}) {
  const data = await api("/api/jobs");
  const jobs = data.jobs || [];
  jobs.sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
  state.jobs = jobs;
  renderJobsTable();
  if (!quiet) {
    showToast("Jobs refreshed", "success");
  }
}

async function loadJobDetail(jobId) {
  const data = await api(`/api/jobs/${jobId}`);
  state.currentJobId = jobId;
  byId("job_detail").textContent = JSON.stringify(data, null, 2);
  return data;
}

async function pollJob(jobId, outputElId) {
  const myToken = ++state.pollToken;
  byId(outputElId).textContent = `Job ${jobId} started. Polling...`;

  while (myToken === state.pollToken) {
    const detail = await loadJobDetail(jobId);
    byId(outputElId).textContent = JSON.stringify(detail, null, 2);
    await refreshJobs({ quiet: true });

    if (detail.status === "succeeded" || detail.status === "failed") {
      const level = detail.status === "succeeded" ? "success" : "error";
      showToast(`Job ${jobId.slice(0, 8)} ${detail.status}`, level);
      return;
    }

    await sleep(2200);
  }
}

async function startSftJob() {
  const payload = {
    model_name: byId("session_model_name").value.trim() || null,
    examples_path: byId("sft_examples_path").value.trim(),
    log_path: byId("sft_log_path").value.trim() || null,
    num_epochs: Number(byId("sft_epochs").value || "1"),
    learning_rate: Number(byId("sft_lr").value || "0.0001"),
  };

  if (!payload.examples_path) {
    throw new Error("SFT examples_path is required.");
  }

  const data = await api("/api/jobs/sft", "POST", payload);
  switchView("training");
  await pollJob(data.job_id, "sft_job_out");
}

async function startDpoJob() {
  const payload = {
    model_name: byId("session_model_name").value.trim() || null,
    preferences_path: byId("dpo_pref_path").value.trim(),
    log_path: byId("dpo_log_path").value.trim() || null,
    dpo_beta: Number(byId("dpo_beta").value || "0.1"),
    learning_rate: Number(byId("dpo_lr").value || "0.00001"),
  };

  if (!payload.preferences_path) {
    throw new Error("DPO preferences_path is required.");
  }

  const data = await api("/api/jobs/dpo", "POST", payload);
  switchView("training");
  await pollJob(data.job_id, "dpo_job_out");
}

function attachHandlers() {
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchView(btn.dataset.viewTarget));
  });

  byId("quick_create_btn").addEventListener("click", () => {
    switchView("sessions");
    createSession().catch((err) => showToast(err.message, "error"));
  });

  byId("session_create_btn").addEventListener("click", () => {
    createSession().catch((err) => showToast(err.message, "error"));
  });

  byId("sessions_refresh_btn").addEventListener("click", () => {
    refreshSessions({ quiet: false }).catch((err) => showToast(err.message, "error"));
  });

  byId("sessions_switch_btn").addEventListener("click", () => {
    switchSelectedSession().catch((err) => showToast(err.message, "error"));
  });

  byId("sessions_delete_btn").addEventListener("click", () => {
    deleteSelectedSession().catch((err) => showToast(err.message, "error"));
  });

  byId("session_select").addEventListener("change", () => {
    state.activeSessionId = byId("session_select").value || null;
    renderSessions();
  });

  byId("session_model_name").addEventListener("input", updateModelMetaHint);
  byId("session_model_name").addEventListener("change", updateModelMetaHint);

  byId("session_search").addEventListener("input", renderSessionsTable);
  byId("global_search").addEventListener("input", renderSessionsTable);

  byId("sessions_table_body").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-session-action]");
    if (!button) return;
    const id = button.dataset.sessionId;
    if (!id) return;

    if (button.dataset.sessionAction === "switch") {
      state.activeSessionId = id;
      refreshActiveSessionDetails().then(() => {
        switchView("active");
      }).catch((err) => showToast(err.message, "error"));
      return;
    }

    if (button.dataset.sessionAction === "delete") {
      deleteSelectedSession(id).catch((err) => showToast(err.message, "error"));
    }
  });

  byId("send_prompt_btn").addEventListener("click", () => {
    sendPrompt().catch((err) => showToast(err.message, "error"));
  });

  byId("feedback_accept_btn").addEventListener("click", () => {
    sendFeedback("accept").then(() => {
      showToast("Response accepted", "success");
    }).catch((err) => showToast(err.message, "error"));
  });

  byId("feedback_revise_btn").addEventListener("click", () => {
    sendFeedback("revise").catch((err) => showToast(err.message, "error"));
  });

  byId("feedback_ideal_btn").addEventListener("click", () => {
    sendFeedback("ideal").catch((err) => showToast(err.message, "error"));
  });

  byId("active_clear_btn").addEventListener("click", () => {
    clearActiveSession().catch((err) => showToast(err.message, "error"));
  });

  byId("jobs_refresh_btn").addEventListener("click", () => {
    refreshJobs({ quiet: false }).catch((err) => showToast(err.message, "error"));
  });

  byId("run_sft_btn").addEventListener("click", () => {
    startSftJob().catch((err) => showToast(err.message, "error"));
  });

  byId("run_dpo_btn").addEventListener("click", () => {
    startDpoJob().catch((err) => showToast(err.message, "error"));
  });

  byId("jobs_table_body").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-job-action]");
    if (!button) return;
    const id = button.dataset.jobId;
    if (!id) return;
    loadJobDetail(id).catch((err) => showToast(err.message, "error"));
  });
}

async function init() {
  setupModelCatalog();
  attachHandlers();
  switchView("sessions");

  try {
    await refreshSessions({ quiet: true });
    await refreshJobs({ quiet: true });
    showToast("Frontend ready", "success");
  } catch (err) {
    showToast(err.message || String(err), "error");
  }
}

init();
