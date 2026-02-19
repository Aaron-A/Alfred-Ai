/* ═══════════════════════════════════════
   ALFRED AI - Dashboard JS
   Vanilla JS, no frameworks.
   ═══════════════════════════════════════ */

const API = '';  // Same origin
const REFRESH_INTERVAL = 10000;  // 10 seconds

// ─── State ────────────────────────────────────────

let state = {
  status: null,
  agents: [],
  metrics: null,
  schedules: {},  // agent_name -> [schedule, ...]
  providerModels: {},  // provider -> [model, ...]
};

// ─── API Helpers ──────────────────────────────────

async function fetchJSON(path) {
  try {
    const res = await fetch(API + path);
    if (!res.ok) return null;
    return await res.json();
  } catch (e) {
    console.warn('Fetch failed:', path, e);
    return null;
  }
}

async function patchJSON(path, body) {
  try {
    const res = await fetch(API + path, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return await res.json();
  } catch (e) {
    console.error('PATCH failed:', path, e);
    return null;
  }
}

async function postJSON(path, body = {}) {
  try {
    const res = await fetch(API + path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) {
      // Attach HTTP status so callers can inspect errors
      data._status = res.status;
      data._error = data.detail || data.message || 'Request failed';
    }
    return data;
  } catch (e) {
    console.error('POST failed:', path, e);
    return null;
  }
}

async function deleteJSON(path) {
  try {
    const res = await fetch(API + path, { method: 'DELETE' });
    const data = await res.json();
    if (!res.ok) {
      data._status = res.status;
      data._error = data.detail || data.message || 'Request failed';
    }
    return data;
  } catch (e) {
    console.error('DELETE failed:', path, e);
    return null;
  }
}

// ─── Data Fetching ────────────────────────────────

async function refreshData() {
  // Read selected period from the Metrics tab dropdown
  const period = document.getElementById('metricsPeriod')?.value || 'session';
  const metricsUrl = '/v1/metrics' + (period !== 'session' ? '?period=' + period : '');

  const [status, agentsData, metricsData, providersData] = await Promise.all([
    fetchJSON('/v1/status'),
    fetchJSON('/v1/agents'),
    fetchJSON(metricsUrl),
    Object.keys(state.providerModels).length ? null : fetchJSON('/v1/providers'),
  ]);

  state.status = status;
  state.agents = agentsData?.agents || [];
  state.metrics = metricsData;

  // Only set providers once (they don't change at runtime)
  if (providersData?.providers) {
    state.providerModels = providersData.providers;
  }

  // Fetch schedules for each agent
  const schedPromises = state.agents.map(a =>
    fetchJSON(`/v1/agents/${a.name}/schedules`).then(d => [a.name, d])
  );
  const schedResults = await Promise.all(schedPromises);
  state.schedules = {};
  for (const [name, data] of schedResults) {
    if (data?.schedules?.length) {
      state.schedules[name] = data.schedules;
    }
  }

  // Also fetch session counts per agent
  const sessPromises = state.agents.map(a =>
    fetchJSON(`/v1/sessions/${a.name}`).then(d => [a.name, d])
  );
  const sessResults = await Promise.all(sessPromises);
  state.sessions = {};
  for (const [name, data] of sessResults) {
    state.sessions[name] = data?.sessions || [];
  }

  renderAll();
  updateRefreshInfo();
}

// ─── Formatters ───────────────────────────────────

function fmtNum(n) {
  if (n == null || isNaN(n)) return '0';
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
  return n.toLocaleString();
}

function fmtTokens(n) {
  if (n == null || isNaN(n)) return '0';
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
  return n.toString();
}

function fmtMs(ms) {
  if (!ms) return '-';
  if (ms >= 1000) return (ms / 1000).toFixed(1) + 's';
  return ms + 'ms';
}

function fmtTime(iso) {
  if (!iso) return '-';
  return iso.slice(0, 16).replace('T', ' ');
}

function fmtRelative(iso) {
  if (!iso) return '-';
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return mins + 'm ago';
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return hrs + 'h ago';
  const days = Math.floor(hrs / 24);
  return days + 'd ago';
}

function escHtml(s) {
  if (!s) return '';
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function statusBadge(status) {
  if (status === 'active') return '<span class="badge badge--active">active</span>';
  if (status === 'paused') return '<span class="badge badge--paused">paused</span>';
  return '<span class="badge badge--info">' + escHtml(status) + '</span>';
}

// ─── Aggregate Metrics ────────────────────────────

function aggregateMetrics() {
  const agents = state.metrics?.agents || {};
  let totalMessages = 0, totalTokens = 0, totalMs = 0, totalErrors = 0;
  let agentCount = 0;

  for (const [name, m] of Object.entries(agents)) {
    totalMessages += m.messages || 0;
    totalTokens += (m.input_tokens || 0) + (m.output_tokens || 0);
    totalMs += m.total_elapsed_ms || 0;
    totalErrors += m.errors || 0;
    if (m.messages > 0) agentCount++;
  }

  const avgMs = totalMessages > 0 ? Math.round(totalMs / totalMessages) : 0;

  return { totalMessages, totalTokens, totalMs, totalErrors, avgMs, agentCount };
}

// ─── Render: Stats Grid ───────────────────────────

function renderStats() {
  const agg = aggregateMetrics();
  const status = state.status;
  const agents = state.agents;

  // Agents
  const active = agents.filter(a => a.status !== 'paused').length;
  const total = agents.length;
  document.getElementById('statAgents').textContent = total;
  document.getElementById('statAgentsSub').textContent = `${active} active` + (total - active > 0 ? `, ${total - active} paused` : '');

  // Messages
  document.getElementById('statMessages').textContent = fmtNum(agg.totalMessages);
  document.getElementById('statMessagesSub').textContent = agg.totalErrors > 0
    ? `${agg.totalErrors} error${agg.totalErrors !== 1 ? 's' : ''}`
    : 'total processed';

  // Tokens
  document.getElementById('statTokens').textContent = fmtTokens(agg.totalTokens);
  const metricsAgents = state.metrics?.agents || {};
  let totalIn = 0, totalOut = 0;
  for (const m of Object.values(metricsAgents)) {
    totalIn += m.input_tokens || 0;
    totalOut += m.output_tokens || 0;
  }
  document.getElementById('statTokensSub').textContent = `${fmtTokens(totalIn)} in / ${fmtTokens(totalOut)} out`;

  // Avg Response
  document.getElementById('statAvgMs').textContent = fmtMs(agg.avgMs);
  document.getElementById('statAvgMsSub').textContent = agg.totalMessages > 0
    ? `across ${agg.totalMessages} messages`
    : 'no messages yet';

  // Discord
  const discord = status?.discord;
  const discordEl = document.getElementById('statDiscord');
  const discordSub = document.getElementById('statDiscordSub');

  if (!discord?.configured) {
    discordEl.textContent = 'N/A';
    discordEl.className = 'stat-value';
    discordSub.textContent = 'not configured';
  } else if (discord.healthy) {
    discordEl.textContent = 'Online';
    discordEl.className = 'stat-value stat-value--green';
    discordSub.textContent = discord.health_message || `${discord.channels} channel${discord.channels !== 1 ? 's' : ''}`;
  } else if (discord.running) {
    discordEl.textContent = 'Warning';
    discordEl.className = 'stat-value stat-value--orange';
    discordSub.textContent = discord.health_message || 'unresponsive';
  } else {
    discordEl.textContent = 'Offline';
    discordEl.className = 'stat-value stat-value--red';
    discordSub.textContent = discord.health_message || 'not running';
  }

  // Header status dot
  const dot = document.getElementById('statusDot');
  const statusText = document.getElementById('statusText');
  if (status) {
    dot.className = 'status-dot';
    statusText.textContent = 'connected';
  } else {
    dot.className = 'status-dot status-dot--offline';
    statusText.textContent = 'disconnected';
  }
}

// ─── Render: Agents Table ─────────────────────────

function renderAgentsTable() {
  const tbody = document.getElementById('agentsBody');
  const agents = state.agents;
  const metricsAgents = state.metrics?.agents || {};

  if (!agents.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No agents configured. Run: alfred setup</td></tr>';
    return;
  }

  let html = '';
  for (const agent of agents) {
    const m = metricsAgents[agent.name] || {};
    const sessions = state.sessions?.[agent.name] || [];
    const totalTokens = (m.input_tokens || 0) + (m.output_tokens || 0);
    const provider = agent.provider || state.status?.primary_llm?.provider || '?';
    const model = agent.model || state.status?.primary_llm?.model || '?';
    const modelShort = model.length > 25 ? model.slice(0, 22) + '...' : model;

    // Clean text display with edit pencil + delete button
    const editBtn = Object.keys(state.providerModels).length
      ? `<button class="edit-btn" onclick="openConfigModal('${escHtml(agent.name)}')" title="Edit provider/model">&#9998;</button>`
      : '';
    const deleteBtn = state.agents.length > 1
      ? `<button class="delete-btn" onclick="openDeleteAgentModal('${escHtml(agent.name)}')" title="Delete agent">&#10005;</button>`
      : '';

    html += `<tr>
      <td class="fw-600" style="color:#fff;">${escHtml(agent.name)}</td>
      <td>${statusBadge(agent.status || 'active')}</td>
      <td><span class="text-dim">${escHtml(provider)}/${escHtml(modelShort)}</span> ${editBtn}${deleteBtn}</td>
      <td>${sessions.length || 0}</td>
      <td>${fmtNum(m.messages || 0)}</td>
      <td>${fmtTokens(totalTokens)}</td>
      <td>${m.errors ? '<span class="text-red">' + m.errors + '</span>' : '<span class="text-dim">0</span>'}</td>
    </tr>`;
  }

  tbody.innerHTML = html;
}

// ─── Config Modal ─────────────────────────────────

function buildProviderOptions(currentProvider) {
  const providers = Object.keys(state.providerModels);
  return providers.map(p =>
    `<option value="${escHtml(p)}"${p === currentProvider ? ' selected' : ''}>${escHtml(p)}</option>`
  ).join('');
}

function buildModelOptions(provider, currentModel) {
  const models = state.providerModels[provider] || [];
  return models.map(m =>
    `<option value="${escHtml(m)}"${m === currentModel ? ' selected' : ''}>${escHtml(m)}</option>`
  ).join('');
}

function openConfigModal(agentName) {
  const agent = state.agents.find(a => a.name === agentName);
  if (!agent) return;

  const modal = document.getElementById('configModal');
  const provider = agent.provider || state.status?.primary_llm?.provider || 'anthropic';
  const model = agent.model || state.status?.primary_llm?.model || '';

  // Set agent name
  document.getElementById('modalAgentName').textContent = agentName;
  modal.dataset.agent = agentName;

  // Populate provider dropdown
  const providerSelect = document.getElementById('modalProvider');
  providerSelect.innerHTML = buildProviderOptions(provider);

  // Populate model dropdown
  const modelSelect = document.getElementById('modalModel');
  modelSelect.innerHTML = buildModelOptions(provider, model);

  // Current display
  document.getElementById('modalCurrent').textContent = `${provider} / ${model}`;

  // Reset button states
  const saveBtn = document.getElementById('modalSaveBtn');
  saveBtn.textContent = 'SAVE';
  saveBtn.disabled = true;
  saveBtn.className = 'modal-btn modal-btn--save';

  // Show modal
  modal.classList.add('active');
}

function closeConfigModal() {
  document.getElementById('configModal').classList.remove('active');
}

function onModalProviderChange() {
  const providerSelect = document.getElementById('modalProvider');
  const modelSelect = document.getElementById('modalModel');
  const newProvider = providerSelect.value;

  // Repopulate model dropdown with first model selected
  const models = state.providerModels[newProvider] || [];
  modelSelect.innerHTML = buildModelOptions(newProvider, models[0] || '');

  checkModalDirty();
}

function onModalModelChange() {
  checkModalDirty();
}

function checkModalDirty() {
  const modal = document.getElementById('configModal');
  const agentName = modal.dataset.agent;
  const agent = state.agents.find(a => a.name === agentName);
  if (!agent) return;

  const origProvider = agent.provider || state.status?.primary_llm?.provider || 'anthropic';
  const origModel = agent.model || state.status?.primary_llm?.model || '';
  const newProvider = document.getElementById('modalProvider').value;
  const newModel = document.getElementById('modalModel').value;

  const isDirty = (newProvider !== origProvider || newModel !== origModel);
  const saveBtn = document.getElementById('modalSaveBtn');
  saveBtn.disabled = !isDirty;
  saveBtn.className = isDirty ? 'modal-btn modal-btn--save modal-btn--active' : 'modal-btn modal-btn--save';
}

async function saveModalConfig() {
  const modal = document.getElementById('configModal');
  const agentName = modal.dataset.agent;
  const provider = document.getElementById('modalProvider').value;
  const model = document.getElementById('modalModel').value;

  const saveBtn = document.getElementById('modalSaveBtn');
  saveBtn.disabled = true;
  saveBtn.textContent = 'SAVING...';

  const result = await patchJSON(`/v1/agents/${agentName}/config`, { provider, model });

  if (result?.message) {
    // Update local state
    const agent = state.agents.find(a => a.name === agentName);
    if (agent) {
      agent.provider = provider;
      agent.model = model;
    }

    saveBtn.textContent = 'SAVED';
    saveBtn.className = 'modal-btn modal-btn--save modal-btn--saved';

    // Refresh the table immediately
    renderAgentsTable();

    // Close modal after brief success flash, then show restart banner
    setTimeout(() => {
      closeConfigModal();
      showRestartBanner();
    }, 800);
  } else {
    saveBtn.textContent = 'FAILED';
    saveBtn.className = 'modal-btn modal-btn--save modal-btn--error';
    setTimeout(() => {
      saveBtn.textContent = 'SAVE';
      saveBtn.disabled = false;
      checkModalDirty();
    }, 2000);
  }
}

// ─── Restart Banner ───────────────────────────────

function showRestartBanner() {
  let banner = document.getElementById('restartBanner');
  if (!banner) {
    banner = document.createElement('div');
    banner.id = 'restartBanner';
    banner.className = 'restart-banner';
    banner.innerHTML = `
      <span class="restart-banner-text">Config saved. Restart daemon to apply changes.</span>
      <button class="modal-btn modal-btn--restart" onclick="restartDaemon()">RESTART DAEMON</button>
      <button class="modal-btn modal-btn--dismiss" onclick="dismissBanner()">DISMISS</button>
    `;
    const header = document.querySelector('.header');
    header.parentNode.insertBefore(banner, header.nextSibling);
  }
  banner.style.display = 'flex';
}

function dismissBanner() {
  const banner = document.getElementById('restartBanner');
  if (banner) banner.style.display = 'none';
}

async function restartDaemon() {
  const banner = document.getElementById('restartBanner');
  const btn = banner?.querySelector('.modal-btn--restart');
  const textEl = banner?.querySelector('.restart-banner-text');
  if (btn) {
    btn.disabled = true;
    btn.textContent = 'RESTARTING...';
  }

  const result = await postJSON('/v1/admin/reload');

  if (result?.status === 'restarting' || result?.status === 'restarted') {
    if (textEl) textEl.textContent = 'Daemon restarting — waiting for API...';

    // Poll until the API comes back (old process dies, new one starts)
    let attempts = 0;
    const maxAttempts = 20;
    const pollInterval = 1500;

    const poll = setInterval(async () => {
      attempts++;
      try {
        const res = await fetch(API + '/health');
        if (res.ok) {
          clearInterval(poll);
          if (btn) {
            btn.textContent = 'RESTARTED';
            btn.className = 'modal-btn modal-btn--restart modal-btn--saved';
          }
          if (textEl) textEl.textContent = 'Daemon restarted successfully.';
          setTimeout(() => {
            dismissBanner();
            refreshData();
          }, 1500);
        }
      } catch (e) {
        // API not up yet — keep polling
        if (textEl) textEl.textContent = `Restarting... (${attempts}/${maxAttempts})`;
      }
      if (attempts >= maxAttempts) {
        clearInterval(poll);
        if (btn) {
          btn.textContent = 'TIMEOUT';
          btn.className = 'modal-btn modal-btn--restart modal-btn--error';
          setTimeout(() => {
            btn.textContent = 'RESTART DAEMON';
            btn.disabled = false;
            btn.className = 'modal-btn modal-btn--restart';
          }, 3000);
        }
        if (textEl) textEl.textContent = 'Restart timed out. Try: alfred stop && alfred start';
      }
    }, pollInterval);
  } else {
    const msg = result?.message || 'Restart failed';
    if (btn) {
      btn.textContent = 'FAILED';
      btn.className = 'modal-btn modal-btn--restart modal-btn--error';
      setTimeout(() => {
        btn.textContent = 'RESTART DAEMON';
        btn.disabled = false;
        btn.className = 'modal-btn modal-btn--restart';
      }, 3000);
    }
    if (textEl) textEl.textContent = msg;
  }
}

// ─── Create Agent Modal ──────────────────────────

function openCreateAgentModal() {
  const modal = document.getElementById('createAgentModal');

  // Reset all fields
  document.getElementById('newAgentName').value = '';
  document.getElementById('newAgentDesc').value = '';
  document.getElementById('newAgentDiscordChannel').value = '';
  document.getElementById('newAgentSoul').value = '';

  // Reset hints
  const nameHint = document.getElementById('nameHint');
  nameHint.textContent = 'Lowercase, a-z, 0-9, hyphens only';
  nameHint.className = 'modal-hint';

  // Populate provider dropdown
  const providerSelect = document.getElementById('newAgentProvider');
  const providers = Object.keys(state.providerModels);
  if (providers.length) {
    providerSelect.innerHTML = providers.map(p =>
      `<option value="${escHtml(p)}">${escHtml(p)}</option>`
    ).join('');
    onNewAgentProviderChange();
  }

  // Reset create button
  const createBtn = document.getElementById('createAgentBtn');
  createBtn.textContent = 'CREATE';
  createBtn.disabled = true;
  createBtn.className = 'modal-btn modal-btn--save';

  // Remove error styling from inputs
  document.getElementById('newAgentName').classList.remove('modal-input--error');

  modal.classList.add('active');

  // Focus name field
  setTimeout(() => document.getElementById('newAgentName').focus(), 100);
}

function closeCreateAgentModal() {
  document.getElementById('createAgentModal').classList.remove('active');
}

function onNewAgentProviderChange() {
  const provider = document.getElementById('newAgentProvider').value;
  const modelSelect = document.getElementById('newAgentModel');
  const models = state.providerModels[provider] || [];
  modelSelect.innerHTML = models.map(m =>
    `<option value="${escHtml(m)}">${escHtml(m)}</option>`
  ).join('');
}

function validateCreateForm() {
  const nameInput = document.getElementById('newAgentName');
  const descInput = document.getElementById('newAgentDesc');
  const nameHint = document.getElementById('nameHint');
  const createBtn = document.getElementById('createAgentBtn');
  const name = nameInput.value.trim();
  const desc = descInput.value.trim();

  let valid = true;

  // Validate name format
  if (!name) {
    nameHint.textContent = 'Lowercase, a-z, 0-9, hyphens only';
    nameHint.className = 'modal-hint';
    nameInput.classList.remove('modal-input--error');
    valid = false;
  } else if (!/^[a-z][a-z0-9-]{0,29}$/.test(name)) {
    nameHint.textContent = 'Must start with a letter, only a-z, 0-9, hyphens (max 30)';
    nameHint.className = 'modal-hint modal-hint--error';
    nameInput.classList.add('modal-input--error');
    valid = false;
  } else {
    // Check if name already exists
    const exists = state.agents.some(a => a.name === name);
    if (exists) {
      nameHint.textContent = `Agent "${name}" already exists`;
      nameHint.className = 'modal-hint modal-hint--error';
      nameInput.classList.add('modal-input--error');
      valid = false;
    } else {
      nameHint.textContent = 'Looks good';
      nameHint.className = 'modal-hint';
      nameInput.classList.remove('modal-input--error');
    }
  }

  // Description required
  if (!desc) valid = false;

  // Update button state
  createBtn.disabled = !valid;
  createBtn.className = valid
    ? 'modal-btn modal-btn--save modal-btn--active'
    : 'modal-btn modal-btn--save';
}

async function createAgent() {
  const createBtn = document.getElementById('createAgentBtn');
  const name = document.getElementById('newAgentName').value.trim();
  const description = document.getElementById('newAgentDesc').value.trim();
  const provider = document.getElementById('newAgentProvider').value;
  const model = document.getElementById('newAgentModel').value;
  const soulPrompt = document.getElementById('newAgentSoul').value.trim();

  // Disable button and show spinner
  createBtn.disabled = true;
  createBtn.innerHTML = `<span class="btn-spinner"></span>${soulPrompt ? 'GENERATING...' : 'CREATING...'}`;
  createBtn.className = 'modal-btn modal-btn--save';

  const discordChannelId = document.getElementById('newAgentDiscordChannel').value.trim();

  const body = { name, description, provider, model };
  if (soulPrompt) body.soul_prompt = soulPrompt;
  if (discordChannelId) body.discord_channel_id = discordChannelId;

  const result = await postJSON('/v1/agents', body);

  if (result && !result._error) {
    createBtn.textContent = 'CREATED';
    createBtn.className = 'modal-btn modal-btn--save modal-btn--saved';

    // Refresh data and close
    await refreshData();
    setTimeout(() => {
      closeCreateAgentModal();
      showRestartBanner();
    }, 800);
  } else {
    const errMsg = result?._error || 'Creation failed';
    createBtn.textContent = 'FAILED';
    createBtn.className = 'modal-btn modal-btn--save modal-btn--error';

    // Show error in name hint
    const nameHint = document.getElementById('nameHint');
    nameHint.textContent = errMsg;
    nameHint.className = 'modal-hint modal-hint--error';

    setTimeout(() => {
      createBtn.textContent = 'CREATE';
      createBtn.className = 'modal-btn modal-btn--save';
      validateCreateForm();
    }, 3000);
  }
}

// ─── Delete Agent Modal ──────────────────────────

let deleteTargetAgent = '';

function openDeleteAgentModal(name) {
  deleteTargetAgent = name;
  const modal = document.getElementById('deleteAgentModal');
  document.getElementById('deleteAgentTitle').textContent = `Delete "${name}"`;
  document.getElementById('deleteConfirmInput').value = '';
  document.getElementById('deleteConfirmHint').textContent = `Type "${name}" to confirm`;
  document.getElementById('deleteConfirmHint').className = 'modal-hint';

  const deleteBtn = document.getElementById('deleteAgentBtn');
  deleteBtn.disabled = true;
  deleteBtn.textContent = 'DELETE';
  deleteBtn.className = 'modal-btn modal-btn--danger';

  modal.classList.add('active');
  setTimeout(() => document.getElementById('deleteConfirmInput').focus(), 100);
}

function closeDeleteAgentModal() {
  document.getElementById('deleteAgentModal').classList.remove('active');
  deleteTargetAgent = '';
}

function validateDeleteConfirm() {
  const typed = document.getElementById('deleteConfirmInput').value;
  const matches = typed === deleteTargetAgent;
  const deleteBtn = document.getElementById('deleteAgentBtn');
  deleteBtn.disabled = !matches;
  deleteBtn.className = matches
    ? 'modal-btn modal-btn--danger modal-btn--danger-active'
    : 'modal-btn modal-btn--danger';
}

async function deleteAgent() {
  const deleteBtn = document.getElementById('deleteAgentBtn');
  deleteBtn.disabled = true;
  deleteBtn.innerHTML = '<span class="btn-spinner"></span>DELETING...';

  const result = await deleteJSON(`/v1/agents/${deleteTargetAgent}`);

  if (result && !result._error) {
    deleteBtn.textContent = 'DELETED';
    deleteBtn.className = 'modal-btn modal-btn--danger modal-btn--saved';
    await refreshData();
    setTimeout(() => {
      closeDeleteAgentModal();
      showRestartBanner();
    }, 800);
  } else {
    const errMsg = result?._error || 'Deletion failed';
    deleteBtn.textContent = 'FAILED';
    deleteBtn.className = 'modal-btn modal-btn--danger modal-btn--error';

    const hint = document.getElementById('deleteConfirmHint');
    hint.textContent = errMsg;
    hint.className = 'modal-hint modal-hint--error';

    setTimeout(() => {
      deleteBtn.textContent = 'DELETE';
      deleteBtn.className = 'modal-btn modal-btn--danger';
      validateDeleteConfirm();
    }, 3000);
  }
}

// ─── Render: Schedules Table ──────────────────────

function renderSchedulesTable() {
  const tbody = document.getElementById('schedulesBody');
  const schedules = state.schedules;

  const allSchedules = [];
  for (const [agent, scheds] of Object.entries(schedules)) {
    for (const s of scheds) {
      allSchedules.push({ ...s, agent_name: agent });
    }
  }

  if (!allSchedules.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="empty-state">No schedules configured. Run: alfred agent schedule add &lt;name&gt;</td></tr>';
    return;
  }

  let html = '';
  for (const s of allSchedules) {
    const task = s.task?.length > 35 ? s.task.slice(0, 32) + '...' : (s.task || '?');

    // Status
    let statusHtml = s.enabled
      ? '<span class="badge badge--active">active</span>'
      : '<span class="badge badge--paused">paused</span>';
    if (s.consecutive_failures >= 3) {
      statusHtml += ' <span class="text-red" style="font-size:10px;">(' + s.consecutive_failures + ' fails)</span>';
    }

    // Runs
    let runsHtml = '<span class="text-dim">0</span>';
    if (s.run_count > 0) {
      const rate = s.success_count / s.run_count * 100;
      const rateColor = rate >= 90 ? 'text-green' : rate >= 50 ? 'text-yellow' : 'text-red';
      runsHtml = `${s.run_count} <span class="${rateColor}">(${rate.toFixed(0)}%)</span>`;
    }

    // Last run
    const lastRun = s.last_run ? fmtRelative(s.last_run) : '<span class="text-dim">never</span>';

    // Next run
    const nextRun = s.next_run ? fmtRelative_future(s.next_run) : '<span class="text-dim">\u2014</span>';

    // Run Now button
    const runId = `run-btn-${s.agent_name}-${s.id}`;
    const isRunning = s._running;
    let runBtn;
    if (!s.enabled) {
      runBtn = '<button class="run-btn" disabled title="Schedule is paused">&#9654;</button>';
    } else if (isRunning) {
      runBtn = `<button class="run-btn run-btn--running" id="${runId}" disabled><span class="run-spinner"></span></button>`;
    } else {
      runBtn = `<button class="run-btn" id="${runId}" onclick="runScheduleNow('${escHtml(s.agent_name)}','${escHtml(s.id)}')" title="Run now">&#9654;</button>`;
    }

    html += `<tr>
      <td class="fw-600" style="color:#fff;">${escHtml(s.agent_name)}</td>
      <td style="color:var(--cyan);">${escHtml(s.id)}</td>
      <td>${escHtml(s.human_schedule || s.cron)}</td>
      <td>${escHtml(task)}</td>
      <td>${statusHtml}</td>
      <td>${runsHtml}</td>
      <td>${lastRun}</td>
      <td>${nextRun}</td>
      <td>${runBtn}</td>
    </tr>`;
  }

  tbody.innerHTML = html;
}

function fmtRelative_future(iso) {
  if (!iso) return '-';
  const diff = new Date(iso).getTime() - Date.now();
  if (diff < 0) return 'now';
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return `<span class="text-green">${mins}m</span>`;
  const hrs = Math.floor(mins / 60);
  const remainMins = mins % 60;
  if (hrs < 24) return `<span class="text-green">${hrs}h ${remainMins}m</span>`;
  const days = Math.floor(hrs / 24);
  return `<span class="text-green">${days}d ${hrs % 24}h</span>`;
}

// ─── Schedule: Run Now ───────────────────────────

async function runScheduleNow(agentName, scheduleId) {
  const btnId = `run-btn-${agentName}-${scheduleId}`;
  const btn = document.getElementById(btnId);
  if (!btn || btn.disabled) return;

  // Show spinner
  btn.disabled = true;
  btn.innerHTML = '<span class="run-spinner"></span>';
  btn.classList.add('run-btn--running');
  _markScheduleRunning(agentName, scheduleId, true);

  const result = await postJSON(`/v1/agents/${agentName}/schedules/${scheduleId}/run`);

  if (result && !result._error) {
    // Started — poll for completion
    _pollScheduleDone(agentName, scheduleId, btn);
  } else {
    // Failed to start
    btn.innerHTML = '\u2717';
    btn.classList.remove('run-btn--running');
    btn.classList.add('run-btn--error');
    btn.title = result?._error || 'Failed to trigger';
    _markScheduleRunning(agentName, scheduleId, false);
    setTimeout(() => { btn.innerHTML = '\u25b6'; btn.disabled = false; btn.classList.remove('run-btn--error'); btn.title = 'Run now'; }, 3000);
  }
}

function _markScheduleRunning(agentName, scheduleId, running) {
  const scheds = state.schedules[agentName];
  if (!scheds) return;
  const s = scheds.find(x => x.id === scheduleId);
  if (s) s._running = running;
}

function _pollScheduleDone(agentName, scheduleId, btn) {
  let attempts = 0;
  const iv = setInterval(async () => {
    attempts++;
    const st = await fetchJSON(`/v1/agents/${agentName}/schedules/${scheduleId}/status`);

    if (st && !st.running) {
      clearInterval(iv);
      btn.innerHTML = '\u2713';
      btn.classList.remove('run-btn--running');
      btn.classList.add('run-btn--success');
      _markScheduleRunning(agentName, scheduleId, false);
      await refreshData();
      setTimeout(() => { btn.innerHTML = '\u25b6'; btn.disabled = false; btn.classList.remove('run-btn--success'); btn.title = 'Run now'; }, 2000);
    }

    if (attempts >= 120) {
      clearInterval(iv);
      btn.innerHTML = '?';
      btn.classList.remove('run-btn--running');
      btn.title = 'Timed out waiting for completion';
      _markScheduleRunning(agentName, scheduleId, false);
      setTimeout(() => { btn.innerHTML = '\u25b6'; btn.disabled = false; btn.title = 'Run now'; }, 3000);
    }
  }, 1000);
}

// ─── Render: Metrics Cards ────────────────────────

function renderMetricsCards() {
  const grid = document.getElementById('metricsGrid');
  const metricsAgents = state.metrics?.agents || {};

  if (!Object.keys(metricsAgents).length) {
    grid.innerHTML = `<div class="empty-state">
      <div class="empty-state-icon">&#128202;</div>
      No metrics yet. Send some messages to agents first.
    </div>`;
    return;
  }

  let html = '';
  for (const [name, m] of Object.entries(metricsAgents)) {
    const totalTokens = (m.input_tokens || 0) + (m.output_tokens || 0);
    const avgMs = m.messages > 0 ? Math.round((m.total_elapsed_ms || 0) / m.messages) : 0;

    html += `<div class="metric-card">
      <div class="metric-card-header">
        <span class="metric-card-name">${escHtml(name)}</span>
        ${statusBadge('active')}
      </div>
      <div class="metric-row"><span class="metric-label">Messages</span><span class="metric-value">${fmtNum(m.messages || 0)}</span></div>
      <div class="metric-row"><span class="metric-label">Tool Calls</span><span class="metric-value">${fmtNum(m.tool_calls || 0)}</span></div>
      <div class="metric-row"><span class="metric-label">Errors</span><span class="metric-value ${m.errors ? 'text-red' : ''}">${m.errors || 0}</span></div>
      <div class="metric-row"><span class="metric-label">Avg Response</span><span class="metric-value">${fmtMs(avgMs)}</span></div>
      <div class="metric-row"><span class="metric-label">Input Tokens</span><span class="metric-value">${fmtTokens(m.input_tokens || 0)}</span></div>
      <div class="metric-row"><span class="metric-label">Output Tokens</span><span class="metric-value">${fmtTokens(m.output_tokens || 0)}</span></div>
      <div class="metric-row"><span class="metric-label">Total Tokens</span><span class="metric-value fw-600">${fmtTokens(totalTokens)}</span></div>
      <div class="metric-row"><span class="metric-label">Last Activity</span><span class="metric-value">${m.last_activity ? fmtRelative(m.last_activity) : '<span class="text-dim">-</span>'}</span></div>`;

    // Recent errors
    const errors = m.recent_errors || [];
    if (errors.length) {
      html += `<div class="error-list">
        <div class="error-list-title">RECENT ERRORS</div>`;
      for (const err of errors.slice(0, 5)) {
        html += `<div class="error-item"><span class="error-time">${fmtTime(err.timestamp)}</span>${escHtml(err.error?.slice(0, 80))}</div>`;
      }
      html += '</div>';
    }

    html += '</div>';
  }

  grid.innerHTML = html;
}

// ─── Render: Models Table ─────────────────────────

function renderModelsTable() {
  const tbody = document.getElementById('modelsBody');
  const models = state.metrics?.models || {};

  if (!Object.keys(models).length) {
    tbody.innerHTML = '<tr><td colspan="8" class="empty-state">No model metrics yet. Send some messages first.</td></tr>';
    return;
  }

  let html = '';
  for (const [key, m] of Object.entries(models)) {
    // key is "provider/model"
    const slashIdx = key.indexOf('/');
    const provider = slashIdx > -1 ? key.slice(0, slashIdx) : key;
    const model = slashIdx > -1 ? key.slice(slashIdx + 1) : '';
    const modelShort = model.length > 30 ? model.slice(0, 27) + '...' : model;
    const totalTokens = (m.input_tokens || 0) + (m.output_tokens || 0);
    const avgMs = m.messages > 0 ? Math.round((m.total_elapsed_ms || 0) / m.messages) : 0;

    html += `<tr>
      <td style="color:var(--cyan);">${escHtml(provider)}</td>
      <td class="fw-600" style="color:#fff;">${escHtml(modelShort)}</td>
      <td>${fmtNum(m.messages || 0)}</td>
      <td>${fmtTokens(m.input_tokens || 0)}</td>
      <td>${fmtTokens(m.output_tokens || 0)}</td>
      <td class="fw-600">${fmtTokens(totalTokens)}</td>
      <td>${fmtMs(avgMs)}</td>
      <td>${m.errors ? '<span class="text-red">' + m.errors + '</span>' : '<span class="text-dim">0</span>'}</td>
    </tr>`;
  }

  tbody.innerHTML = html;
}

// ─── Render All ───────────────────────────────────

function renderAll() {
  renderStats();
  renderAgentsTable();
  renderSchedulesTable();
  renderModelsTable();
  renderMetricsCards();
}

// ─── Tabs ─────────────────────────────────────────

function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      // Deactivate all
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));

      // Activate clicked
      btn.classList.add('active');
      const panel = document.getElementById('tab-' + btn.dataset.tab);
      if (panel) panel.classList.add('active');
    });
  });
}

// ─── Refresh Timer ────────────────────────────────

function updateRefreshInfo() {
  const el = document.getElementById('refreshInfo');
  const now = new Date();
  el.textContent = 'updated ' + now.toLocaleTimeString();
}

// ─── Init ─────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  refreshData();
  setInterval(refreshData, REFRESH_INTERVAL);

  // Period selector triggers immediate refresh
  const periodSelect = document.getElementById('metricsPeriod');
  if (periodSelect) {
    periodSelect.addEventListener('change', () => refreshData());
  }

  // Close modals on backdrop click
  const configModal = document.getElementById('configModal');
  if (configModal) {
    configModal.addEventListener('click', (e) => {
      if (e.target === configModal) closeConfigModal();
    });
  }

  const createModal = document.getElementById('createAgentModal');
  if (createModal) {
    createModal.addEventListener('click', (e) => {
      if (e.target === createModal) closeCreateAgentModal();
    });
  }

  const deleteModal = document.getElementById('deleteAgentModal');
  if (deleteModal) {
    deleteModal.addEventListener('click', (e) => {
      if (e.target === deleteModal) closeDeleteAgentModal();
    });
  }
});
