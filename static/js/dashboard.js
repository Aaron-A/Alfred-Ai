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
  const period = document.getElementById('metricsPeriod')?.value || 'day';
  const metricsUrl = '/v1/metrics?period=' + period;

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

  // Fetch trading data for all agents and vector queries (on every refresh)
  fetchAllTradingData();
  fetchVectorQueries();
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

function fmtCost(n) {
  if (n == null || isNaN(n) || n === 0) return '$0.00';
  if (n < 0.01) return '<$0.01';
  return '$' + n.toFixed(2);
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
  let totalMessages = 0, totalTokens = 0, totalMs = 0, totalErrors = 0, totalCost = 0;
  let agentCount = 0;

  for (const [name, m] of Object.entries(agents)) {
    totalMessages += m.messages || 0;
    totalTokens += (m.input_tokens || 0) + (m.output_tokens || 0);
    totalMs += m.total_elapsed_ms || 0;
    totalErrors += m.errors || 0;
    totalCost += m.estimated_cost || 0;
    if (m.messages > 0) agentCount++;
  }

  const avgMs = totalMessages > 0 ? Math.round(totalMs / totalMessages) : 0;

  return { totalMessages, totalTokens, totalMs, totalErrors, avgMs, agentCount, totalCost };
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

  // Est. Cost
  document.getElementById('statCost').textContent = fmtCost(agg.totalCost);
  const costPerMsg = agg.totalMessages > 0 ? (agg.totalCost / agg.totalMessages) : 0;
  document.getElementById('statCostSub').textContent = agg.totalCost > 0
    ? `~${fmtCost(costPerMsg)}/msg`
    : 'no cost data';

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
    tbody.innerHTML = '<tr><td colspan="8" class="empty-state">No agents configured. Run: alfred setup</td></tr>';
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

    const cost = m.estimated_cost || 0;
    html += `<tr>
      <td class="fw-600" style="color:#fff;">${escHtml(agent.name)}</td>
      <td>${statusBadge(agent.status || 'active')}</td>
      <td><span class="text-dim">${escHtml(provider)}/${escHtml(modelShort)}</span> ${editBtn}${deleteBtn}</td>
      <td>${sessions.length || 0}</td>
      <td>${fmtNum(m.messages || 0)}</td>
      <td>${fmtTokens(totalTokens)}</td>
      <td class="cost-value">${fmtCost(cost)}</td>
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

  // Populate config fields with current values (or defaults)
  document.getElementById('modalMaxRounds').value = agent.max_tool_rounds ?? 10;
  document.getElementById('modalScheduleRounds').value = agent.schedule_max_tool_rounds ?? 15;
  document.getElementById('modalDailyCost').value = agent.max_daily_cost ?? 0;
  document.getElementById('modalContextBudget').value = Math.round((agent.context_budget_pct ?? 0.60) * 100);

  // SAVE always enabled — let user save whenever they want
  const saveBtn = document.getElementById('modalSaveBtn');
  saveBtn.textContent = 'SAVE';
  saveBtn.disabled = false;
  saveBtn.className = 'modal-btn modal-btn--save modal-btn--active';

  // Show modal
  modal.classList.add('active');
}

function closeConfigModal() {
  document.getElementById('configModal').classList.remove('active');
  hideTooltip();
}

function showTooltip(el, text, evt) {
  if (evt) { evt.preventDefault(); evt.stopPropagation(); }
  const tooltip = document.getElementById('modalTooltip');
  // Toggle — click again to dismiss
  if (tooltip.classList.contains('active') && tooltip._source === el) {
    hideTooltip();
    return;
  }
  tooltip.textContent = text;
  tooltip._source = el;

  // Position below the ? icon
  const rect = el.getBoundingClientRect();
  const modalBox = el.closest('.modal-box');
  const modalRect = modalBox.getBoundingClientRect();
  tooltip.style.top = (rect.bottom - modalRect.top + 6) + 'px';
  tooltip.style.left = '0';
  tooltip.style.right = '0';
  tooltip.classList.add('active');
}

function hideTooltip() {
  const tooltip = document.getElementById('modalTooltip');
  tooltip.classList.remove('active');
  tooltip._source = null;
}

// Dismiss tooltip when clicking outside
document.addEventListener('click', (e) => {
  if (!e.target.classList.contains('modal-help')) {
    hideTooltip();
  }
});

function onModalProviderChange() {
  const providerSelect = document.getElementById('modalProvider');
  const modelSelect = document.getElementById('modalModel');
  const newProvider = providerSelect.value;

  // Repopulate model dropdown with first model selected
  const models = state.providerModels[newProvider] || [];
  modelSelect.innerHTML = buildModelOptions(newProvider, models[0] || '');

}

function onModalModelChange() {
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

  const isDirty = (
    newProvider !== origProvider ||
    newModel !== origModel ||
    parseInt(document.getElementById('modalMaxRounds').value) !== (agent.max_tool_rounds ?? 10) ||
    parseInt(document.getElementById('modalScheduleRounds').value) !== (agent.schedule_max_tool_rounds ?? 15) ||
    parseFloat(document.getElementById('modalDailyCost').value) !== (agent.max_daily_cost ?? 0) ||
    parseInt(document.getElementById('modalContextBudget').value) !== Math.round((agent.context_budget_pct ?? 0.60) * 100)
  );
  const saveBtn = document.getElementById('modalSaveBtn');
  saveBtn.disabled = !isDirty;
  saveBtn.className = isDirty ? 'modal-btn modal-btn--save modal-btn--active' : 'modal-btn modal-btn--save';
}

async function saveModalConfig() {
  const modal = document.getElementById('configModal');
  const agentName = modal.dataset.agent;
  const provider = document.getElementById('modalProvider').value;
  const model = document.getElementById('modalModel').value;
  const max_tool_rounds = parseInt(document.getElementById('modalMaxRounds').value);
  const schedule_max_tool_rounds = parseInt(document.getElementById('modalScheduleRounds').value);
  const max_daily_cost = parseFloat(document.getElementById('modalDailyCost').value);
  const context_budget_pct = parseInt(document.getElementById('modalContextBudget').value) / 100;

  const saveBtn = document.getElementById('modalSaveBtn');
  saveBtn.disabled = true;
  saveBtn.textContent = 'SAVING...';

  const result = await patchJSON(`/v1/agents/${agentName}/config`, {
    provider, model, max_tool_rounds, schedule_max_tool_rounds,
    max_daily_cost, context_budget_pct,
  });

  if (result?.message) {
    // Update local state
    const agent = state.agents.find(a => a.name === agentName);
    if (agent) {
      agent.provider = provider;
      agent.model = model;
      agent.max_tool_rounds = max_tool_rounds;
      agent.schedule_max_tool_rounds = schedule_max_tool_rounds;
      agent.max_daily_cost = max_daily_cost;
      agent.context_budget_pct = context_budget_pct;
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
      <div class="metric-row"><span class="metric-label">Estimated Cost</span><span class="metric-value cost-value">${fmtCost(m.estimated_cost || 0)}</span></div>
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
    tbody.innerHTML = '<tr><td colspan="9" class="empty-state">No model metrics yet. Send some messages first.</td></tr>';
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

    const cost = m.estimated_cost || 0;
    html += `<tr>
      <td style="color:var(--cyan);">${escHtml(provider)}</td>
      <td class="fw-600" style="color:#fff;">${escHtml(modelShort)}</td>
      <td>${fmtNum(m.messages || 0)}</td>
      <td>${fmtTokens(m.input_tokens || 0)}</td>
      <td>${fmtTokens(m.output_tokens || 0)}</td>
      <td class="fw-600">${fmtTokens(totalTokens)}</td>
      <td class="cost-value">${fmtCost(cost)}</td>
      <td>${fmtMs(avgMs)}</td>
      <td>${m.errors ? '<span class="text-red">' + m.errors + '</span>' : '<span class="text-dim">0</span>'}</td>
    </tr>`;
  }

  tbody.innerHTML = html;
}

// ─── Header Restart ──────────────────────────────

async function headerRestart() {
  const btn = document.getElementById('headerRestartBtn');
  if (!btn || btn.disabled) return;

  btn.disabled = true;
  btn.textContent = 'RESTARTING...';
  btn.className = 'header-restart-btn header-restart-btn--restarting';

  const result = await postJSON('/v1/admin/reload');

  if (result?.status === 'restarting' || result?.status === 'restarted') {
    let attempts = 0;
    const poll = setInterval(async () => {
      attempts++;
      try {
        const res = await fetch(API + '/health');
        if (res.ok) {
          clearInterval(poll);
          btn.textContent = 'RESTARTED';
          btn.className = 'header-restart-btn header-restart-btn--done';
          setTimeout(() => {
            btn.textContent = 'RESTART';
            btn.className = 'header-restart-btn';
            btn.disabled = false;
            refreshData();
          }, 2000);
        }
      } catch (e) {
        // API not up yet
      }
      if (attempts >= 20) {
        clearInterval(poll);
        btn.textContent = 'TIMEOUT';
        btn.className = 'header-restart-btn header-restart-btn--error';
        setTimeout(() => {
          btn.textContent = 'RESTART';
          btn.className = 'header-restart-btn';
          btn.disabled = false;
        }, 3000);
      }
    }, 1500);
  } else {
    btn.textContent = 'FAILED';
    btn.className = 'header-restart-btn header-restart-btn--error';
    setTimeout(() => {
      btn.textContent = 'RESTART';
      btn.className = 'header-restart-btn';
      btn.disabled = false;
    }, 3000);
  }
}

// ─── Trading Tab (Multi-Agent) ───────────────────

// Per-agent trading state: { data, barsBuffer, meta }
const tradingState = {};
let tradingAgents = [];       // list from /v1/trading/agents
let activeTradingAgent = '';  // currently displayed agent
const MAX_BARS_HOURS = 7;

async function fetchTradingAgents() {
  const agents = await fetchJSON('/v1/trading/agents');
  if (!agents || !agents.length) return;
  tradingAgents = agents;

  // Default to first agent if none selected
  if (!activeTradingAgent || !agents.find(a => a.name === activeTradingAgent)) {
    activeTradingAgent = agents[0].name;
  }

  // Initialize state for each agent
  for (const a of agents) {
    if (!tradingState[a.name]) {
      tradingState[a.name] = { data: null, barsBuffer: [] };
    }
  }

  renderTradingAgentTabs();
}

function renderTradingAgentTabs() {
  const container = document.getElementById('tradingAgentTabs');
  if (!container) return;

  container.innerHTML = tradingAgents.map(a => {
    const active = a.name === activeTradingAgent ? ' active' : '';
    return `<button class="trading-agent-tab${active}" data-agent="${escHtml(a.name)}">${escHtml(a.display_name)}</button>`;
  }).join('');

  // Click handlers
  container.querySelectorAll('.trading-agent-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      activeTradingAgent = btn.dataset.agent;
      container.querySelectorAll('.trading-agent-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderTradingTab();
    });
  });
}

async function fetchAllTradingData() {
  // Discover agents on first call
  if (!tradingAgents.length) {
    await fetchTradingAgents();
  }

  // Fetch data for ALL agents in parallel (so tab switching is instant)
  const promises = tradingAgents.map(a => fetchTradingDataForAgent(a.name));
  await Promise.all(promises);

  renderTradingTab();
}

async function fetchTradingDataForAgent(agentName) {
  const agentState = tradingState[agentName] || { data: null, barsBuffer: [] };
  tradingState[agentName] = agentState;

  // Incremental: only fetch bars since the last one we have
  let url = `/v1/trading/status?agent=${encodeURIComponent(agentName)}`;
  if (agentState.barsBuffer.length > 0) {
    const lastT = agentState.barsBuffer[agentState.barsBuffer.length - 1].t;
    url += '&since=' + encodeURIComponent(lastT);
  }

  const data = await fetchJSON(url);
  if (!data) return;

  agentState.data = data;

  // Merge new bars into buffer
  const newBars = data.bars || [];
  if (agentState.barsBuffer.length === 0) {
    agentState.barsBuffer = newBars;
  } else if (newBars.length > 0) {
    const lastKnown = agentState.barsBuffer[agentState.barsBuffer.length - 1].t;
    for (const bar of newBars) {
      if (bar.t > lastKnown) agentState.barsBuffer.push(bar);
    }
  }

  // Trim: crypto 1m → 420 bars, stock 5m → 84 bars (7h)
  const assetType = data.asset_type || 'crypto';
  const barsPerHour = assetType === 'stock' ? 12 : 60;
  const maxBars = MAX_BARS_HOURS * barsPerHour;
  if (agentState.barsBuffer.length > maxBars) {
    agentState.barsBuffer = agentState.barsBuffer.slice(-maxBars);
  }
}

function renderTradingTab() {
  const agentState = tradingState[activeTradingAgent];
  if (!agentState?.data) return;

  const data = agentState.data;
  const meta = tradingAgents.find(a => a.name === activeTradingAgent) || {};

  renderTradingStatusCards(data);
  renderTradingChart(agentState.barsBuffer, meta);
  renderTradesTable(data.trades || [], data.asset_type);
}

function fmtUSD(val) {
  const n = parseFloat(val);
  if (isNaN(n)) return '—';
  return '$' + n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtTradeTime(isoStr) {
  if (!isoStr) return '—';
  try {
    const d = new Date(isoStr);
    if (isNaN(d.getTime())) return isoStr;
    return d.toLocaleString('en-US', {
      month: 'short', day: 'numeric',
      hour: 'numeric', minute: '2-digit',
      hour12: true
    });
  } catch { return isoStr; }
}

function renderTradingStatusCards(data) {
  const assetType = data.asset_type || 'crypto';

  // Equity
  const equity = data.account?.equity;
  const eqEl = document.getElementById('tradingEquity');
  const eqSub = document.getElementById('tradingEquitySub');
  if (equity) {
    eqEl.textContent = fmtUSD(equity);
    eqEl.style.color = 'var(--green)';
    const bp = data.account?.buying_power;
    eqSub.textContent = bp ? `Buying power: ${fmtUSD(bp)}` : '';
  } else {
    eqEl.textContent = '—';
    eqEl.style.color = '';
    eqSub.textContent = 'No account data';
  }

  // Position
  const pos = data.position;
  const posEl = document.getElementById('tradingPosition');
  const posSub = document.getElementById('tradingPositionSub');
  if (pos) {
    const qty = parseFloat(pos.qty || 0);
    const side = qty > 0 ? 'LONG' : qty < 0 ? 'SHORT' : 'FLAT';
    const qtyStr = assetType === 'stock' ? Math.abs(qty).toFixed(0) : Math.abs(qty).toFixed(4);
    posEl.textContent = `${side} ${qtyStr}`;
    const pl = parseFloat(pos.unrealized_pl || 0);
    posEl.style.color = pl >= 0 ? 'var(--green)' : 'var(--red)';
    const plStr = (pl >= 0 ? '+' : '') + fmtUSD(pl);
    const entry = pos.entry_price ? `Entry: ${fmtUSD(pos.entry_price)}` : '';
    posSub.textContent = `${plStr}${entry ? ' | ' + entry : ''}`;
  } else {
    posEl.textContent = 'FLAT';
    posEl.style.color = 'var(--text-dim)';
    posSub.textContent = 'No open position';
  }

  // Bot status
  const botEl = document.getElementById('tradingBotStatus');
  const botSub = document.getElementById('tradingBotSub');
  if (data.bot_running) {
    botEl.textContent = 'RUNNING';
    botEl.style.color = 'var(--green)';
    botSub.textContent = 'Connected to Alpaca';
  } else {
    botEl.textContent = 'STOPPED';
    botEl.style.color = 'var(--red)';
    botSub.textContent = 'Bot is not running';
  }

  // Total P&L — equity-based (ground truth from Alpaca)
  const pnlEl = document.getElementById('tradingPnl');
  const pnlSub = document.getElementById('tradingPnlSub');
  const eqVal = parseFloat(equity || 0);
  const initialEquity = parseFloat(data.initial_equity || 0);
  const trades = data.trades || [];
  const tradeCount = trades.filter(t => parseFloat(t.realized_pnl || t.pnl || 0) !== 0).length;

  if (eqVal > 0 && initialEquity > 0) {
    const totalPnl = eqVal - initialEquity;
    const returnPct = ((totalPnl / initialEquity) * 100).toFixed(2);
    pnlEl.textContent = (totalPnl >= 0 ? '+' : '') + fmtUSD(totalPnl);
    pnlEl.style.color = totalPnl >= 0 ? 'var(--green)' : 'var(--red)';
    // Sum estimated commissions from trade data
    const totalComm = trades.reduce((sum, t) => sum + parseFloat(t.commission || 0), 0);
    let subText = `${returnPct}% return`;
    if (tradeCount > 0) subText += ` | ${tradeCount} trades`;
    if (totalComm > 0) subText += ` | $${totalComm.toFixed(2)} fees`;
    pnlSub.textContent = subText;
  } else {
    pnlEl.textContent = '$0.00';
    pnlEl.style.color = 'var(--text-dim)';
    pnlSub.textContent = 'No account data';
  }
}

function renderTradingChart(bars, meta) {
  const canvas = document.getElementById('tradingChart');
  if (!canvas) return;

  const assetType = meta.asset_type || 'crypto';
  const symbol = meta.symbol || '—';

  // Always update chart header (even with no bars)
  const symbolEl = document.getElementById('chartSymbol');
  const timeframeEl = document.getElementById('chartTimeframe');
  const priceEl = document.getElementById('chartPrice');
  const changeEl = document.getElementById('chartChange');
  if (symbolEl) symbolEl.textContent = symbol.replace('/', ' / ');
  if (timeframeEl) timeframeEl.textContent = assetType === 'stock' ? '5m bars' : '1m bars';

  const ctx = canvas.getContext('2d');

  // No bars — clear canvas and show message
  if (!bars.length) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = '#555';
    ctx.font = '12px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Market closed — waiting for data', rect.width / 2, rect.height / 2);
    if (priceEl) priceEl.textContent = '—';
    if (changeEl) { changeEl.textContent = ''; changeEl.className = 'chart-change'; }
    return;
  }
  const dpr = window.devicePixelRatio || 1;

  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;
  const PAD_TOP = 20;
  const PAD_BOTTOM = 24;
  const PAD_LEFT = 60;
  const PAD_RIGHT = 16;
  const chartW = W - PAD_LEFT - PAD_RIGHT;
  const chartH = H - PAD_TOP - PAD_BOTTOM;

  const closes = bars.map(b => b.c);
  const highs = bars.map(b => b.h);
  const lows = bars.map(b => b.l);
  const allPrices = [...highs, ...lows];
  const minP = Math.min(...allPrices);
  const maxP = Math.max(...allPrices);
  const range = maxP - minP || 1;
  const padRange = range * 0.05;
  const pMin = minP - padRange;
  const pMax = maxP + padRange;

  const priceToY = (p) => PAD_TOP + chartH - ((p - pMin) / (pMax - pMin)) * chartH;
  const barW = chartW / bars.length;

  // Price label formatting — 2 decimals for stocks, 0 for BTC
  const priceFmt = assetType === 'stock'
    ? { minimumFractionDigits: 2, maximumFractionDigits: 2 }
    : { minimumFractionDigits: 0, maximumFractionDigits: 0 };

  ctx.clearRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = '#1a1a1a';
  ctx.lineWidth = 0.5;
  const gridCount = 5;
  for (let i = 0; i <= gridCount; i++) {
    const y = PAD_TOP + (i / gridCount) * chartH;
    ctx.beginPath();
    ctx.moveTo(PAD_LEFT, y);
    ctx.lineTo(W - PAD_RIGHT, y);
    ctx.stroke();

    const price = pMax - (i / gridCount) * (pMax - pMin);
    ctx.fillStyle = '#555';
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(price.toLocaleString('en-US', priceFmt), PAD_LEFT - 8, y + 3);
  }

  // Candlesticks
  for (let i = 0; i < bars.length; i++) {
    const b = bars[i];
    const x = PAD_LEFT + i * barW + barW / 2;
    const isUp = b.c >= b.o;

    ctx.strokeStyle = isUp ? '#22c55e' : '#ef4444';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, priceToY(b.h));
    ctx.lineTo(x, priceToY(b.l));
    ctx.stroke();

    const bodyTop = priceToY(Math.max(b.o, b.c));
    const bodyBot = priceToY(Math.min(b.o, b.c));
    const bodyH = Math.max(bodyBot - bodyTop, 1);
    const candleW = Math.max(barW * 0.6, 2);

    ctx.fillStyle = isUp ? '#22c55e' : '#ef4444';
    ctx.fillRect(x - candleW / 2, bodyTop, candleW, bodyH);
  }

  // Current price line
  const lastClose = closes[closes.length - 1];
  const lastY = priceToY(lastClose);
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 0.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(PAD_LEFT, lastY);
  ctx.lineTo(W - PAD_RIGHT, lastY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = '#fff';
  ctx.font = 'bold 10px JetBrains Mono, monospace';
  ctx.textAlign = 'left';
  ctx.fillText(lastClose.toLocaleString('en-US', priceFmt), W - PAD_RIGHT + 4, lastY + 3);

  // Time labels
  ctx.fillStyle = '#555';
  ctx.font = '8px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  const labelInterval = Math.max(Math.floor(bars.length / 6), 1);
  for (let i = 0; i < bars.length; i += labelInterval) {
    const b = bars[i];
    const x = PAD_LEFT + i * barW + barW / 2;
    const d = new Date(b.t);
    const hh = d.getHours().toString().padStart(2, '0');
    const mm = d.getMinutes().toString().padStart(2, '0');
    ctx.fillText(`${hh}:${mm}`, x, H - 6);
  }

  // Update header price (priceEl/changeEl declared at top of function)
  if (priceEl) priceEl.textContent = fmtUSD(lastClose);

  if (changeEl && bars.length > 1) {
    const firstClose = closes[0];
    const diff = lastClose - firstClose;
    const pct = (diff / firstClose * 100).toFixed(2);
    const sign = diff >= 0 ? '+' : '';
    changeEl.textContent = `${sign}${pct}%`;
    changeEl.className = diff >= 0 ? 'chart-change chart-change--up' : 'chart-change chart-change--down';
  }
}

function renderTradesTable(trades, assetType) {
  const tbody = document.getElementById('tradesBody');
  if (!tbody) return;

  if (!trades.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="text-dim" style="text-align:center;padding:24px;">No trades yet</td></tr>';
    return;
  }

  const qtyDecimals = (assetType === 'stock') ? 0 : 5;

  let html = '';
  const reversed = [...trades].reverse();
  for (const t of reversed) {
    const time = fmtTradeTime(t.exit_time || t.entry_time);
    const dir = (t.direction || '').toUpperCase();
    const dirClass = dir === 'LONG' ? 'text-green' : dir === 'SHORT' ? 'text-red' : '';
    const qty = t.qty ? parseFloat(t.qty).toFixed(qtyDecimals) : '—';
    const entryPrice = t.entry_price ? fmtUSD(t.entry_price) : '—';
    const exitPrice = t.exit_price ? fmtUSD(t.exit_price) : '—';
    const reason = t.reason || '—';
    const isOpen = t.status === 'open';

    let pnlHtml = '<span class="text-dim">—</span>';
    if (t.pnl) {
      const pnlNum = parseFloat(t.pnl);
      if (!isNaN(pnlNum)) {
        const pnlColor = pnlNum >= 0 ? 'text-green' : 'text-red';
        pnlHtml = `<span class="${pnlColor}">${pnlNum >= 0 ? '+' : ''}${fmtUSD(pnlNum)}</span>`;
        if (t.commission) {
          pnlHtml += `<div style="font-size:9px;color:var(--text-dim);margin-top:1px;">fee: $${parseFloat(t.commission).toFixed(2)}</div>`;
        }
      }
    }

    const rowStyle = isOpen ? ' style="opacity:0.7;font-style:italic;"' : '';
    const openTag = isOpen ? ' <span class="text-dim">(open)</span>' : '';

    html += `<tr${rowStyle}>
      <td>${escHtml(time)}</td>
      <td class="${dirClass} fw-600">${escHtml(dir)}${openTag}</td>
      <td>${escHtml(qty)}</td>
      <td>${entryPrice}</td>
      <td>${exitPrice}</td>
      <td>${pnlHtml}</td>
      <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;">${escHtml(reason)}</td>
    </tr>`;
  }

  tbody.innerHTML = html;
}

// ─── Vector Queries ───────────────────────────────

async function fetchVectorQueries() {
  const period = document.getElementById('metricsPeriod')?.value || 'day';
  const data = await fetchJSON(`/v1/vector-queries?period=${period}&limit=30`);
  if (data) {
    state.vectorQueries = data.queries || [];
    renderVectorQueries();
  }
}

function renderVectorQueries() {
  const tbody = document.getElementById('vectorQueriesBody');
  if (!tbody) return;

  const queries = state.vectorQueries || [];
  if (!queries.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="text-dim" style="text-align:center;padding:24px;">No vector queries yet</td></tr>';
    return;
  }

  let html = '';
  for (const q of queries) {
    const queryText = (q.query_text || '').length > 50
      ? q.query_text.slice(0, 47) + '...'
      : (q.query_text || '-');
    const avgDist = q.avg_distance != null ? q.avg_distance.toFixed(3) : '-';
    const distColor = q.avg_distance != null
      ? (q.avg_distance < 0.4 ? 'text-green' : q.avg_distance < 0.7 ? 'text-yellow' : 'text-red')
      : 'text-dim';

    html += `<tr>
      <td>${fmtRelative(q.timestamp)}</td>
      <td style="color:var(--cyan);">${escHtml(q.agent || '-')}</td>
      <td class="vq-query" title="${escHtml(q.query_text || '')}">${escHtml(queryText)}</td>
      <td>${escHtml(q.memory_type || 'all')}</td>
      <td>${q.result_count || 0}</td>
      <td class="${distColor}">${avgDist}</td>
      <td>${q.elapsed_ms ? q.elapsed_ms + 'ms' : '-'}</td>
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
  // Trading tab renders from its own data fetch
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
  const refreshId = setInterval(refreshData, REFRESH_INTERVAL);

  // Clean up interval when navigating away
  window.addEventListener('beforeunload', () => clearInterval(refreshId));

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

  // Redraw chart on resize (throttled to avoid excessive redraws)
  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (barsBuffer.length) renderBTCChart(barsBuffer);
    }, 200);
  });
});
