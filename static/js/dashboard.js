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

// ─── Data Fetching ────────────────────────────────

async function refreshData() {
  const [status, agentsData, metricsData] = await Promise.all([
    fetchJSON('/v1/status'),
    fetchJSON('/v1/agents'),
    fetchJSON('/v1/metrics'),
  ]);

  state.status = status;
  state.agents = agentsData?.agents || [];
  state.metrics = metricsData;

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
    // Shorten model name for display
    const modelShort = model.length > 25 ? model.slice(0, 22) + '...' : model;

    html += `<tr>
      <td class="fw-600" style="color:#fff;">${escHtml(agent.name)}</td>
      <td>${statusBadge(agent.status || 'active')}</td>
      <td class="text-dim">${escHtml(provider)}/${escHtml(modelShort)}</td>
      <td>${sessions.length || 0}</td>
      <td>${fmtNum(m.messages || 0)}</td>
      <td>${fmtTokens(totalTokens)}</td>
      <td>${m.errors ? '<span class="text-red">' + m.errors + '</span>' : '<span class="text-dim">0</span>'}</td>
    </tr>`;
  }

  tbody.innerHTML = html;
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
    tbody.innerHTML = '<tr><td colspan="8" class="empty-state">No schedules configured. Run: alfred agent schedule add &lt;name&gt;</td></tr>';
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

    html += `<tr>
      <td class="fw-600" style="color:#fff;">${escHtml(s.agent_name)}</td>
      <td style="color:var(--cyan);">${escHtml(s.id)}</td>
      <td>${escHtml(s.human_schedule || s.cron)}</td>
      <td>${escHtml(task)}</td>
      <td>${statusHtml}</td>
      <td>${runsHtml}</td>
      <td>${lastRun}</td>
      <td>${nextRun}</td>
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

// ─── Render All ───────────────────────────────────

function renderAll() {
  renderStats();
  renderAgentsTable();
  renderSchedulesTable();
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
});
