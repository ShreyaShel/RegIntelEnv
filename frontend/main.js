const API_BASE = '';
const driftLog = document.getElementById('driftLog');

const logMessages = [
  "Analyzing EU AI Act Article 52 metadata...",
  "Compliance drift detected in NIS2 supply chain module.",
  "Recalculating multi-dimensional reward for GDPR compliance...",
  "Agent 'Regulatory_Sentinel_04' has completed phase 1.",
  "System cooling initiated for intense compliance processing.",
  "New regulatory signal received: PSD3 Draft Update.",
  "Optimizing coalition engine for multi-agent coordination.",
  "Compliance Score updated: +0.02%",
  "Exporting audit trail to secure ledger node...",
  "Sector 09: Atmospheric drift anomaly detected.",
  "Verifying B2B silo parity for GDPR Data Lifecycle."
];

function addLog(msg, isHighPriority = false) {
  if (!driftLog) return;
  
  const time = new Date().toLocaleTimeString('en-GB', { hour12: false });
  const message = msg || logMessages[Math.floor(Math.random() * logMessages.length)];
  const priority = isHighPriority || (msg ? false : Math.random() > 0.7);
  
  const entry = document.createElement('div');
  entry.className = 'flex items-center justify-between p-4 bg-surface-container-low rounded-xl group/item hover:bg-surface-container transition-colors';
  
  entry.innerHTML = `
    <div class="flex items-center gap-4">
        <div class="w-10 h-10 ${priority ? 'bg-error/20' : 'bg-secondary/20'} rounded-full flex items-center justify-center">
            <span class="material-symbols-outlined ${priority ? 'text-error' : 'text-secondary'} text-sm">${priority ? 'warning' : 'info'}</span>
        </div>
        <div>
            <p class="text-sm font-bold text-white">${message}</p>
            <p class="text-[10px] text-on-surface-variant">Sector ${Math.floor(Math.random() * 12).toString().padStart(2, '0')} • ${time}</p>
        </div>
    </div>
    <span class="text-[10px] font-bold ${priority ? 'text-error' : 'text-secondary'} tracking-widest">${priority ? 'HIGH' : 'STABLE'}</span>
  `;
  
  driftLog.prepend(entry);
  if (driftLog.children.length > 10) driftLog.removeChild(driftLog.lastChild);
}

// Backend Integration
async function updateDashboardState() {
  try {
    const res = await fetch(`${API_BASE}/state`);
    const data = await res.json();
    const state = data.state;
    
    if (state && state.episode_active && state.last_observation) {
        updateUIFromState(state);
    }
  } catch (err) {
    console.error("Sync error", err);
  }
}

function updateUIFromState(state) {
    const obs = state.last_observation;
    const rew = state.last_reward || { total: 0, issue_identification_score: 0, suggestion_quality_score: 0, regulation_accuracy_score: 0, reasoning_quality_score: 0 };

    // Update Progress Gauge
    const gauge = document.getElementById('gaugeProgress');
    const scoreVal = document.getElementById('scoreValue');
    if (gauge && scoreVal) {
        const percentage = (rew.total * 100).toFixed(1);
        const offset = 276 - (276 * (rew.total)); // 276 is the circumference
        gauge.style.strokeDashoffset = offset;
        scoreVal.innerHTML = `${percentage}<span class="text-xl text-secondary">%</span>`;
    }

    // Update Meta
    const sceneName = document.getElementById('activeScenarioName');
    if (sceneName) sceneName.textContent = obs.task_id.toUpperCase();

    const title = document.getElementById('evidenceTitle');
    if (title) title.textContent = obs.process_name;

    const body = document.getElementById('evidenceBody');
    if (body) body.innerHTML = `<div class="space-y-4">
        <div class="p-4 bg-primary/5 border-l-2 border-primary rounded text-xs text-primary mb-4">
            <strong>REGULATION:</strong> ${obs.regulation_name}<br>
            <strong>ENTITY:</strong> ${obs.company_name}
        </div>
        <div class="whitespace-pre-line">${obs.process_description}</div>
    </div>`;

    const statusPill = document.getElementById('statusPill');
    if (statusPill) {
        statusPill.innerHTML = `
            <span class="w-1.5 h-1.5 rounded-full bg-secondary animate-ping"></span>
            <span class="text-[8px] font-bold text-secondary tracking-widest uppercase">AUDIT ACTIVE</span>
        `;
    }

    const counter = document.getElementById('stepCounter');
    if (counter) counter.textContent = `STEP: ${obs.step_number} / ${obs.max_steps}`;

    // Update Drift Alert
    const driftAlert = document.getElementById('driftAlert');
    const driftText = document.getElementById('driftText');
    if (obs.regulatory_drift && driftAlert && driftText) {
        driftText.textContent = obs.regulatory_drift;
        driftAlert.classList.remove('hidden');
        if (driftText.dataset.lastDrift !== obs.regulatory_drift) {
            addLog("REGULATORY DRIFT DETECTED", true);
            driftText.dataset.lastDrift = obs.regulatory_drift;
        }
    } else if (driftAlert) {
        driftAlert.classList.add('hidden');
    }

    // Update Bars
    updateBar('bar1', 'bar1Value', rew.issue_identification_score);
    updateBar('bar2', 'bar2Value', rew.suggestion_quality_score);
    updateBar('bar3', 'bar3Value', rew.regulation_accuracy_score);
    updateBar('bar4', 'bar4Value', rew.reasoning_quality_score);
}

function updateBar(id, valId, val) {
    const bar = document.getElementById(id);
    const text = document.getElementById(valId);
    if (bar && text) {
        bar.style.width = (val * 100) + '%';
        text.textContent = val.toFixed(2);
    }
}

async function loadTasks() {
    const taskContainer = document.getElementById('task-grid');
    if (!taskContainer) return;

    try {
        const res = await fetch(`${API_BASE}/tasks`);
        const { tasks } = await res.json();
        
        taskContainer.innerHTML = '';
        Object.values(tasks).forEach(task => {
            const card = document.createElement('div');
            const diffClass = {'easy':'text-secondary','medium':'text-tertiary','hard':'text-error'}[task.difficulty];
            card.className = `group relative p-8 rounded-xl bg-surface-container-high transition-all hover:bg-surface-bright border border-transparent hover:border-primary/20 overflow-hidden`;
            card.innerHTML = `
                <div class="flex justify-between items-start mb-6">
                    <div class="flex items-center gap-3">
                        <span class="material-symbols-outlined ${diffClass}">security</span>
                        <span class="text-[10px] font-bold tracking-widest text-on-surface-variant uppercase">REF: ${task.task_id}</span>
                    </div>
                </div>
                <h3 class="text-2xl font-black mb-2 tracking-tight">${task.process_name.toUpperCase()}</h3>
                <p class="text-on-surface-variant text-sm mb-8">${task.company_name} | ${task.regulation_name}</p>
                <div class="flex justify-between items-end">
                    <span class="text-[10px] font-bold ${diffClass} uppercase tracking-widest">${task.difficulty} MODE</span>
                    <button onclick="startSession('${task.task_id}')" class="px-6 py-2 rounded-lg bg-primary text-on-primary-fixed font-bold hover:brightness-110 transition-all">INITIALIZE</button>
                </div>
            `;
            taskContainer.appendChild(card);
        });
    } catch (err) { console.error(err); }
}

window.startSession = async (taskId) => {
    try {
        const res = await fetch(`${API_BASE}/reset`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ task_id: taskId })
        });
        window.location.href = '/';
    } catch (err) { alert("Reset failed."); }
};

window.runDemo = () => {
    const input = document.getElementById('actionInput');
    if (!input) return;
    
    addLog("✨ Demo Protocol Initiated: Syncing optimized compliance data...", true);
    input.value = "Identified lack of formal DPO appointment (CFO performing role as side-duty) violating GDPR Art.37. Also detected missing Standard Contractual Clauses (SCCs) for US data transfers violating Art.46.";
    
    setTimeout(() => {
        submitAction('flag');
    }, 1500);
}

window.submitAction = async (type) => {
    const input = document.getElementById('actionInput');
    const val = input.value.trim();
    if (type !== 'conclude' && !val) return;

    const payload = {
        action: {
            action_type: type,
            identified_issues: type === 'flag' ? [val] : [],
            suggestions: type === 'suggest' ? [val] : [],
            regulation_references: [],
            reasoning: "Manual override via Command Console.",
            confidence: 1.0
        }
    };

    input.value = '';
    addLog(`Deploying ${type.toUpperCase()} command...`);

    try {
        const res = await fetch(`${API_BASE}/step`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        updateUIFromState({
            episode_active: true,
            last_observation: data.observation,
            last_reward: data.reward
        });
        addLog(`Protocol Evaluated: Reward +${data.reward.total.toFixed(3)}`, data.reward.total > 0.5);
    } catch (err) { addLog("ACTION FAILED", true); }
};

// --- Insights, Settings, Profile Logic ---

async function showInsights() {
    try {
        const response = await fetch(`${API_BASE}/state`);
        const data = await response.json();
        const history = data.history || [];
        
        if (history.length === 0) {
            alert("📊 NO HISTORY YET\n\nRun an audit task first to generate regulatory insights.");
            return;
        }
        
        const avgScore = (history.reduce((a, b) => a + b.total, 0) / history.length).toFixed(2);
        const feedback = history.map((h, i) => `Audit ${i+1}: Score ${h.total.toFixed(2)}`).join('\n');
        
        alert(`📊 REGULATORY INSIGHTS\n\nAverage Integrity Score: ${avgScore}\n\nRecent Audits:\n${feedback}`);
    } catch (e) {
        alert("Unable to fetch insights. Ensure backend is running.");
    }
}

async function showSettings() {
    try {
        const response = await fetch(`${API_BASE}/config`);
        const config = await response.json();
        
        const newModel = prompt("⚙️ ENVIRONMENT SETTINGS\n\nConfigure AI Grader Model:", config.model_name);
        if (newModel && newModel !== config.model_name) {
            await fetch(`${API_BASE}/config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ...config, model_name: newModel })
            });
            alert("Settings Updated Successfully!");
        }
    } catch (e) {
        alert("Settings currently unavailable.");
    }
}

function updateProfile(state) {
    const totalReward = state.cumulative_reward || 0;
    let rank = "Junior Auditor";
    if (totalReward > 10) rank = "Senior Compliance Officer";
    if (totalReward > 50) rank = "Lead Regulatory Sentinel";
    
    const profileBtn = document.getElementById('btn-profile');
    if (profileBtn) profileBtn.title = `Rank: ${rank}`;
}

// Initializing UI Wiring
document.addEventListener('DOMContentLoaded', () => {
    const insightsBtn = document.getElementById('btn-insights');
    const settingsBtn = document.getElementById('btn-settings');
    const profileBtn = document.getElementById('btn-profile');
    const toggleDevBtn = document.getElementById('btn-toggle-dev');

    if (insightsBtn) insightsBtn.onclick = showInsights;
    if (settingsBtn) settingsBtn.onclick = showSettings;
    if (profileBtn) profileBtn.onclick = () => alert("👤 AUDITOR PROFILE\n\nName: Shreya\nSystem Status: Lead Regulatory Sentinel\nNetwork: Encrypted (OpenEnv)");
    
    if (toggleDevBtn) {
        toggleDevBtn.onclick = () => {
            const ui = document.getElementById('uiModeSection');
            const dev = document.getElementById('openEnvSection');
            if (ui && dev) {
                if (ui.classList.contains('hidden')) {
                    ui.classList.remove('hidden');
                    dev.classList.add('hidden');
                    toggleDevBtn.innerHTML = '<span class="material-symbols-outlined text-sm">terminal</span> OpenEnv Playground';
                } else {
                    ui.classList.add('hidden');
                    dev.classList.remove('hidden');
                    toggleDevBtn.innerHTML = '<span class="material-symbols-outlined text-sm">dashboard</span> Exit Playground';
                }
            }
        };
    }
});

// --- OpenEnv Playground Logic ---
function logOpenEnv(data, prefix = '') {
    const out = document.getElementById('openEnvOutput');
    if (!out) return;
    const time = new Date().toLocaleTimeString('en-GB', { hour12: false, minute: '2-digit', second: '2-digit' });
    const content = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
    out.innerHTML += `<div class="mb-4"><span class="text-primary">[${time}] ${prefix}</span>\n${content}</div>`;
    out.scrollTop = out.scrollHeight;
}

window.openEnvReset = async () => {
    logOpenEnv('Calling env.reset(difficulty="medium")...', 'REQUEST');
    try {
        const res = await fetch(`${API_BASE}/reset`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ difficulty: 'medium' })
        });
        const obs = await res.json();
        logOpenEnv(obs, 'RESPONSE');
        // Keep UI in sync
        updateDashboardState();
    } catch (err) { logOpenEnv(err.message, 'ERROR'); }
};

window.openEnvState = async () => {
    logOpenEnv('Calling env.state()...', 'REQUEST');
    try {
        const res = await fetch(`${API_BASE}/state`);
        const state = await res.json();
        logOpenEnv(state, 'RESPONSE');
    } catch (err) { logOpenEnv(err.message, 'ERROR'); }
};

window.openEnvStep = async () => {
    const inputStr = document.getElementById('openEnvActionInput').value;
    let actionObj;
    try {
        actionObj = JSON.parse(inputStr);
    } catch (e) {
        logOpenEnv("Invalid JSON format in Action Input.", "ERROR");
        return;
    }

    logOpenEnv(actionObj, 'REQUEST (env.step)');
    try {
        const res = await fetch(`${API_BASE}/step`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ action: actionObj })
        });
        const data = await res.json();
        logOpenEnv(data, 'RESPONSE');
        // Keep UI in sync
        updateUIFromState({
            episode_active: true,
            last_observation: data.observation,
            last_reward: data.reward
        });
    } catch (err) { logOpenEnv(err.message, 'ERROR'); }
};

// Init
if (driftLog) {
    for(let i=0; i<3; i++) addLog();
    setInterval(addLog, 8000);
    setInterval(updateDashboardState, 2000); // Polling state
}

updateDashboardState();
if (document.getElementById('task-grid')) loadTasks();
