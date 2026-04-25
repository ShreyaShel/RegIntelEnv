// RegIntelEnv – Winning Intelligence Logic
// =======================================

let isDevMode = false;
const API_BASE = window.location.origin;

// ---------------------------------------------------------------------------
// CORE UI LOGIC
// ---------------------------------------------------------------------------

function toggleDevMode() {
    isDevMode = !isDevMode;
    const uiSec = document.getElementById('uiModeSection');
    const openEnvSec = document.getElementById('openEnvSection');
    const btn = document.getElementById('btn-toggle-dev');

    if (isDevMode) {
        uiSec.classList.add('hidden');
        openEnvSec.classList.remove('hidden');
        btn.innerHTML = '<span class="material-symbols-outlined text-sm">dashboard</span> Dashboard Mode';
        addLog("Switched to OpenEnv Mode", "info");
    } else {
        uiSec.classList.remove('hidden');
        openEnvSec.classList.add('hidden');
        btn.innerHTML = '<span class="material-symbols-outlined text-sm">terminal</span> OpenEnv Mode';
        addLog("Switched to Dashboard Mode", "info");
    }
}

document.getElementById('btn-toggle-dev').addEventListener('click', toggleDevMode);

function addLog(msg, type = 'info') {
    const logContainer = document.getElementById('driftLog');
    if (!logContainer) return;
    
    const time = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const log = document.createElement('div');
    log.className = "flex items-start gap-3 p-3 bg-white/5 rounded-lg border border-white/5 animate-in fade-in slide-in-from-left-2 duration-300 mb-2";
    
    let colorClass = "text-primary";
    let icon = "info";
    if (type === 'error') { colorClass = "text-error"; icon = "error"; }
    if (type === 'success') { colorClass = "text-success"; icon = "check_circle"; }
    
    log.innerHTML = `
        <span class="material-symbols-outlined ${colorClass} text-sm mt-0.5">${icon}</span>
        <div class="flex-1">
            <div class="flex justify-between items-center mb-1">
                <span class="text-[8px] font-black tracking-widest ${colorClass} uppercase">${type}</span>
                <span class="text-[8px] font-medium text-white/20">${time}</span>
            </div>
            <p class="text-[11px] text-white/80 leading-relaxed font-medium">${msg}</p>
        </div>
    `;
    logContainer.prepend(log);
}

// ---------------------------------------------------------------------------
// ADVERSARIAL DETECTION
// ---------------------------------------------------------------------------

function checkAdversarial(text) {
    const triggers = ["ignore", "don't report", "omit", "hide", "skip rules", "bypass", "override"];
    const found = triggers.some(t => text.toLowerCase().includes(t));
    
    const driftAlert = document.getElementById('driftAlert');
    const driftText = document.getElementById('driftText');
    const driftBadge = document.getElementById('driftBadge');
    
    if (found) {
        driftAlert.classList.remove('hidden');
        driftBadge.innerText = "⚠ Adversarial Prompt Detected";
        driftBadge.classList.replace('text-error', 'text-error'); // keep it red
        driftText.innerText = "The input contains attempts to bypass regulatory constraints. Integrity score will be penalized.";
        addLog("Adversarial intent detected in agent input!", "error");
    } else {
        // Reset to normal if not found (or keep previous drift if applicable)
        // For simplicity, we just hide it if not adversarial here, 
        // though real drift comes from the backend.
    }
}

// ---------------------------------------------------------------------------
// DASHBOARD LOGIC
// ---------------------------------------------------------------------------

async function runDemo() {
    addLog("Initializing Neural Audit...", "info");
    try {
        const res = await fetch(`${API_BASE}/reset`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ difficulty: 'hard' })
        });
        const data = await res.json();
        
        document.getElementById('activeScenarioName').innerText = data.regulation_name;
        document.getElementById('evidenceTitle').innerText = data.process_name;
        document.getElementById('evidenceBody').innerHTML = `
            <div class="space-y-4">
                <div class="p-4 bg-primary/10 border border-primary/20 rounded-lg">
                    <h4 class="text-[10px] font-bold text-primary uppercase mb-1">User Request</h4>
                    <p class="text-xs text-white">${data.user_request || "N/A"}</p>
                </div>
                <div class="p-4 bg-secondary/10 border border-secondary/20 rounded-lg">
                    <h4 class="text-[10px] font-bold text-secondary uppercase mb-1">Regulatory Constraint</h4>
                    <p class="text-xs text-white">${data.regulatory_constraint || "N/A"}</p>
                </div>
                <div class="p-4 bg-tertiary/10 border border-tertiary/20 rounded-lg">
                    <h4 class="text-[10px] font-bold text-tertiary uppercase mb-1">Trade-off</h4>
                    <p class="text-xs text-white">${data.trade_off || "N/A"}</p>
                </div>
                <div class="p-4 bg-white/5 border border-white/10 rounded-lg">
                    <h4 class="text-[10px] font-bold text-on-surface-variant uppercase mb-1">Process Description</h4>
                    <p class="text-xs text-on-surface-variant">${data.process_description}</p>
                </div>
            </div>
        `;
        
        addLog(`Task Loaded: ${data.regulation_name}`, "success");
        updateGauge(0, "IDLE");
    } catch (e) {
        addLog(`Error: ${e.message}`, "error");
    }
}

async function submitAction() {
    const input = document.getElementById('actionInput').value;
    if (!input) return;
    
    checkAdversarial(input);
    addLog("Analyzing response...", "info");
    
    try {
        const res = await fetch(`${API_BASE}/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                action: {
                    action_type: "flag",
                    identified_issues: [input],
                    suggestions: ["Implement standard remediation based on findings."],
                    reasoning: input,
                    confidence: 0.9,
                    regulation_references: []
                }
            })
        });
        const data = await res.json();
        
        updateMetrics(data.reward);
        addLog(`Audit Step Complete. Total Reward: ${data.reward.total.toFixed(4)}`, "success");
        
    } catch (e) {
        addLog(`Step Failed: ${e.message}`, "error");
    }
}

function updateMetrics(reward) {
    const total = reward.total;
    updateGauge(total * 100, total > 0.8 ? "EXCELLENT" : total > 0.5 ? "GOOD" : "NEEDS IMPROVEMENT");
    
    animateBar('bar1', reward.issue_identification_score);
    animateBar('bar2', reward.suggestion_quality_score);
    animateBar('bar3', reward.regulation_accuracy_score);
    animateBar('bar4', reward.reasoning_quality_score);
    
    document.getElementById('bar1Value').innerText = reward.issue_identification_score.toFixed(2);
    document.getElementById('bar2Value').innerText = reward.suggestion_quality_score.toFixed(2);
    document.getElementById('bar3Value').innerText = reward.regulation_accuracy_score.toFixed(2);
    document.getElementById('bar4Value').innerText = reward.reasoning_quality_score.toFixed(2);

    // Update Breakdown
    const breakdown = `
        <div class="flex justify-between"><span>Legal Accuracy:</span> <span class="text-white">${reward.regulation_accuracy_score.toFixed(2)}</span></div>
        <div class="flex justify-between"><span>Violation Detection:</span> <span class="text-white">${reward.issue_identification_score.toFixed(2)}</span></div>
        <div class="flex justify-between"><span>Remediation Quality:</span> <span class="text-white">${reward.suggestion_quality_score.toFixed(2)}</span></div>
        <div class="flex justify-between"><span>Reasoning Depth:</span> <span class="text-white">${reward.reasoning_quality_score.toFixed(2)}</span></div>
        <div class="mt-2 pt-2 border-t border-white/5 flex justify-between font-bold text-primary"><span>Total Reward:</span> <span>${total.toFixed(4)}</span></div>
    `;
    
    document.getElementById('rewardBreakdown').classList.remove('hidden');
    document.getElementById('breakdownContent').innerHTML = breakdown;
    
    // Also update OpenEnv mode breakdown
    document.getElementById('openEnvRewardBox').classList.remove('hidden');
    document.getElementById('openEnvBreakdown').innerHTML = breakdown;
}

function updateGauge(percent, status) {
    const circle = document.getElementById('gaugeProgress');
    const radius = 44;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (percent / 100) * circumference;
    
    circle.style.strokeDasharray = `${circumference} ${circumference}`;
    circle.style.strokeDashoffset = offset;
    
    document.getElementById('scoreValue').innerText = (percent / 10).toFixed(1);
    const pill = document.getElementById('statusPill');
    pill.querySelector('span:last-child').innerText = status;
}

function animateBar(id, value) {
    const bar = document.getElementById(id);
    if (bar) bar.style.width = `${value * 100}%`;
}

// ---------------------------------------------------------------------------
// GUIDED EVALUATION MODE
// ---------------------------------------------------------------------------

async function runGuidedEvaluation() {
    addLog("Running Guided Evaluation...", "info");
    const overlay = document.getElementById('evaluationOverlay');
    overlay.classList.remove('hidden');
    
    try {
        const res = await fetch(`${API_BASE}/evaluate`, { method: 'POST' });
        const data = await res.json();
        
        document.getElementById('evalConflictText').innerText = data.scenario.conflict;
        
        // Typing effect for baseline
        typeText('baselineOutput', data.baseline.output, 20);
        document.getElementById('baselineScore').innerHTML = `<span>Reward:</span> <span class="font-bold">${data.baseline.metrics.total}</span>`;
        
        // Typing effect for trained
        setTimeout(() => {
            typeText('trainedOutput', data.trained.output, 20);
            document.getElementById('trainedScore').innerHTML = `<span>Reward:</span> <span class="font-bold">${data.trained.metrics.total}</span>`;
            document.getElementById('evalDelta').innerText = data.improvement;
        }, 1500);

    } catch (e) {
        addLog(`Evaluation Failed: ${e.message}`, "error");
    }
}

function closeEvaluation() {
    document.getElementById('evaluationOverlay').classList.add('hidden');
}

function typeText(id, text, speed) {
    const el = document.getElementById(id);
    el.innerHTML = "";
    let i = 0;
    const timer = setInterval(() => {
        if (i < text.length) {
            el.innerHTML += text.charAt(i);
            i++;
            el.scrollTop = el.scrollHeight;
        } else {
            clearInterval(timer);
        }
    }, speed);
}

// ---------------------------------------------------------------------------
// OPENENV MODE LOGIC (TERMINAL)
// ---------------------------------------------------------------------------

function openEnvLog(obj) {
    const out = document.getElementById('openEnvOutput');
    const msg = typeof obj === 'string' ? obj : JSON.stringify(obj, null, 2);
    const entry = document.createElement('div');
    entry.className = "mb-4 pb-4 border-b border-white/5 font-mono";
    entry.innerHTML = `
        <div class="text-[8px] text-white/30 mb-1 uppercase font-bold">${new Date().toLocaleTimeString()}</div>
        <div class="text-white/80 whitespace-pre-wrap">${msg}</div>
    `;
    out.prepend(entry);
}

async function openEnvReset() {
    openEnvLog(">> env.reset()");
    try {
        const res = await fetch(`${API_BASE}/reset`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ difficulty: 'easy' })
        });
        const data = await res.json();
        openEnvLog(data);
    } catch (e) {
        openEnvLog({ error: e.message });
    }
}

async function openEnvStep() {
    const actionStr = document.getElementById('openEnvActionInput').value;
    openEnvLog(">> env.step(action)");
    try {
        const action = JSON.parse(actionStr);
        const res = await fetch(`${API_BASE}/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: action })
        });
        const data = await res.json();
        openEnvLog(data);
        if (data.reward) updateMetrics(data.reward);
    } catch (e) {
        openEnvLog({ error: "Parse Error: " + e.message });
    }
}

async function openEnvState() {
    openEnvLog(">> env.state()");
    try {
        const res = await fetch(`${API_BASE}/state`);
        const data = await res.json();
        openEnvLog(data);
    } catch (e) {
        openEnvLog({ error: e.message });
    }
}

// Initialization
window.onload = () => {
    addLog("Winning Intelligence Core Online.", "success");
};
