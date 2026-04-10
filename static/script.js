document.addEventListener('DOMContentLoaded', () => {
    fetchHistory();

    document.getElementById('prediction-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const btn = document.getElementById('analyze-btn');
        btn.textContent = 'Analyzing...';
        btn.disabled = true;

        const formData = {
            model: document.getElementById('model').value,
            age: parseInt(document.getElementById('age').value),
            gender: document.getElementById('gender').value,
            systolic: parseFloat(document.getElementById('systolic').value),
            diastolic: parseFloat(document.getElementById('diastolic').value),
            cholesterol: parseFloat(document.getElementById('cholesterol').value),
            heart_rate: parseFloat(document.getElementById('heart_rate').value),
            smoking: document.getElementById('smoking').checked ? 1 : 0,
            diabetes: document.getElementById('diabetes').checked ? 1 : 0,
            hypertension: document.getElementById('hypertension').checked ? 1 : 0,
            obesity: document.getElementById('obesity').checked ? 1 : 0,
            triglyceride: parseFloat(document.getElementById('triglyceride').value),
            ldl: parseFloat(document.getElementById('ldl').value),
            hdl: parseFloat(document.getElementById('hdl').value),
            diet_score: parseFloat(document.getElementById('diet_score').value),
            stress_level: parseFloat(document.getElementById('stress_level').value),
            pollution: parseFloat(document.getElementById('pollution').value),
            alcohol: parseFloat(document.getElementById('alcohol').value),
            physical_activity: parseFloat(document.getElementById('physical_activity').value),
            family_history: document.getElementById('family_history').checked ? 1 : 0,
            heart_attack_history: document.getElementById('heart_attack_history').checked ? 1 : 0,
            healthcare: document.getElementById('healthcare').checked ? 1 : 0
        };

        try {
            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            const data = await res.json();
            
            if (res.ok) {
                showResult(data.risk_percentage, data.model_used);
                fetchHistory();
                
                // Automatically close modal after 1.5s
                setTimeout(() => {
                    document.getElementById('upload-modal').classList.remove('active');
                    // Reset form styles
                    document.getElementById('prediction-result').classList.add('hidden');
                }, 1500);
            } else {
                alert('Error: ' + data.error);
            }
        } catch (err) {
            console.error(err);
            alert('Failed to connect to server.');
        } finally {
            btn.textContent = 'Analyze Risk';
            btn.disabled = false;
        }
    });
});

function showResult(riskPercentage, modelUsed) {
    const resBox = document.getElementById('prediction-result');
    const resText = document.getElementById('result-text');
    const resSub = document.getElementById('result-subtext');
    
    resBox.classList.remove('hidden');
    
    if (riskPercentage < 20) {
        resBox.style.borderColor = 'var(--success)';
        resBox.style.background = 'rgba(5, 150, 105, 0.05)';
        resText.style.color = 'var(--success)';
    } else if (riskPercentage < 50) {
        resBox.style.borderColor = 'var(--warning)';
        resBox.style.background = 'rgba(217, 119, 6, 0.05)';
        resText.style.color = 'var(--warning)';
    } else {
        resBox.style.borderColor = 'var(--danger)';
        resBox.style.background = 'rgba(223, 42, 42, 0.05)';
        resText.style.color = 'var(--danger)';
    }

    resText.textContent = `Risk Probability: ${riskPercentage}%`;
    resSub.textContent = `Processed with ${modelUsed}`;
}

let riskChartInstance = null;
let vitalsChartInstance = null;

async function fetchHistory() {
    try {
        const res = await fetch('/history');
        const logs = await res.json();
        
        updateTable(logs);
        updateCharts(logs);
        updateHRStats(logs);
    } catch (err) {
        console.error("Failed to fetch history", err);
    }
}

function updateTable(logs) {
    const tbody = document.querySelector('#logs-table tbody');
    tbody.innerHTML = '';
    
    const reversedLogs = [...logs].reverse();
    // Only show last 5 in the quick view
    const latestFive = reversedLogs.slice(0, 5);
    
    latestFive.forEach(log => {
        // format date cleanly (MMM DD)
        const dateObj = new Date(log.date);
        const dateStr = dateObj.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${dateStr}</td>
            <td style="color: ${getRiskColor(log.risk_percentage)}; font-weight:700">${log.risk_percentage}%</td>
            <td>${log.cholesterol_level}</td>
            <td>${log.systolic_bp}/${log.diastolic_bp}</td>
            <td><button class="btn-action">Edit</button></td>
        `;
        tbody.appendChild(tr);
    });
}

function getRiskColor(risk) {
    if (risk < 20) return 'var(--text-main)';
    if (risk < 50) return 'var(--warning)';
    return 'var(--danger)';
}

function updateHRStats(logs) {
    if (logs.length === 0) return;
    const latest = logs[logs.length - 1]; // chronologically last
    const avgHrEl = document.getElementById('avg-hr');
    if (avgHrEl) {
        avgHrEl.textContent = latest.heart_rate;
    }

    const statusBox = document.getElementById('status-card-box');
    const statusTitle = document.getElementById('status-card-title');
    const statusDesc = document.getElementById('status-card-desc');

    if (!statusBox || !statusTitle || !statusDesc) {
        return;
    }
    
    if (latest.risk_percentage > 50) {
        statusBox.style.background = 'var(--danger)';
        statusTitle.textContent = 'STATUS: CRITICAL';
        statusDesc.textContent = 'High cardiovascular risk detected. Medical consultation recommended.';
    } else {
        statusBox.style.background = 'var(--accent)';
        statusTitle.textContent = 'STATUS: STABLE';
        statusDesc.textContent = 'No significant anomalies detected in recent clinical telemetry.';
    }
}

function updateCharts(logs) {
    // Only show last 10 for charts
    let chartLogs = logs;
    if(logs.length > 10) chartLogs = logs.slice(logs.length - 10);
    
    const dates = chartLogs.map(l => {
        const d = new Date(l.date);
        return d.toLocaleDateString('en-US', { day: 'numeric', month: 'short' }).toUpperCase();
    });
    const risks = chartLogs.map(l => l.risk_percentage);
    const chol = chartLogs.map(l => l.cholesterol_level);
    const sys = chartLogs.map(l => l.systolic_bp);
    const dia = chartLogs.map(l => l.diastolic_bp);

    // Light theme Chart.js config
    Chart.defaults.color = '#9ca3af';
    Chart.defaults.font.family = 'Inter';

    const gridOpts = {
        color: 'rgba(0,0,0,0.03)',
        drawBorder: false,
        borderDash: [5, 5]
    };
    const noGrid = { display: false };

    // Risk Chart
    const ctxRisk = document.getElementById('riskChart').getContext('2d');
    if (riskChartInstance) riskChartInstance.destroy();
    
    riskChartInstance = new Chart(ctxRisk, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Risk %',
                data: risks,
                borderColor: '#DF2A2A',
                backgroundColor: 'rgba(223, 42, 42, 0.08)',
                borderWidth: 3,
                tension: 0.1,
                fill: true,
                pointBackgroundColor: '#DF2A2A',
                pointRadius: 0,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: noGrid, ticks: { font: { size: 10 } } },
                y: { grid: gridOpts, min: 0, max: 100, border: { display: false } }
            },
            plugins: {
                legend: { display: false },
                tooltip: { backgroundColor: 'white', titleColor: 'black', bodyColor: 'black', borderColor: '#eee', borderWidth: 1 }
            }
        }
    });

    // Vitals Chart
    const ctxVitals = document.getElementById('vitalsChart').getContext('2d');
    if (vitalsChartInstance) vitalsChartInstance.destroy();
    
    vitalsChartInstance = new Chart(ctxVitals, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Cholesterol',
                    data: chol,
                    borderColor: '#D97706',
                    borderWidth: 3,
                    borderDash: [3, 4],
                    tension: 0.2,
                    pointRadius: 0,
                    pointHoverRadius: 5
                },
                {
                    label: 'Systolic BP',
                    data: sys,
                    borderColor: '#2563eb',
                    borderWidth: 3,
                    tension: 0.2,
                    pointRadius: 0,
                    pointHoverRadius: 5
                },
                {
                    label: 'Diastolic BP',
                    data: dia,
                    borderColor: '#059669',
                    borderWidth: 3,
                    tension: 0.2,
                    pointRadius: 0,
                    pointHoverRadius: 5
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: noGrid, display: false }, // Hide x axis completely for vitals to look cleaner
                y: { grid: gridOpts, border: { display: false } }
            },
            plugins: {
                legend: { display: false },
                tooltip: { backgroundColor: 'white', titleColor: 'black', bodyColor: 'black', borderColor: '#eee', borderWidth: 1 }
            }
        }
    });
}
