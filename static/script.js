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
            alcohol: document.getElementById('alcohol-user').checked ? 1 : 0,
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
                showResult(data.risk_percentage, data.model_used, data.ensemble_breakdown);
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

    const cadButton = document.getElementById('cad-btn');
    if (cadButton) {
        cadButton.addEventListener('click', async () => {
            const payload = {
                age: parseFloat(document.getElementById('age').value),
                gender: document.getElementById('gender').value,
                height: parseFloat(document.getElementById('height').value),
                weight: parseFloat(document.getElementById('weight').value),
                ap_hi: parseFloat(document.getElementById('systolic').value),
                ap_lo: parseFloat(document.getElementById('diastolic').value),
                cholesterol: parseInt(document.getElementById('cholesterol').value, 10),
                gluc: parseInt(document.getElementById('gluc').value, 10),
                smoke: document.getElementById('smoking').checked ? 1 : 0,
                alco: document.getElementById('alcohol-user').checked ? 1 : 0,
                active: parseFloat(document.getElementById('physical_activity').value) > 0 ? 1 : 0
            };

            const resultBox = document.getElementById('cad-result');
            const resultText = document.getElementById('cad-result-text');
            const resultSub = document.getElementById('cad-result-subtext');
            resultBox.classList.remove('hidden');
            resultText.textContent = 'Running CAD prediction...';
            resultSub.textContent = '';

            cadButton.disabled = true;
            cadButton.textContent = 'Running...';

            try {
                const res = await fetch('/predict/cad', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();

                if (!res.ok) {
                    resultText.textContent = `Error: ${data.error || 'CAD prediction failed.'}`;
                    return;
                }

                resultText.textContent = `CAD Result: ${data.risk_label}`;
                resultSub.textContent = `Probability: ${data.probability ?? 'N/A'}%`;
            } catch (err) {
                resultText.textContent = 'Error: Failed to connect to CAD endpoint.';
            } finally {
                cadButton.disabled = false;
                cadButton.textContent = 'Run CAD Prediction';
            }
        });
    }

    const ecgForm = document.getElementById('ecg-form');
    if (ecgForm) {
        ecgForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const fileInput = document.getElementById('ecg-image');
            const resultBox = document.getElementById('ecg-result');
            resultBox.classList.remove('hidden');

            if (!fileInput.files || fileInput.files.length === 0) {
                resultBox.textContent = 'Error: Please choose an ECG image file.';
                return;
            }

            const formData = new FormData();
            formData.append('ecg_image', fileInput.files[0]);
            resultBox.textContent = 'Running ECG prediction...';

            try {
                const res = await fetch('/predict/ecg', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();

                if (!res.ok) {
                    resultBox.textContent = `Error: ${data.error || 'ECG prediction failed.'}`;
                    return;
                }

                resultBox.textContent = `Result: ${data.label} (class ${data.prediction}) | Confidence: ${data.probability ?? 'N/A'}% | Preprocess: ${data.preprocess}`;
            } catch (err) {
                resultBox.textContent = 'Error: Failed to connect to ECG endpoint.';
            }
        });
    }
});

function showResult(riskPercentage, modelUsed, ensembleBreakdown) {
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

    if (ensembleBreakdown && ensembleBreakdown.method === 'average' && ensembleBreakdown.rf != null && ensembleBreakdown.xgb != null) {
        resSub.textContent = `${modelUsed} | RF: ${ensembleBreakdown.rf}% | XGB: ${ensembleBreakdown.xgb}%`;
    } else {
        resSub.textContent = `Processed with ${modelUsed}`;
    }
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
