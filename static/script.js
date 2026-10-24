document.addEventListener('DOMContentLoaded', () => {
    const predictBtn = document.getElementById('predict-btn');
    const smilesInput = document.getElementById('smiles-input');
    const errorMsg = document.getElementById('error-msg');
    const skeletonGrid = document.getElementById('results-skeleton');
    const resultsGrid = document.getElementById('results-dashboard');
    const visGrid = document.getElementById('visualization-dashboard');
    const svgContainer = document.getElementById('polymer-svg-container');
    const ctx = document.getElementById('radarChart').getContext('2d');

    const clearBtn = document.createElement('button');
    clearBtn.className = 'clear-btn hidden';
    clearBtn.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>';
    smilesInput.parentNode.insertBefore(clearBtn, smilesInput.nextSibling);

    let radarChart = null;
    let currentPredictions = null;

    const formatters = {
        'Tg': { unit: '°C', desc: 'Glass Transition Temp', error: 8.5, min: -100, max: 400, info: 'Temperature region where the polymer transitions from a hard, glassy material to a soft, rubbery material.' },
        'FFV': { unit: '', desc: 'Fractional Free Volume', error: 0.012, min: 0.1, max: 0.5, info: 'The ratio of unoccupied volume to total polymer volume, crucial for gas permeability.' },
        'Tc': { unit: 'W/(m·K)', desc: 'Thermal Conductivity', error: 0.015, min: 0.1, max: 0.4, info: 'The rate at which heat passes through the polymer material.' },
        'Density': { unit: 'g/cm³', desc: 'Polymer Density', error: 0.02, min: 0.8, max: 2.0, info: 'Mass per unit volume of the polymer chain network.' },
        'Rg': { unit: 'Å', desc: 'Radius of Gyration', error: 1.2, min: 5, max: 40, info: 'The root-mean-square distance of the parts of the polymer chain from its center of mass.' }
    };

    smilesInput.addEventListener('input', () => {
        if (smilesInput.value.length > 0) {
            clearBtn.classList.remove('hidden');
            smilesInput.classList.remove('input-error');
        } else {
            clearBtn.classList.add('hidden');
        }
    });

    clearBtn.addEventListener('click', () => {
        smilesInput.value = '';
        clearBtn.classList.add('hidden');
        smilesInput.focus();
    });

    predictBtn.addEventListener('click', async () => {
        const smiles = smilesInput.value.trim();
        
        if (!smiles) {
            showError("Invalid SMILES format. Please ensure the string is not empty and includes wildcard (*) atoms.");
            smilesInput.classList.add('input-error');
            return;
        }

        // Button Loading State
        const originalBtnText = predictBtn.innerHTML;
        predictBtn.innerHTML = '<span class="spinner"></span> Analyzing...';
        predictBtn.disabled = true;

        // Hide errors, show skeleton
        errorMsg.classList.remove('error-visible');
        errorMsg.classList.add('error-hidden');
        resultsGrid.innerHTML = '';
        resultsGrid.classList.add('hidden');
        visGrid.classList.add('hidden');
        skeletonGrid.classList.remove('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ smiles: smiles })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Server Error');
            }

            const data = await response.json();
            currentPredictions = data.predictions;
            renderResults(data.predictions);
            renderVisualizations(data.svg, data.predictions);
        } catch (error) {
            skeletonGrid.classList.add('hidden');
            showError(error.message);
            smilesInput.classList.add('input-error');
        } finally {
            predictBtn.innerHTML = originalBtnText;
            predictBtn.disabled = false;
        }
    });

    function showError(message) {
        errorMsg.textContent = message;
        errorMsg.classList.remove('error-hidden');
        errorMsg.classList.add('error-visible');
    }

    function renderResults(predictions) {
        skeletonGrid.classList.add('hidden');
        resultsGrid.classList.remove('hidden');

        const entries = Object.entries(predictions || {});
        
        entries.forEach(([key, value], index) => {
            const card = document.createElement('div');
            card.className = `stat-card card-${key} card-animate`;
            card.style.animationDelay = `${index * 0.1}s`;

            const fmt = formatters[key];
            const valStr = value.toFixed(3);
            const [whole, decimal] = valStr.split('.');
            
            // Calculate progress percentage
            let progress = ((value - fmt.min) / (fmt.max - fmt.min)) * 100;
            progress = Math.max(0, Math.min(100, progress));

            card.innerHTML = `
                <div class="card-header">
                    <div class="card-title-wrap">
                        <div class="card-title">${fmt.desc}</div>
                        <div class="tooltip-icon">i
                            <span class="tooltip-text">${fmt.info}</span>
                        </div>
                    </div>
                    <button class="copy-btn" title="Copy value" data-val="${valStr}">
                        <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                    </button>
                </div>
                <div class="card-value">${whole}.<span class="dim-decimal">${decimal}</span> <span class="card-unit">${fmt.unit}</span></div>
                <div class="card-error">± ${fmt.error} ${fmt.unit} (MAE)</div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: ${progress}%"></div>
                </div>
            `;
            resultsGrid.appendChild(card);
        });

        // Add Export Button as the 6th card
        const exportCard = document.createElement('div');
        exportCard.className = 'stat-card export-card card-animate';
        exportCard.style.animationDelay = `${entries.length * 0.1}s`;
        exportCard.innerHTML = `
            <div class="export-content">
                <svg viewBox="0 0 24 24" width="32" height="32" stroke="currentColor" stroke-width="1.5" fill="none"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                <h3>Export Data</h3>
                <p>Download predictions as CSV</p>
                <button id="export-csv-btn" class="export-btn">Download CSV</button>
            </div>
        `;
        resultsGrid.appendChild(exportCard);

        document.getElementById('export-csv-btn').addEventListener('click', exportToCSV);

        // Bind copy buttons
        document.querySelectorAll('.copy-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const val = e.currentTarget.getAttribute('data-val');
                navigator.clipboard.writeText(val);
                const originalHTML = e.currentTarget.innerHTML;
                e.currentTarget.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" stroke="#10b981" stroke-width="2" fill="none"><polyline points="20 6 9 17 4 12"></polyline></svg>';
                setTimeout(() => e.currentTarget.innerHTML = originalHTML, 2000);
            });
        });
    }

    function exportToCSV() {
        if (!currentPredictions) return;
        const smiles = smilesInput.value.trim();
        let csvContent = "data:text/csv;charset=utf-8,";
        csvContent += "SMILES,Property,Value,Unit,Margin of Error\n";
        
        Object.entries(currentPredictions).forEach(([key, value]) => {
            const fmt = formatters[key];
            csvContent += `"${smiles}",${key},${value.toFixed(3)},${fmt.unit},±${fmt.error}\n`;
        });
        
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "polymer_predictions.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    function renderVisualizations(svgString, predictions) {
        visGrid.classList.remove('hidden');
        
        // 1. Render SVG
        if (svgString) {
            svgContainer.innerHTML = svgString;
            // Make SVG responsive
            const svgElement = svgContainer.querySelector('svg');
            if (svgElement) {
                svgElement.style.width = '100%';
                svgElement.style.height = 'auto';
            }
        } else {
            svgContainer.innerHTML = '<p style="color:var(--text-secondary)">No structure available</p>';
        }

        // 2. Render Radar Chart
        // Normalize values to 0-1 range for a balanced radar chart
        const norm = {
            'Tg': Math.max(0, Math.min(1, (predictions.Tg + 150) / 650)),
            'FFV': Math.max(0, Math.min(1, predictions.FFV / 1.0)),
            'Tc': Math.max(0, Math.min(1, predictions.Tc / 1.5)),
            'Density': Math.max(0, Math.min(1, predictions.Density / 2.5)),
            'Rg': Math.max(0, Math.min(1, predictions.Rg / 50.0))
        };

        const chartData = {
            labels: ['Tg', 'FFV', 'Tc', 'Density', 'Rg'],
            datasets: [{
                label: 'Normalized Properties',
                data: [norm.Tg, norm.FFV, norm.Tc, norm.Density, norm.Rg],
                fill: true,
                backgroundColor: 'rgba(59, 130, 246, 0.2)',
                borderColor: 'rgba(59, 130, 246, 1)',
                pointBackgroundColor: 'rgba(139, 92, 246, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(139, 92, 246, 1)'
            }]
        };

        if (radarChart) {
            radarChart.destroy();
        }

        radarChart = new Chart(ctx, {
            type: 'radar',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        pointLabels: {
                            color: '#94a3b8',
                            font: { family: 'Outfit', size: 14 }
                        },
                        ticks: {
                            display: false,
                            min: 0,
                            max: 1
                        }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.chart.data.labels[context.dataIndex];
                                const rawVal = predictions[label];
                                return ` ${label}: ${rawVal.toFixed(3)} ${formatters[label].unit}`;
                            }
                        }
                    }
                }
            }
        });
    }

    smilesInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            predictBtn.click();
        }
    });
});
