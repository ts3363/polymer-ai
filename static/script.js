document.addEventListener('DOMContentLoaded', () => {
    const predictBtn = document.getElementById('predict-btn');
    const smilesInput = document.getElementById('smiles-input');
    const errorMsg = document.getElementById('error-msg');
    const skeletonGrid = document.getElementById('results-skeleton');
    const resultsGrid = document.getElementById('results-dashboard');
    const visGrid = document.getElementById('visualization-dashboard');
    const svgContainer = document.getElementById('polymer-svg-container');
    const ctx = document.getElementById('radarChart').getContext('2d');

    let radarChart = null;

    const formatters = {
        'Tg': { unit: '°C', desc: 'Glass Transition Temp' },
        'FFV': { unit: '', desc: 'Fractional Free Volume' },
        'Tc': { unit: 'W/(m·K)', desc: 'Thermal Conductivity' },
        'Density': { unit: 'g/cm³', desc: 'Polymer Density' },
        'Rg': { unit: 'Å', desc: 'Radius of Gyration' }
    };

    predictBtn.addEventListener('click', async () => {
        const smiles = smilesInput.value.trim();
        
        if (!smiles) {
            showError("Please enter a valid SMILES string.");
            return;
        }

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
            renderResults(data.predictions);
            renderVisualizations(data.svg, data.predictions);
        } catch (error) {
            skeletonGrid.classList.add('hidden');
            showError(error.message);
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

            const fmt = formatters[key] || { unit: '', desc: key };
            const formattedVal = typeof value === 'number' ? value.toFixed(3) : value;

            card.innerHTML = `
                <div class="card-title">${fmt.desc}</div>
                <div class="card-value">${formattedVal} <span class="card-unit">${fmt.unit}</span></div>
            `;
            resultsGrid.appendChild(card);
        });
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
