// Global variables
let chart = null;
let modelLoaded = false;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    loadAvailableModels();
    setupEventListeners();
    initChart();
});

// Setup event listeners
function setupEventListeners() {
    document.getElementById('loadModelBtn').addEventListener('click', loadModel);
    document.getElementById('inferBtn').addEventListener('click', runInference);
}

// Load available models
async function loadAvailableModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        const select = document.getElementById('modelSelect');
        select.innerHTML = '';
        
        if (data.models.length === 0) {
            select.innerHTML = '<option value="">No models found</option>';
            return;
        }
        
        data.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            select.appendChild(option);
        });
        
        // Select first model by default
        select.selectedIndex = 0;
    } catch (error) {
        console.error('Error loading models:', error);
        showError('Failed to load models');
    }
}

// Load model
async function loadModel() {
    const modelName = document.getElementById('modelSelect').value;
    
    if (!modelName) {
        showError('Please select a model');
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch('/api/load_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_name: modelName })
        });
        
        const data = await response.json();
        
        if (data.success) {
            modelLoaded = true;
            document.getElementById('inferBtn').disabled = false;
            document.getElementById('sampleRange').textContent = `Range: 0 - ${data.num_samples - 1}`;
            document.getElementById('sampleIdx').max = data.num_samples - 1;
            
            showSuccess(`Model ${modelName} loaded successfully!`);
            await updateModelInfo();
        } else {
            showError(data.error || 'Failed to load model');
        }
    } catch (error) {
        console.error('Error loading model:', error);
        showError('Failed to load model: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// Update model info
async function updateModelInfo() {
    try {
        const response = await fetch('/api/info');
        const data = await response.json();
        
        if (data.loaded) {
            const infoDiv = document.getElementById('modelInfo');
            infoDiv.innerHTML = `
                <p><span class="status-dot green"></span><strong>Model:</strong> ${data.model}</p>
                <p><strong>Dataset:</strong> ${data.dataset}</p>
                <p><strong>Input Length:</strong> ${data.seq_len}</p>
                <p><strong>Prediction Length:</strong> ${data.pred_len}</p>
                <p><strong>Channels:</strong> ${data.num_channels}</p>
                <p><strong>Test Samples:</strong> ${data.num_samples}</p>
            `;
            
            // Update channel select
            const channelSelect = document.getElementById('channelIdx');
            channelSelect.innerHTML = '';
            for (let i = 0; i < data.num_channels; i++) {
                const option = document.createElement('option');
                option.value = i;
                option.textContent = `Channel ${i}`;
                channelSelect.appendChild(option);
            }
        }
    } catch (error) {
        console.error('Error updating model info:', error);
    }
}

// Run inference
async function runInference() {
    if (!modelLoaded) {
        showError('Please load a model first');
        return;
    }
    
    const sampleIdx = parseInt(document.getElementById('sampleIdx').value);
    const channelIdx = parseInt(document.getElementById('channelIdx').value);
    
    showLoading(true);
    
    try {
        const response = await fetch('/api/inference', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                sample_idx: sampleIdx,
                channel_idx: channelIdx
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            updateChart(data);
            updateMetrics(data.metrics);
            showSuccess('Inference completed!');
        } else {
            showError(data.error || 'Inference failed');
        }
    } catch (error) {
        console.error('Error running inference:', error);
        showError('Failed to run inference: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// Initialize chart
function initChart() {
    const ctx = document.getElementById('chart').getContext('2d');
    
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Input',
                    data: [],
                    borderColor: 'rgb(102, 126, 234)',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 2,
                    tension: 0.4
                },
                {
                    label: 'Prediction',
                    data: [],
                    borderColor: 'rgb(56, 239, 125)',
                    backgroundColor: 'rgba(56, 239, 125, 0.1)',
                    borderWidth: 2,
                    tension: 0.4
                },
                {
                    label: 'Ground Truth',
                    data: [],
                    borderColor: 'rgb(255, 107, 107)',
                    backgroundColor: 'rgba(255, 107, 107, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 15,
                        font: {
                            size: 12,
                            weight: 'bold'
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: {
                        size: 14
                    },
                    bodyFont: {
                        size: 13
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Timestep',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Value',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                }
            }
        }
    });
}

// Update chart with new data
function updateChart(data) {
    const seqLen = data.seq_len;
    const predLen = data.pred_len;
    
    // Create labels
    const labels = [];
    for (let i = 0; i < seqLen + predLen; i++) {
        labels.push(i);
    }
    
    // Prepare data arrays
    const inputData = [...data.input, ...Array(predLen).fill(null)];
    const predData = [...Array(seqLen).fill(null), ...data.prediction];
    const gtData = [...Array(seqLen).fill(null), ...data.ground_truth];
    
    // Update chart
    chart.data.labels = labels;
    chart.data.datasets[0].data = inputData;
    chart.data.datasets[1].data = predData;
    chart.data.datasets[2].data = gtData;
    chart.update();
}

// Update metrics display
function updateMetrics(metrics) {
    const metricsDiv = document.getElementById('metrics');
    metricsDiv.innerHTML = `
        <div class="metric-item">
            <span class="metric-label">MAE:</span>
            <span class="metric-value">${metrics.mae.toFixed(4)}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">MSE:</span>
            <span class="metric-value">${metrics.mse.toFixed(4)}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">RMSE:</span>
            <span class="metric-value">${metrics.rmse.toFixed(4)}</span>
        </div>
    `;
}

// Show loading overlay
function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (show) {
        overlay.classList.add('show');
    } else {
        overlay.classList.remove('show');
    }
}

// Show success message
function showSuccess(message) {
    // Simple alert for now, can be replaced with toast notification
    console.log('Success:', message);
}

// Show error message
function showError(message) {
    alert('Error: ' + message);
}

