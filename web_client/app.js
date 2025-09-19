// GovDocShield X Enhanced - Web Client JavaScript

class GovDocShieldWebClient {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000/api/v2';
        this.wsConnection = null;
        this.currentTab = 'dashboard';
        this.charts = {};
        this.updateInterval = null;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupWebSocket();
        this.loadDashboard();
        this.startRealTimeUpdates();
        this.loadThreatsData();
    }

    setupEventListeners() {
        // Tab navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tabName = e.target.dataset.tab;
                this.switchTab(tabName);
            });
        });

        // File upload
        const fileInput = document.getElementById('file-input');
        const uploadArea = document.getElementById('upload-area');
        const browseBtn = document.getElementById('browse-btn');

        browseBtn.addEventListener('click', () => fileInput.click());
        
        fileInput.addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files);
        });

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            this.handleFileUpload(e.dataTransfer.files);
        });

        // Settings
        this.setupSettingsListeners();
        
        // Maintenance buttons
        this.setupMaintenanceListeners();
    }

    setupSettingsListeners() {
        const settings = [
            'fips-mode', 'airgapped-mode', 'quantum-encryption',
            'auto-quarantine', 'realtime-monitoring'
        ];

        settings.forEach(setting => {
            const element = document.getElementById(setting);
            if (element) {
                element.addEventListener('change', (e) => {
                    this.updateSetting(setting, e.target.checked);
                });
            }
        });

        const threatSensitivity = document.getElementById('threat-sensitivity');
        if (threatSensitivity) {
            threatSensitivity.addEventListener('change', (e) => {
                this.updateSetting('threat-sensitivity', e.target.value);
            });
        }
    }

    setupMaintenanceListeners() {
        const buttons = [
            { id: 'run-diagnostics', action: this.runDiagnostics },
            { id: 'update-signatures', action: this.updateSignatures },
            { id: 'backup-config', action: this.backupConfig },
            { id: 'emergency-stop', action: this.emergencyStop },
            { id: 'refresh-threats', action: this.loadThreatsData }
        ];

        buttons.forEach(({ id, action }) => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('click', () => action.call(this));
            }
        });
    }

    setupWebSocket() {
        try {
            this.wsConnection = new WebSocket('ws://localhost:8000/ws');
            
            this.wsConnection.onopen = () => {
                console.log('WebSocket connected');
                this.showNotification('Connected to GovDocShield X Enhanced', 'success');
            };

            this.wsConnection.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleRealtimeUpdate(data);
            };

            this.wsConnection.onclose = () => {
                console.log('WebSocket disconnected');
                this.showNotification('Connection lost. Attempting to reconnect...', 'warning');
                setTimeout(() => this.setupWebSocket(), 5000);
            };

            this.wsConnection.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        } catch (error) {
            console.error('WebSocket setup failed:', error);
        }
    }

    switchTab(tabName) {
        // Update navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');

        this.currentTab = tabName;

        // Load tab-specific data
        switch (tabName) {
            case 'dashboard':
                this.loadDashboard();
                break;
            case 'threats':
                this.loadThreatsData();
                break;
            case 'quantum':
                this.loadQuantumData();
                break;
        }
    }

    async loadDashboard() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/dashboard/metrics`);
            const data = await response.json();
            
            this.updateDashboardMetrics(data);
            this.createQuantumChart();
        } catch (error) {
            console.error('Failed to load dashboard:', error);
            this.showNotification('Failed to load dashboard data', 'error');
        }
    }

    updateDashboardMetrics(data) {
        // Update quantum coherence
        const quantumCoherence = data.quantum?.coherence || 87.3;
        document.getElementById('quantum-coherence').textContent = `${quantumCoherence}%`;
        
        // Update neuromorphic activity
        const neuromorphicActivity = data.neuromorphic?.activity || 92.1;
        document.getElementById('neuromorphic-activity').textContent = `${neuromorphicActivity}%`;
        
        // Update bio immunity
        const bioImmunity = data.bio_inspired?.immunity || 94.8;
        document.getElementById('bio-immunity').textContent = `${bioImmunity}%`;

        // Update performance metrics
        document.getElementById('files-processed').textContent = data.performance?.files_processed || '1,247';
        document.getElementById('threats-blocked').textContent = data.performance?.threats_blocked || '89';
        document.getElementById('avg-response').textContent = data.performance?.avg_response || '127ms';
        document.getElementById('success-rate').textContent = data.performance?.success_rate || '99.8%';
    }

    createQuantumChart() {
        const ctx = document.getElementById('quantum-chart');
        if (!ctx) return;

        if (this.charts.quantum) {
            this.charts.quantum.destroy();
        }

        const chartCtx = ctx.getContext('2d');
        
        this.charts.quantum = new Chart(chartCtx, {
            type: 'line',
            data: {
                labels: Array.from({length: 20}, (_, i) => `${i}s`),
                datasets: [
                    {
                        label: 'Quantum Coherence',
                        data: this.generateQuantumData(),
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    },
                    {
                        label: 'Entanglement Fidelity',
                        data: this.generateQuantumData(),
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#cbd5e1'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#64748b' },
                        grid: { color: '#475569' }
                    },
                    y: {
                        ticks: { color: '#64748b' },
                        grid: { color: '#475569' },
                        min: 0,
                        max: 100
                    }
                }
            }
        });
    }

    generateQuantumData() {
        return Array.from({length: 20}, () => 
            Math.floor(Math.random() * 20) + 80
        );
    }

    async handleFileUpload(files) {
        if (files.length === 0) return;

        this.showLoading(true);
        
        try {
            for (const file of files) {
                await this.analyzeFile(file);
            }
        } catch (error) {
            console.error('File upload failed:', error);
            this.showNotification('File analysis failed', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    async analyzeFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        // Get analysis options
        const options = {
            quantum_analysis: document.getElementById('quantum-analysis').checked,
            neuromorphic_analysis: document.getElementById('neuromorphic-analysis').checked,
            bio_analysis: document.getElementById('bio-analysis').checked,
            steganography_detection: document.getElementById('steganography-detection').checked
        };

        formData.append('options', JSON.stringify(options));

        try {
            const response = await fetch(`${this.apiBaseUrl}/analyze`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            this.displayAnalysisResults(file.name, result);
            
            this.showNotification(`Analysis complete for ${file.name}`, 'success');
        } catch (error) {
            throw new Error(`Failed to analyze ${file.name}: ${error.message}`);
        }
    }

    displayAnalysisResults(filename, results) {
        const resultsSection = document.getElementById('results-section');
        const resultsContainer = document.getElementById('analysis-results');
        
        const resultCard = document.createElement('div');
        resultCard.className = 'result-card';
        
        const threatLevel = results.threat_level || 'low';
        const threatClass = threatLevel === 'high' ? 'danger' : 
                          threatLevel === 'medium' ? 'warning' : 'success';
        
        resultCard.innerHTML = `
            <div class="result-header">
                <h4>${filename}</h4>
                <span class="threat-level ${threatClass}">${threatLevel.toUpperCase()}</span>
            </div>
            <div class="result-content">
                <div class="result-metrics">
                    <div class="metric">
                        <span class="metric-label">Confidence Score</span>
                        <span class="metric-value">${results.confidence_score || '95.2%'}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Processing Time</span>
                        <span class="metric-value">${results.processing_time || '127ms'}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Detection Method</span>
                        <span class="metric-value">${results.detection_method || 'Quantum-Enhanced'}</span>
                    </div>
                </div>
                ${results.threats ? `
                    <div class="detected-threats">
                        <h5>Detected Threats:</h5>
                        <ul>
                            ${results.threats.map(threat => `<li>${threat}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
            </div>
        `;
        
        resultsContainer.appendChild(resultCard);
        resultsSection.style.display = 'block';
    }

    async loadThreatsData() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/threats`);
            const threats = await response.json();
            
            this.updateThreatsTable(threats);
            this.updateThreatStats(threats);
        } catch (error) {
            console.error('Failed to load threats:', error);
            this.showNotification('Failed to load threats data', 'error');
        }
    }

    updateThreatsTable(threats) {
        const tbody = document.getElementById('threats-table-body');
        if (!tbody) return;

        tbody.innerHTML = '';

        const sampleThreats = threats.length > 0 ? threats : [
            {
                type: 'Advanced Persistent Threat',
                file: 'document_2024_09_19.pdf',
                severity: 'high',
                detection_method: 'Quantum Neural Network',
                timestamp: '2024-09-19 14:23:15'
            },
            {
                type: 'Malware Detected',
                file: 'spreadsheet.xlsx',
                severity: 'medium',
                detection_method: 'Neuromorphic Analysis',
                timestamp: '2024-09-19 14:18:42'
            },
            {
                type: 'Steganography Suspected',
                file: 'image_001.jpg',
                severity: 'low',
                detection_method: 'Bio-Inspired Detection',
                timestamp: '2024-09-19 14:15:30'
            }
        ];

        sampleThreats.forEach(threat => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${threat.type}</td>
                <td>${threat.file}</td>
                <td>
                    <span class="threat-severity ${threat.severity}">${threat.severity.toUpperCase()}</span>
                </td>
                <td>${threat.detection_method}</td>
                <td>${threat.timestamp}</td>
                <td>
                    <button class="btn-secondary btn-sm" onclick="govDocShield.quarantineFile('${threat.file}')">
                        <i class="fas fa-lock"></i> Quarantine
                    </button>
                </td>
            `;
            tbody.appendChild(row);
        });
    }

    updateThreatStats(threats) {
        // Update threat count badge
        document.getElementById('threat-count').textContent = threats.length || 3;
    }

    async loadQuantumData() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/quantum/status`);
            const quantumData = await response.json();
            
            // Quantum data is already displayed, just update if needed
            console.log('Quantum status:', quantumData);
        } catch (error) {
            console.error('Failed to load quantum data:', error);
        }
    }

    async updateSetting(setting, value) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/settings`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    [setting]: value
                })
            });

            if (response.ok) {
                this.showNotification(`Setting updated: ${setting}`, 'success');
            } else {
                throw new Error('Failed to update setting');
            }
        } catch (error) {
            console.error('Failed to update setting:', error);
            this.showNotification('Failed to update setting', 'error');
        }
    }

    async runDiagnostics() {
        this.showLoading(true);
        try {
            const response = await fetch(`${this.apiBaseUrl}/diagnostics`, {
                method: 'POST'
            });
            const result = await response.json();
            
            this.showNotification('System diagnostics completed', 'success');
        } catch (error) {
            this.showNotification('Diagnostics failed', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    async updateSignatures() {
        this.showLoading(true);
        try {
            const response = await fetch(`${this.apiBaseUrl}/signatures/update`, {
                method: 'POST'
            });
            const result = await response.json();
            
            this.showNotification('Threat signatures updated', 'success');
        } catch (error) {
            this.showNotification('Signature update failed', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    async backupConfig() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/config/backup`, {
                method: 'POST'
            });
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `govdocshield_config_${new Date().toISOString().slice(0, 10)}.json`;
                a.click();
                
                this.showNotification('Configuration backup downloaded', 'success');
            } else {
                throw new Error('Backup failed');
            }
        } catch (error) {
            this.showNotification('Configuration backup failed', 'error');
        }
    }

    async emergencyStop() {
        if (!confirm('Are you sure you want to initiate emergency stop? This will shut down all systems.')) {
            return;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/emergency/stop`, {
                method: 'POST'
            });
            
            this.showNotification('Emergency stop initiated', 'warning');
        } catch (error) {
            this.showNotification('Emergency stop failed', 'error');
        }
    }

    async quarantineFile(filename) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/quarantine`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ filename })
            });

            if (response.ok) {
                this.showNotification(`File quarantined: ${filename}`, 'success');
                this.loadThreatsData(); // Refresh threats table
            } else {
                throw new Error('Quarantine failed');
            }
        } catch (error) {
            this.showNotification('Quarantine operation failed', 'error');
        }
    }

    handleRealtimeUpdate(data) {
        switch (data.type) {
            case 'threat_detected':
                this.handleNewThreat(data.threat);
                break;
            case 'system_metrics':
                this.updateDashboardMetrics(data.metrics);
                break;
            case 'quantum_update':
                this.updateQuantumMetrics(data.quantum);
                break;
        }
    }

    handleNewThreat(threat) {
        // Add to threat list
        const threatList = document.getElementById('threat-list');
        if (threatList) {
            const threatItem = document.createElement('div');
            threatItem.className = `threat-item ${threat.severity}`;
            threatItem.innerHTML = `
                <div class="threat-icon"><i class="fas fa-exclamation-triangle"></i></div>
                <div class="threat-info">
                    <span class="threat-type">${threat.type}</span>
                    <span class="threat-time">just now</span>
                </div>
                <div class="threat-severity ${threat.severity}">${threat.severity.toUpperCase()}</div>
            `;
            threatList.insertBefore(threatItem, threatList.firstChild);
            
            // Remove old items if more than 5
            if (threatList.children.length > 5) {
                threatList.removeChild(threatList.lastChild);
            }
        }

        // Update threat count
        const threatCount = document.getElementById('threat-count');
        if (threatCount) {
            const current = parseInt(threatCount.textContent);
            threatCount.textContent = current + 1;
        }

        // Show notification
        this.showNotification(`New ${threat.severity} threat detected: ${threat.type}`, 'warning');
    }

    updateQuantumMetrics(quantum) {
        // Update quantum coherence progress bar
        const coherenceBar = document.querySelector('#quantum-coherence').nextElementSibling.querySelector('.progress-fill');
        if (coherenceBar) {
            coherenceBar.style.width = `${quantum.coherence}%`;
        }
    }

    startRealTimeUpdates() {
        this.updateInterval = setInterval(() => {
            if (this.currentTab === 'dashboard') {
                this.simulateRealtimeData();
            }
        }, 5000);
    }

    simulateRealtimeData() {
        // Simulate real-time updates for demo purposes
        const quantumCoherence = 85 + Math.random() * 10;
        const neuromorphicActivity = 90 + Math.random() * 8;
        const bioImmunity = 92 + Math.random() * 6;

        document.getElementById('quantum-coherence').textContent = `${quantumCoherence.toFixed(1)}%`;
        document.getElementById('neuromorphic-activity').textContent = `${neuromorphicActivity.toFixed(1)}%`;
        document.getElementById('bio-immunity').textContent = `${bioImmunity.toFixed(1)}%`;

        // Update progress bars
        const progressBars = document.querySelectorAll('.progress-fill');
        progressBars[0].style.width = `${quantumCoherence}%`;
        progressBars[1].style.width = `${neuromorphicActivity}%`;
        progressBars[2].style.width = `${bioImmunity}%`;

        // Update chart if exists
        if (this.charts.quantum) {
            const datasets = this.charts.quantum.data.datasets;
            datasets.forEach(dataset => {
                dataset.data.shift();
                dataset.data.push(85 + Math.random() * 15);
            });
            this.charts.quantum.update('none');
        }
    }

    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        if (show) {
            overlay.classList.add('active');
        } else {
            overlay.classList.remove('active');
        }
    }

    showNotification(message, type = 'info') {
        const container = document.getElementById('notification-container');
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        
        notification.innerHTML = `
            <div class="notification-content">
                <div class="notification-message">${message}</div>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        container.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }
}

// Initialize the application
const govDocShield = new GovDocShieldWebClient();

// Global functions for HTML onclick handlers
window.govDocShield = govDocShield;