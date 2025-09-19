# 🛡️ GovDocShield X - Quick Start Guide

## Overview
GovDocShield X is a revolutionary quantum-neuromorphic autonomous cyber defense gateway designed for government and defense organizations. This guide will help you get started quickly with the platform.

## 🚀 Quick Installation

### Option 1: PowerShell Setup (Windows)
```powershell
# Clone the repository
git clone https://github.com/gov/govdocshield-x.git
cd govdocshield-x

# Run the automated setup
.\install.ps1

# Start the dashboard
.\start-dashboard.bat
```

### Option 2: Docker Deployment
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f govdocshield-api
```

### Option 3: Kubernetes Production
```bash
# Deploy to production cluster
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n govdocshield

# Access the dashboard
kubectl port-forward svc/govdocshield-dashboard-service 8501:8501 -n govdocshield
```

## 💻 Using the CLI

### Basic File Analysis
```bash
# Analyze a single file
python govdocshield.py analyze suspicious_document.pdf

# Comprehensive analysis with all engines
python govdocshield.py analyze document.docx --mode comprehensive

# High priority analysis
python govdocshield.py analyze malware.exe --priority critical

# Save detailed forensic report
python govdocshield.py analyze file.zip --save-report forensic_report.json
```

### Batch Processing
```bash
# Analyze all PDF files in a directory
python govdocshield.py batch ./documents --pattern "*.pdf"

# Process with parallel analysis
python govdocshield.py batch ./files --parallel --max-files 50

# Generate batch report
python govdocshield.py batch ./uploads --output-report batch_analysis.json
```

### System Management
```bash
# Check system status
python govdocshield.py status

# View performance metrics
python govdocshield.py metrics --include-quantum --include-neuromorphic

# Initialize system
python govdocshield.py init

# Generate forensic report
python govdocshield.py forensic ANALYSIS_ID_123 --blockchain-proof
```

## 🌐 Web Dashboard Usage

### Accessing the Dashboard
1. **Local Development**: http://localhost:8501
2. **Docker**: http://localhost:8501
3. **Production**: https://dashboard.govdocshield.defense.gov

### Dashboard Features

#### Single File Analysis
1. Navigate to "Single File" tab
2. Upload file using the file uploader
3. Configure analysis settings:
   - Enable desired engines (Quantum, Neuromorphic, Bio-Inspired)
   - Set priority level
   - Configure DNA storage if needed
4. Click "🔍 Analyze File"
5. View real-time analysis progress
6. Review comprehensive results with threat level and recommendations

#### Batch Analysis
1. Go to "Batch Analysis" tab
2. Configure batch settings (max files, patterns, parallel processing)
3. Click "📁 Simulate Batch Analysis" or upload multiple files
4. Monitor processing progress
5. Review batch summary with statistics and high-risk file identification

#### Real-time Monitoring
1. Switch to "Real-time Monitor" tab
2. View live threat detection timeline
3. Monitor engine performance metrics
4. Check system resource utilization
5. Observe quantum, neuromorphic, and bio-inspired engine status

#### Forensic Reports
1. Access "Forensic Reports" tab
2. View available analysis reports
3. Generate court-admissible forensic reports
4. Include blockchain proof for evidence integrity
5. Export in multiple formats (PDF, JSON, XML, DOCX)

## 🔧 API Usage

### REST API Endpoints

#### File Analysis
```bash
# Single file analysis
curl -X POST "http://localhost:8080/api/v1/analyze/file" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf" \
  -F "analysis_mode=comprehensive" \
  -F "priority_level=high"

# Batch analysis
curl -X POST "http://localhost:8080/api/v1/analyze/batch" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": ["/path/to/file1.pdf", "/path/to/file2.docx"],
    "analysis_mode": "comprehensive",
    "priority_level": "normal"
  }'
```

#### System Status
```bash
# Health check
curl http://localhost:8080/health

# System metrics
curl http://localhost:8080/metrics

# Performance statistics
curl http://localhost:8080/api/v1/system/performance
```

### gRPC API Usage

#### Python Client Example
```python
import grpc
from api.proto import defense_pb2_grpc, defense_pb2

# Create gRPC channel
channel = grpc.insecure_channel('localhost:50051')
client = defense_pb2_grpc.DefenseServiceStub(channel)

# Analyze file
request = defense_pb2.AnalyzeRequest(
    file_path='/path/to/file.pdf',
    mode=defense_pb2.AnalysisMode.COMPREHENSIVE
)

response = client.AnalyzeFile(request)
print(f"Threat probability: {response.threat_probability}")
```

## 🔐 Security Configuration

### Authentication Setup
```bash
# Set API token
export GOVDOCSHIELD_TOKEN="your-secure-token"

# Configure API endpoint
export GOVDOCSHIELD_ENDPOINT="https://api.govdocshield.defense.gov"
```

### Classification Levels
- **UNCLASSIFIED**: Default level for routine operations
- **CONFIDENTIAL**: Sensitive but unclassified information
- **SECRET**: National security information
- **TOP SECRET**: Highest classification level

### Security Features
1. **Quantum-Resistant Cryptography**: All communications protected
2. **Zero-Trust Architecture**: Verify everything, trust nothing
3. **Byzantine Fault Tolerance**: Resilient to node compromise
4. **Forensic Blockchain**: Tamper-proof audit logs
5. **Air-Gap Support**: Offline deployment capability

## 📊 Monitoring and Alerting

### Prometheus Metrics
Access metrics at: http://localhost:9090

Key metrics to monitor:
- `govdocshield:api_request_rate` - Request volume
- `govdocshield:api_error_rate` - Error frequency
- `govdocshield:quantum_accuracy` - Quantum engine performance
- `govdocshield:threat_detection_rate` - Threat identification rate

### Grafana Dashboards
Access dashboards at: http://localhost:3000
- Username: admin
- Password: govdocshield_admin

### Log Analysis
```bash
# View API logs
docker-compose logs govdocshield-api

# Follow real-time logs
kubectl logs -f deployment/govdocshield-api -n govdocshield

# Search for specific events
grep "THREAT_DETECTED" /var/log/govdocshield/api.log
```

## 🛠️ Troubleshooting

### Common Issues

#### "Modules not found" Error
```bash
# Ensure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

#### Docker Service Won't Start
```bash
# Check Docker status
docker-compose ps

# Restart services
docker-compose restart

# View detailed logs
docker-compose logs --tail=50 govdocshield-api
```

#### Kubernetes Pod Failures
```bash
# Check pod status
kubectl get pods -n govdocshield

# Describe failing pod
kubectl describe pod POD_NAME -n govdocshield

# View pod logs
kubectl logs POD_NAME -n govdocshield
```

### Performance Optimization

#### CPU-Intensive Workloads
- Increase CPU limits in Kubernetes deployments
- Enable parallel processing for batch operations
- Configure horizontal pod autoscaling

#### Memory Usage
- Monitor memory consumption with Prometheus
- Adjust memory limits based on file sizes
- Enable DNA storage for large file archival

#### Network Latency
- Use gRPC for high-performance communications
- Deploy closer to data sources
- Enable compression for large file transfers

## 🎯 Use Cases

### Government Document Screening
1. Upload classified documents for analysis
2. Enable comprehensive scanning mode
3. Review threat assessment and recommendations
4. Generate forensic reports for audit trails

### Zero-Day Threat Detection
1. Configure bio-inspired algorithms for unknown threats
2. Enable quantum-enhanced pattern recognition
3. Monitor for steganographic content
4. Implement automatic quarantine procedures

### Incident Response
1. Analyze suspicious files in isolated environment
2. Generate detailed forensic evidence
3. Trace attack vectors with neuromorphic processing
4. Deploy counter-exploitation measures

### Inter-Agency Collaboration
1. Enable federated learning across agencies
2. Share threat intelligence without exposing data
3. Maintain classification boundaries
4. Ensure audit compliance

## 📞 Support and Resources

### Documentation
- **API Reference**: `/docs/api/`
- **Security Features**: `/docs/SECURITY_FEATURES.md`
- **Architecture Guide**: `/docs/ARCHITECTURE.md`
- **Deployment Guide**: `/docs/DEPLOYMENT.md`

### Community
- **GitHub Issues**: Report bugs and feature requests
- **Security Advisories**: Subscribe to security updates
- **Training Materials**: Access learning resources

### Government Support
- **Classified Support**: Contact through secure channels
- **Agency Integration**: Dedicated integration assistance
- **Compliance Guidance**: Federal compliance consulting

---

**🛡️ GovDocShield X - Defending the Digital Frontier with Quantum-Neuromorphic Intelligence**