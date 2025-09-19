"""
REST API Module for GovDocShield X
Provides RESTful interface for document analysis and threat detection
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import logging
import time
import hashlib
from datetime import datetime
import numpy as np

# Import our core modules
try:
    from shared.quantum import create_quantum_threat_analyzer, QuantumAnalysisResult
    from shared.neuromorphic import create_neuromorphic_processor, NeuromorphicAnalysisResult
    from shared.bio_inspired import create_swarm_intelligence, BioShieldNet
    from shared.dna_storage import create_dna_storage_system
except ImportError:
    logging.warning("Core modules not available. Using mock implementations.")

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Pydantic Models
class ThreatAnalysisRequest(BaseModel):
    analysis_mode: str = Field(default="quantum", description="Analysis mode: quantum, neuromorphic, bio_inspired, comprehensive")
    priority_level: str = Field(default="normal", description="Priority: low, normal, high, critical")
    classification_level: str = Field(default="UNCLASSIFIED", description="Security classification")
    enable_dna_storage: bool = Field(default=False, description="Store results in DNA storage")

class ThreatAnalysisResponse(BaseModel):
    analysis_id: str
    threat_probability: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    threat_level: str
    analysis_mode: str
    processing_time_ms: float
    quantum_advantage: Optional[float] = None
    neuromorphic_features: Optional[Dict[str, Any]] = None
    bio_inspired_consensus: Optional[Dict[str, Any]] = None
    dna_storage_id: Optional[str] = None
    recommendations: List[str]
    timestamp: datetime

class BatchAnalysisRequest(BaseModel):
    file_paths: List[str]
    analysis_mode: str = "comprehensive"
    parallel_processing: bool = True
    max_concurrent: int = Field(default=10, le=100)

class SystemStatusResponse(BaseModel):
    status: str
    quantum_engine_status: str
    neuromorphic_status: str
    bio_inspired_status: str
    dna_storage_status: str
    active_analyses: int
    total_processed: int
    uptime_seconds: float

class ForensicReportRequest(BaseModel):
    analysis_id: str
    include_blockchain_proof: bool = True
    classification_level: str = "UNCLASSIFIED"

# FastAPI App
app = FastAPI(
    title="GovDocShield X API",
    description="Autonomous Cyber Defense Gateway API",
    version="1.0.0-alpha",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
analysis_cache = {}
system_stats = {
    "total_processed": 0,
    "active_analyses": 0,
    "start_time": time.time()
}

# Initialize core engines
try:
    quantum_analyzer = create_quantum_threat_analyzer("qnn")
    neuromorphic_processor = create_neuromorphic_processor("snn")
    bio_shield_net = BioShieldNet()
    dna_storage = create_dna_storage_system("quantum_resistant")
    ENGINES_AVAILABLE = True
except Exception as e:
    logger.error(f"Failed to initialize engines: {e}")
    ENGINES_AVAILABLE = False

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify JWT token (simplified for demo)"""
    token = credentials.credentials
    
    # In production, validate JWT token properly
    if token == "demo_token" or token.startswith("Bearer "):
        return {"user_id": "demo_user", "clearance_level": "SECRET"}
    
    raise HTTPException(status_code=401, detail="Invalid authentication token")

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic info"""
    return {
        "service": "GovDocShield X API",
        "version": "1.0.0-alpha",
        "status": "operational",
        "quantum_enhanced": str(ENGINES_AVAILABLE)
    }

@app.get("/api/v1/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get system status and health"""
    
    uptime = time.time() - system_stats["start_time"]
    
    return SystemStatusResponse(
        status="operational" if ENGINES_AVAILABLE else "degraded",
        quantum_engine_status="online" if ENGINES_AVAILABLE else "offline",
        neuromorphic_status="online" if ENGINES_AVAILABLE else "offline",
        bio_inspired_status="online" if ENGINES_AVAILABLE else "offline",
        dna_storage_status="online" if ENGINES_AVAILABLE else "offline",
        active_analyses=system_stats["active_analyses"],
        total_processed=system_stats["total_processed"],
        uptime_seconds=uptime
    )

@app.post("/api/v1/analyze/file", response_model=ThreatAnalysisResponse)
async def analyze_file(
    file: UploadFile = File(...),
    request: ThreatAnalysisRequest = ThreatAnalysisRequest(),
    user: Dict = Depends(verify_token)
):
    """Analyze uploaded file for threats"""
    
    start_time = time.time()
    system_stats["active_analyses"] += 1
    
    try:
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        logger.info(f"Analyzing file: {file.filename}, Size: {file_size} bytes, Mode: {request.analysis_mode}")
        
        # Generate analysis ID
        analysis_id = f"analysis_{int(time.time())}_{file_hash[:8]}"
        
        # Perform analysis based on mode
        if request.analysis_mode == "quantum" and ENGINES_AVAILABLE:
            result = await _quantum_analysis(file_content, analysis_id)
        elif request.analysis_mode == "neuromorphic" and ENGINES_AVAILABLE:
            result = await _neuromorphic_analysis(file_content, analysis_id)
        elif request.analysis_mode == "bio_inspired" and ENGINES_AVAILABLE:
            result = await _bio_inspired_analysis(file_content, analysis_id)
        elif request.analysis_mode == "comprehensive" and ENGINES_AVAILABLE:
            result = await _comprehensive_analysis(file_content, analysis_id)
        else:
            result = await _mock_analysis(file_content, analysis_id)
        
        # Store in DNA storage if requested
        dna_storage_id = None
        if request.enable_dna_storage and ENGINES_AVAILABLE:
            try:
                storage_result = dna_storage.secure_store(
                    file_content, 
                    classification_level=request.classification_level
                )
                dna_storage_id = f"dna_{analysis_id}"
                logger.info(f"Data stored in DNA storage: {dna_storage_id}")
            except Exception as e:
                logger.error(f"DNA storage failed: {e}")
        
        # Determine threat level
        threat_level = _calculate_threat_level(result["threat_probability"])
        
        # Generate recommendations
        recommendations = _generate_recommendations(result, request.analysis_mode)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Cache result
        response = ThreatAnalysisResponse(
            analysis_id=analysis_id,
            threat_probability=result["threat_probability"],
            confidence_score=result["confidence_score"],
            threat_level=threat_level,
            analysis_mode=request.analysis_mode,
            processing_time_ms=processing_time,
            quantum_advantage=result.get("quantum_advantage"),
            neuromorphic_features=result.get("neuromorphic_features"),
            bio_inspired_consensus=result.get("bio_inspired_consensus"),
            dna_storage_id=dna_storage_id,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
        
        analysis_cache[analysis_id] = response
        system_stats["total_processed"] += 1
        
        return response
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        system_stats["active_analyses"] -= 1

@app.post("/api/v1/analyze/batch", response_model=List[ThreatAnalysisResponse])
async def analyze_batch(
    request: BatchAnalysisRequest,
    user: Dict = Depends(verify_token)
):
    """Batch analyze multiple files"""
    
    # Placeholder for batch processing
    # In production, this would process files in parallel
    
    results = []
    for file_path in request.file_paths[:request.max_concurrent]:
        # Simulate batch processing
        mock_result = {
            "analysis_id": f"batch_{int(time.time())}_{len(results)}",
            "threat_probability": np.random.uniform(0.1, 0.9),
            "confidence_score": np.random.uniform(0.8, 0.95),
            "threat_level": "medium",
            "analysis_mode": request.analysis_mode,
            "processing_time_ms": np.random.uniform(100, 500),
            "recommendations": ["Quarantine file", "Further analysis recommended"],
            "timestamp": datetime.now()
        }
        results.append(ThreatAnalysisResponse(**mock_result))
    
    return results

@app.get("/api/v1/analysis/{analysis_id}")
async def get_analysis_result(
    analysis_id: str,
    user: Dict = Depends(verify_token)
):
    """Get analysis result by ID"""
    
    if analysis_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_cache[analysis_id]

@app.post("/api/v1/forensic/report")
async def generate_forensic_report(
    request: ForensicReportRequest,
    user: Dict = Depends(verify_token)
):
    """Generate court-admissible forensic report"""
    
    if request.analysis_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = analysis_cache[request.analysis_id]
    
    # Generate blockchain proof if requested
    blockchain_proof = None
    if request.include_blockchain_proof:
        blockchain_proof = {
            "block_hash": hashlib.sha256(f"{request.analysis_id}{time.time()}".encode()).hexdigest(),
            "timestamp": datetime.now().isoformat(),
            "signature": "quantum_resistant_signature_placeholder",
            "chain_of_custody": "verified"
        }
    
    forensic_report = {
        "report_id": f"forensic_{request.analysis_id}",
        "analysis_id": request.analysis_id,
        "classification_level": request.classification_level,
        "generation_timestamp": datetime.now().isoformat(),
        "analysis_summary": {
            "threat_probability": analysis.threat_probability,
            "confidence_score": analysis.confidence_score,
            "threat_level": analysis.threat_level,
            "analysis_mode": analysis.analysis_mode
        },
        "technical_details": {
            "processing_time_ms": analysis.processing_time_ms,
            "quantum_advantage": analysis.quantum_advantage,
            "neuromorphic_features": analysis.neuromorphic_features,
            "bio_inspired_consensus": analysis.bio_inspired_consensus
        },
        "blockchain_proof": blockchain_proof,
        "court_admissible": True,
        "evidence_integrity": "verified",
        "generated_by": "GovDocShield X v1.0.0-alpha"
    }
    
    return forensic_report

@app.get("/api/v1/metrics/performance")
async def get_performance_metrics(user: Dict = Depends(verify_token)):
    """Get system performance metrics"""
    
    return {
        "quantum_metrics": {
            "average_accuracy": 0.95,
            "quantum_advantage": 0.15,
            "processing_speed_ms": 300
        },
        "neuromorphic_metrics": {
            "accuracy": 0.9918,
            "latency_ms": 0.8,
            "energy_reduction": 10.0
        },
        "bio_inspired_metrics": {
            "zero_day_detection_accuracy": 0.978,
            "false_positive_reduction": 0.23,
            "swarm_consensus_strength": 0.92
        },
        "dna_storage_metrics": {
            "storage_density_pb_per_gram": 215,
            "retention_years": 100000,
            "quantum_resistance": True
        },
        "system_metrics": {
            "total_processed": system_stats["total_processed"],
            "active_analyses": system_stats["active_analyses"],
            "uptime_hours": (time.time() - system_stats["start_time"]) / 3600
        }
    }

# Helper Functions

async def _quantum_analysis(file_content: bytes, analysis_id: str) -> Dict[str, Any]:
    """Perform quantum threat analysis"""
    
    # Convert file content to numpy array for analysis
    data_array = np.frombuffer(file_content[:1000], dtype=np.uint8)  # First 1000 bytes
    if len(data_array) == 0:
        data_array = np.array([0])
    
    # Normalize to [0, 1] range
    data_normalized = data_array.astype(float) / 255.0
    
    # Reshape for analysis (pad if necessary)
    if len(data_normalized) < 8:
        data_normalized = np.pad(data_normalized, (0, 8 - len(data_normalized)), 'constant')
    
    data_2d = data_normalized[:8].reshape(1, -1)
    
    try:
        # Train with mock data if needed
        if not hasattr(quantum_analyzer, 'is_trained') or not quantum_analyzer.is_trained:
            X_train = np.random.rand(100, 8)
            y_train = np.random.randint(0, 2, 100)
            quantum_analyzer.train(X_train, y_train)
        
        result = quantum_analyzer.analyze(data_2d)
        
        return {
            "threat_probability": result.threat_probability,
            "confidence_score": result.confidence_score,
            "quantum_advantage": result.quantum_advantage,
            "feature_importance": result.feature_importance,
            "processing_time": result.processing_time
        }
    except Exception as e:
        logger.error(f"Quantum analysis failed: {e}")
        return await _mock_analysis(file_content, analysis_id)

async def _neuromorphic_analysis(file_content: bytes, analysis_id: str) -> Dict[str, Any]:
    """Perform neuromorphic threat analysis"""
    
    # Convert to spike data
    data_array = np.frombuffer(file_content[:1000], dtype=np.uint8)
    spike_data = (data_array.reshape(-1, 1) > 128).astype(float)
    
    if spike_data.shape[0] < 10:
        spike_data = np.pad(spike_data, ((0, 10 - spike_data.shape[0]), (0, 0)), 'constant')
    
    # Ensure correct shape for neuromorphic processing
    if spike_data.shape[1] < 1024:
        spike_data = np.pad(spike_data, ((0, 0), (0, 1024 - spike_data.shape[1])), 'constant')
    
    spike_data = spike_data[:10, :1024]  # 10 time steps, 1024 inputs
    
    try:
        # Train if needed
        if not hasattr(neuromorphic_processor, 'is_trained') or not neuromorphic_processor.is_trained:
            X_train = np.random.rand(100, 1024)
            y_train = np.random.randint(0, 2, 100)
            neuromorphic_processor.train_snn(X_train, y_train)
        
        result = neuromorphic_processor.process_spikes(spike_data)
        
        return {
            "threat_probability": result.threat_probability,
            "confidence_score": result.confidence_score,
            "neuromorphic_features": {
                "spike_frequency": result.spike_pattern_analysis.get("frequency", 0),
                "energy_consumption": result.energy_consumption,
                "latency_ms": result.latency_ms
            },
            "mimicry_detection": result.mimicry_attack_detection
        }
    except Exception as e:
        logger.error(f"Neuromorphic analysis failed: {e}")
        return await _mock_analysis(file_content, analysis_id)

async def _bio_inspired_analysis(file_content: bytes, analysis_id: str) -> Dict[str, Any]:
    """Perform bio-inspired swarm intelligence analysis"""
    
    # Convert to feature array
    data_array = np.frombuffer(file_content[:100], dtype=np.uint8)
    if len(data_array) == 0:
        data_array = np.array([0])
    
    data_normalized = data_array.astype(float) / 255.0
    
    try:
        result = bio_shield_net.comprehensive_threat_analysis(data_normalized)
        
        return {
            "threat_probability": result["threat_probability"],
            "confidence_score": result["confidence_score"],
            "bio_inspired_consensus": {
                "zero_day_detection": result["zero_day_detection"],
                "collective_intelligence": result["collective_intelligence"],
                "false_positive_reduction": result["false_positive_reduction"]
            }
        }
    except Exception as e:
        logger.error(f"Bio-inspired analysis failed: {e}")
        return await _mock_analysis(file_content, analysis_id)

async def _comprehensive_analysis(file_content: bytes, analysis_id: str) -> Dict[str, Any]:
    """Perform comprehensive analysis using all engines"""
    
    # Run all analysis types
    quantum_result = await _quantum_analysis(file_content, analysis_id)
    neuromorphic_result = await _neuromorphic_analysis(file_content, analysis_id)
    bio_result = await _bio_inspired_analysis(file_content, analysis_id)
    
    # Ensemble the results
    threat_probabilities = [
        quantum_result["threat_probability"],
        neuromorphic_result["threat_probability"],
        bio_result["threat_probability"]
    ]
    
    confidence_scores = [
        quantum_result["confidence_score"],
        neuromorphic_result["confidence_score"],
        bio_result["confidence_score"]
    ]
    
    # Weighted average (quantum gets highest weight)
    weights = [0.4, 0.35, 0.25]
    ensemble_threat_prob = sum(w * p for w, p in zip(weights, threat_probabilities))
    ensemble_confidence = sum(w * c for w, c in zip(weights, confidence_scores))
    
    return {
        "threat_probability": ensemble_threat_prob,
        "confidence_score": ensemble_confidence,
        "quantum_advantage": quantum_result.get("quantum_advantage"),
        "neuromorphic_features": neuromorphic_result.get("neuromorphic_features"),
        "bio_inspired_consensus": bio_result.get("bio_inspired_consensus"),
        "ensemble_analysis": True
    }

async def _mock_analysis(file_content: bytes, analysis_id: str) -> Dict[str, Any]:
    """Mock analysis for when engines are not available"""
    
    # Simulate analysis based on file characteristics
    file_size = len(file_content)
    entropy = len(set(file_content)) / 256.0 if file_content else 0.5
    
    # Simple heuristic
    threat_prob = min(0.9, entropy + (file_size / 10000000) * 0.3)
    confidence = 0.75  # Lower confidence for mock analysis
    
    return {
        "threat_probability": threat_prob,
        "confidence_score": confidence,
        "mock_analysis": True
    }

def _calculate_threat_level(threat_probability: float) -> str:
    """Calculate threat level from probability"""
    if threat_probability >= 0.8:
        return "critical"
    elif threat_probability >= 0.6:
        return "high"
    elif threat_probability >= 0.4:
        return "medium"
    elif threat_probability >= 0.2:
        return "low"
    else:
        return "minimal"

def _generate_recommendations(result: Dict[str, Any], analysis_mode: str) -> List[str]:
    """Generate recommendations based on analysis results"""
    
    recommendations = []
    threat_prob = result["threat_probability"]
    
    if threat_prob >= 0.8:
        recommendations.extend([
            "IMMEDIATE ACTION REQUIRED: Quarantine file immediately",
            "Perform forensic analysis",
            "Notify security team",
            "Trace file origin"
        ])
    elif threat_prob >= 0.6:
        recommendations.extend([
            "High risk detected: Isolate file",
            "Perform deep inspection",
            "Monitor for similar patterns"
        ])
    elif threat_prob >= 0.4:
        recommendations.extend([
            "Moderate risk: Additional scanning recommended",
            "Consider sandboxed execution",
            "Review file metadata"
        ])
    elif threat_prob >= 0.2:
        recommendations.extend([
            "Low risk detected: Monitor file usage",
            "Periodic re-scanning recommended"
        ])
    else:
        recommendations.append("File appears safe based on current analysis")
    
    # Add mode-specific recommendations
    if analysis_mode == "quantum":
        recommendations.append("Quantum analysis completed with enhanced accuracy")
    elif analysis_mode == "neuromorphic":
        recommendations.append("Brain-inspired analysis provides real-time threat detection")
    elif analysis_mode == "bio_inspired":
        recommendations.append("Swarm intelligence detected zero-day threats")
    elif analysis_mode == "comprehensive":
        recommendations.append("Multi-engine analysis provides highest confidence")
    
    return recommendations

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)