"""
Federated Defense Grid (Gov-Exclusive)
Secure AI model sharing across agencies with privacy-preserving collaborative learning.
Enables government agencies to share threat intelligence while protecting sensitive data.
"""

import os
import json
import time
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import threading
import asyncio
import socket
from concurrent.futures import ThreadPoolExecutor
import sqlite3

logger = logging.getLogger(__name__)

class SecurityClearance(Enum):
    UNCLASSIFIED = "unclassified"
    CONFIDENTIAL = "confidential" 
    SECRET = "secret"
    TOP_SECRET = "top_secret"
    COMPARTMENTED = "compartmented"

class AgencyType(Enum):
    DEFENSE = "defense"
    INTELLIGENCE = "intelligence"
    LAW_ENFORCEMENT = "law_enforcement"
    CYBERSECURITY = "cybersecurity"
    HOMELAND_SECURITY = "homeland_security"
    FOREIGN_AFFAIRS = "foreign_affairs"
    CRITICAL_INFRASTRUCTURE = "critical_infrastructure"

class ModelType(Enum):
    THREAT_DETECTION = "threat_detection"
    MALWARE_CLASSIFICATION = "malware_classification"
    NETWORK_ANOMALY = "network_anomaly"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"
    ATTRIBUTION_ANALYSIS = "attribution_analysis"

@dataclass
class FederatedNode:
    """Represents a node in the federated defense grid"""
    node_id: str
    agency_name: str
    agency_type: AgencyType
    security_clearance: SecurityClearance
    endpoint_url: str
    public_key: bytes
    capabilities: List[str]
    trust_score: float
    last_active: datetime
    data_contribution_count: int
    model_accuracy_score: float
    compliance_status: str

@dataclass
class ModelUpdate:
    """Represents a federated model update"""
    update_id: str
    source_node: str
    model_type: ModelType
    model_weights: bytes  # Encrypted
    metadata: Dict[str, Any]
    timestamp: datetime
    signature: bytes
    contribution_quality: float
    privacy_metrics: Dict[str, float]

@dataclass
class ThreatIntelligence:
    """Shared threat intelligence data"""
    intel_id: str
    source_agency: str
    classification: SecurityClearance
    threat_type: str
    indicators: List[str]
    attribution: Optional[str]
    confidence_score: float
    created_at: datetime
    expires_at: Optional[datetime]
    sharing_restrictions: List[str]

class DifferentialPrivacy:
    """Implements differential privacy for federated learning"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta     # Failure probability
        self.noise_scale = self._calculate_noise_scale()
    
    def _calculate_noise_scale(self) -> float:
        """Calculate Gaussian noise scale for differential privacy"""
        # Simplified calculation - in production use proper DP-SGD
        return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def add_noise_to_gradients(self, gradients: torch.Tensor) -> torch.Tensor:
        """Add calibrated noise to gradients"""
        noise = torch.normal(0, self.noise_scale, gradients.shape)
        return gradients + noise
    
    def clip_gradients(self, gradients: torch.Tensor, clip_norm: float = 1.0) -> torch.Tensor:
        """Clip gradients to bound sensitivity"""
        grad_norm = torch.norm(gradients)
        if grad_norm > clip_norm:
            gradients = gradients * (clip_norm / grad_norm)
        return gradients
    
    def privatize_model_update(self, model_weights: torch.Tensor) -> torch.Tensor:
        """Apply differential privacy to model weights"""
        # Clip weights
        clipped_weights = self.clip_gradients(model_weights)
        
        # Add noise
        private_weights = self.add_noise_to_gradients(clipped_weights)
        
        return private_weights

class SecureAggregation:
    """Implements secure aggregation for federated learning"""
    
    def __init__(self):
        self.aggregation_threshold = 3  # Minimum nodes for aggregation
        self.byzantine_tolerance = 1    # Maximum byzantine nodes
    
    def aggregate_model_updates(self, updates: List[ModelUpdate]) -> torch.Tensor:
        """Securely aggregate model updates from multiple nodes"""
        
        if len(updates) < self.aggregation_threshold:
            raise ValueError(f"Insufficient updates for aggregation: {len(updates)} < {self.aggregation_threshold}")
        
        # Decrypt and validate updates
        decrypted_weights = []
        for update in updates:
            try:
                weights = self._decrypt_model_weights(update.model_weights)
                if self._validate_update(update, weights):
                    decrypted_weights.append(weights)
            except Exception as e:
                logger.warning(f"Failed to process update from {update.source_node}: {e}")
        
        if len(decrypted_weights) == 0:
            raise ValueError("No valid updates to aggregate")
        
        # Byzantine-robust aggregation using coordinate-wise median
        aggregated_weights = self._byzantine_robust_aggregation(decrypted_weights)
        
        return aggregated_weights
    
    def _decrypt_model_weights(self, encrypted_weights: bytes) -> torch.Tensor:
        """Decrypt model weights (placeholder implementation)"""
        # In production, implement proper decryption
        mock_weights = torch.randn(100, 10)  # Mock model weights
        return mock_weights
    
    def _validate_update(self, update: ModelUpdate, weights: torch.Tensor) -> bool:
        """Validate model update integrity and quality"""
        
        # Check contribution quality threshold
        if update.contribution_quality < 0.5:
            return False
        
        # Check weight bounds (prevent adversarial updates)
        if torch.any(torch.abs(weights) > 10.0):
            return False
        
        # Validate signature (placeholder)
        # In production, implement proper signature verification
        
        return True
    
    def _byzantine_robust_aggregation(self, weight_tensors: List[torch.Tensor]) -> torch.Tensor:
        """Perform Byzantine-robust aggregation"""
        
        if len(weight_tensors) == 1:
            return weight_tensors[0]
        
        # Stack tensors for coordinate-wise operations
        stacked_weights = torch.stack(weight_tensors)
        
        # Use coordinate-wise median for Byzantine fault tolerance
        aggregated_weights = torch.median(stacked_weights, dim=0)[0]
        
        return aggregated_weights

class CryptographicProtocols:
    """Handles cryptographic operations for federated learning"""
    
    def __init__(self):
        self.key_size = 2048
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size
        )
        self.public_key = self.private_key.public_key()
    
    def encrypt_model_weights(self, weights: torch.Tensor, recipient_public_key: bytes) -> bytes:
        """Encrypt model weights for secure transmission"""
        
        # Serialize weights
        weights_bytes = weights.numpy().tobytes()
        
        # Generate symmetric key for large data
        symmetric_key = os.urandom(32)  # AES-256 key
        
        # Encrypt weights with symmetric key
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(symmetric_key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        
        # Pad data to block size
        padded_data = self._pad_data(weights_bytes)
        encrypted_weights = encryptor.update(padded_data) + encryptor.finalize()
        
        # Encrypt symmetric key with recipient's public key
        recipient_key = serialization.load_pem_public_key(recipient_public_key)
        encrypted_symmetric_key = recipient_key.encrypt(
            symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Combine encrypted key and data
        encrypted_package = {
            "encrypted_key": encrypted_symmetric_key,
            "iv": iv,
            "encrypted_data": encrypted_weights
        }
        
        return json.dumps({
            k: v.hex() if isinstance(v, bytes) else v 
            for k, v in encrypted_package.items()
        }).encode()
    
    def decrypt_model_weights(self, encrypted_data: bytes) -> torch.Tensor:
        """Decrypt model weights"""
        
        encrypted_package = json.loads(encrypted_data.decode())
        
        # Extract components
        encrypted_key = bytes.fromhex(encrypted_package["encrypted_key"])
        iv = bytes.fromhex(encrypted_package["iv"])
        encrypted_weights = bytes.fromhex(encrypted_package["encrypted_data"])
        
        # Decrypt symmetric key
        symmetric_key = self.private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Decrypt weights
        cipher = Cipher(algorithms.AES(symmetric_key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        decrypted_data = decryptor.update(encrypted_weights) + decryptor.finalize()
        
        # Remove padding and reconstruct tensor
        unpadded_data = self._unpad_data(decrypted_data)
        weights_array = np.frombuffer(unpadded_data, dtype=np.float32)
        
        # Reshape to original tensor shape (this would be stored in metadata)
        weights_tensor = torch.from_numpy(weights_array).reshape(100, 10)  # Mock shape
        
        return weights_tensor
    
    def sign_model_update(self, update_data: bytes) -> bytes:
        """Create digital signature for model update"""
        
        signature = self.private_key.sign(
            update_data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return signature
    
    def verify_signature(self, data: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify digital signature"""
        
        try:
            pub_key = serialization.load_pem_public_key(public_key)
            pub_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    def _pad_data(self, data: bytes) -> bytes:
        """PKCS7 padding for AES"""
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length] * padding_length)
        return data + padding
    
    def _unpad_data(self, padded_data: bytes) -> bytes:
        """Remove PKCS7 padding"""
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]

class FederatedThreatModel(nn.Module):
    """Neural network model for federated threat detection"""
    
    def __init__(self, input_size: int = 1000, hidden_size: int = 256, num_classes: int = 10):
        super(FederatedThreatModel, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(hidden_size // 4, num_classes)
        
        self.threat_type_classifier = nn.Linear(hidden_size // 4, 5)  # 5 threat types
        self.severity_regressor = nn.Linear(hidden_size // 4, 1)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        
        classification = self.classifier(features)
        threat_type = self.threat_type_classifier(features)
        severity = torch.sigmoid(self.severity_regressor(features))
        
        return {
            "classification": classification,
            "threat_type": threat_type,
            "severity": severity,
            "features": features
        }

class FederatedDefenseGrid:
    """Main Federated Defense Grid system"""
    
    def __init__(self, node_id: str, agency_name: str, agency_type: AgencyType, 
                 security_clearance: SecurityClearance):
        self.node_id = node_id
        self.agency_name = agency_name
        self.agency_type = agency_type
        self.security_clearance = security_clearance
        
        # Core components
        self.crypto = CryptographicProtocols()
        self.differential_privacy = DifferentialPrivacy()
        self.secure_aggregation = SecureAggregation()
        
        # Network and models
        self.connected_nodes = {}
        self.global_models = {}
        self.local_models = {}
        
        # Data storage
        self.db_path = f"federated_grid_{node_id}.db"
        self._init_database()
        
        # Training configuration
        self.training_config = {
            "local_epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.001,
            "privacy_budget": 1.0,
            "aggregation_frequency": 10  # rounds
        }
        
        # Statistics
        self.stats = {
            "models_trained": 0,
            "updates_shared": 0,
            "updates_received": 0,
            "intel_shared": 0,
            "intel_received": 0,
            "collaboration_sessions": 0
        }
        
        logger.info(f"Federated Defense Grid node initialized: {node_id} ({agency_name})")
    
    def _init_database(self):
        """Initialize database for federated operations"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS federated_nodes (
                node_id TEXT PRIMARY KEY,
                node_data TEXT,
                trust_score REAL,
                last_active TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_updates (
                update_id TEXT PRIMARY KEY,
                source_node TEXT,
                model_type TEXT,
                timestamp TIMESTAMP,
                performance_metrics TEXT,
                privacy_metrics TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS threat_intelligence (
                intel_id TEXT PRIMARY KEY,
                source_agency TEXT,
                classification TEXT,
                threat_data TEXT,
                timestamp TIMESTAMP,
                expires_at TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collaboration_log (
                session_id TEXT PRIMARY KEY,
                participants TEXT,
                session_type TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                outcomes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_node(self, node: FederatedNode) -> bool:
        """Register a new node in the federation"""
        
        # Validate security clearance compatibility
        if not self._validate_security_clearance(node.security_clearance):
            logger.warning(f"Security clearance incompatible: {node.security_clearance}")
            return False
        
        # Validate agency authorization
        if not self._validate_agency_authorization(node.agency_type):
            logger.warning(f"Agency type not authorized: {node.agency_type}")
            return False
        
        # Add to connected nodes
        self.connected_nodes[node.node_id] = node
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        node_data = {
            "node_id": node.node_id,
            "agency_name": node.agency_name,
            "agency_type": node.agency_type.value,
            "security_clearance": node.security_clearance.value,
            "endpoint_url": node.endpoint_url,
            "capabilities": node.capabilities,
            "trust_score": node.trust_score,
            "compliance_status": node.compliance_status
        }
        
        cursor.execute('''
            INSERT OR REPLACE INTO federated_nodes 
            (node_id, node_data, trust_score, last_active)
            VALUES (?, ?, ?, ?)
        ''', (
            node.node_id,
            json.dumps(node_data),
            node.trust_score,
            datetime.now()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Node registered: {node.node_id} ({node.agency_name})")
        return True
    
    def _validate_security_clearance(self, clearance: SecurityClearance) -> bool:
        """Validate if node security clearance is compatible"""
        
        clearance_hierarchy = {
            SecurityClearance.UNCLASSIFIED: 0,
            SecurityClearance.CONFIDENTIAL: 1,
            SecurityClearance.SECRET: 2,
            SecurityClearance.TOP_SECRET: 3,
            SecurityClearance.COMPARTMENTED: 4
        }
        
        # Node clearance must be >= our clearance
        return clearance_hierarchy[clearance] >= clearance_hierarchy[self.security_clearance]
    
    def _validate_agency_authorization(self, agency_type: AgencyType) -> bool:
        """Validate if agency type is authorized for collaboration"""
        
        # Define collaboration matrix
        authorized_collaborations = {
            AgencyType.DEFENSE: [AgencyType.INTELLIGENCE, AgencyType.CYBERSECURITY, AgencyType.HOMELAND_SECURITY],
            AgencyType.INTELLIGENCE: [AgencyType.DEFENSE, AgencyType.LAW_ENFORCEMENT, AgencyType.FOREIGN_AFFAIRS],
            AgencyType.LAW_ENFORCEMENT: [AgencyType.INTELLIGENCE, AgencyType.CYBERSECURITY, AgencyType.HOMELAND_SECURITY],
            AgencyType.CYBERSECURITY: [AgencyType.DEFENSE, AgencyType.LAW_ENFORCEMENT, AgencyType.CRITICAL_INFRASTRUCTURE],
            AgencyType.HOMELAND_SECURITY: [AgencyType.DEFENSE, AgencyType.LAW_ENFORCEMENT, AgencyType.CYBERSECURITY],
            AgencyType.FOREIGN_AFFAIRS: [AgencyType.INTELLIGENCE, AgencyType.DEFENSE],
            AgencyType.CRITICAL_INFRASTRUCTURE: [AgencyType.CYBERSECURITY, AgencyType.HOMELAND_SECURITY]
        }
        
        authorized = authorized_collaborations.get(self.agency_type, [])
        return agency_type in authorized or agency_type == self.agency_type
    
    def train_local_model(self, model_type: ModelType, training_data: torch.Tensor, 
                         labels: torch.Tensor) -> Dict[str, Any]:
        """Train local model with differential privacy"""
        
        # Initialize or get existing model
        if model_type not in self.local_models:
            self.local_models[model_type] = FederatedThreatModel()
        
        model = self.local_models[model_type]
        optimizer = optim.Adam(model.parameters(), lr=self.training_config["learning_rate"])
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        training_metrics = {
            "initial_loss": None,
            "final_loss": None,
            "accuracy": 0.0,
            "privacy_cost": 0.0
        }
        
        # Training loop with differential privacy
        for epoch in range(self.training_config["local_epochs"]):
            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            # Create batches
            dataset_size = training_data.size(0)
            batch_size = self.training_config["batch_size"]
            
            for i in range(0, dataset_size, batch_size):
                batch_data = training_data[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_data)
                loss = criterion(outputs["classification"], batch_labels)
                
                # Backward pass
                loss.backward()
                
                # Apply differential privacy to gradients
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = self.differential_privacy.clip_gradients(param.grad)
                        param.grad = self.differential_privacy.add_noise_to_gradients(param.grad)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs["classification"], 1)
                total_predictions += batch_labels.size(0)
                correct_predictions += (predicted == batch_labels).sum().item()
            
            avg_loss = epoch_loss / (dataset_size // batch_size)
            if epoch == 0:
                training_metrics["initial_loss"] = avg_loss
            training_metrics["final_loss"] = avg_loss
        
        training_metrics["accuracy"] = correct_predictions / total_predictions
        training_metrics["privacy_cost"] = self.differential_privacy.epsilon
        
        self.stats["models_trained"] += 1
        
        logger.info(f"Local model training completed: {model_type.value}, Accuracy: {training_metrics['accuracy']:.3f}")
        
        return training_metrics
    
    def create_model_update(self, model_type: ModelType) -> ModelUpdate:
        """Create encrypted model update for sharing"""
        
        if model_type not in self.local_models:
            raise ValueError(f"No local model for type: {model_type}")
        
        model = self.local_models[model_type]
        
        # Extract model weights
        model_weights = torch.cat([param.data.flatten() for param in model.parameters()])
        
        # Apply differential privacy
        private_weights = self.differential_privacy.privatize_model_update(model_weights)
        
        # Encrypt weights (mock encryption for demonstration)
        encrypted_weights = self.crypto.encrypt_model_weights(
            private_weights, 
            self.crypto.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        )
        
        # Create update metadata
        metadata = {
            "model_architecture": "FederatedThreatModel",
            "training_rounds": self.training_config["local_epochs"],
            "data_samples": 1000,  # Mock value
            "model_version": "1.0.0",
            "node_capabilities": list(self.connected_nodes.keys())
        }
        
        # Calculate privacy metrics
        privacy_metrics = {
            "epsilon_used": self.differential_privacy.epsilon,
            "delta": self.differential_privacy.delta,
            "noise_scale": self.differential_privacy.noise_scale,
            "privacy_guarantee": "differential_privacy"
        }
        
        update_id = f"UPDATE_{self.node_id}_{int(time.time())}"
        
        # Create update object
        update = ModelUpdate(
            update_id=update_id,
            source_node=self.node_id,
            model_type=model_type,
            model_weights=encrypted_weights,
            metadata=metadata,
            timestamp=datetime.now(),
            signature=self.crypto.sign_model_update(encrypted_weights),
            contribution_quality=0.85,  # Mock quality score
            privacy_metrics=privacy_metrics
        )
        
        self.stats["updates_shared"] += 1
        
        return update
    
    def aggregate_received_updates(self, updates: List[ModelUpdate], 
                                 model_type: ModelType) -> Dict[str, Any]:
        """Aggregate received model updates"""
        
        # Filter updates for the specific model type
        relevant_updates = [u for u in updates if u.model_type == model_type]
        
        if len(relevant_updates) == 0:
            raise ValueError(f"No updates available for model type: {model_type}")
        
        # Validate updates
        valid_updates = []
        for update in relevant_updates:
            if self._validate_model_update(update):
                valid_updates.append(update)
            else:
                logger.warning(f"Invalid update from {update.source_node}")
        
        if len(valid_updates) == 0:
            raise ValueError("No valid updates to aggregate")
        
        # Perform secure aggregation
        try:
            aggregated_weights = self.secure_aggregation.aggregate_model_updates(valid_updates)
            
            # Update global model
            if model_type not in self.global_models:
                self.global_models[model_type] = FederatedThreatModel()
            
            # Apply aggregated weights to global model
            self._apply_weights_to_model(self.global_models[model_type], aggregated_weights)
            
            aggregation_result = {
                "aggregated_updates": len(valid_updates),
                "participating_nodes": [u.source_node for u in valid_updates],
                "aggregation_timestamp": datetime.now().isoformat(),
                "global_model_version": f"v{len(valid_updates)}",
                "privacy_preserved": True,
                "quality_metrics": {
                    "avg_contribution_quality": sum(u.contribution_quality for u in valid_updates) / len(valid_updates),
                    "aggregation_confidence": 0.9  # Mock confidence
                }
            }
            
            self.stats["updates_received"] += len(valid_updates)
            
            logger.info(f"Model aggregation completed: {model_type.value}, {len(valid_updates)} updates")
            
            return aggregation_result
            
        except Exception as e:
            logger.error(f"Model aggregation failed: {e}")
            raise
    
    def _validate_model_update(self, update: ModelUpdate) -> bool:
        """Validate received model update"""
        
        # Check source node authorization
        if update.source_node not in self.connected_nodes:
            return False
        
        source_node = self.connected_nodes[update.source_node]
        
        # Verify digital signature
        public_key_pem = source_node.public_key
        if not self.crypto.verify_signature(update.model_weights, update.signature, public_key_pem):
            return False
        
        # Check contribution quality
        if update.contribution_quality < 0.5:
            return False
        
        # Validate privacy metrics
        if "epsilon_used" not in update.privacy_metrics:
            return False
        
        return True
    
    def _apply_weights_to_model(self, model: nn.Module, weights: torch.Tensor):
        """Apply aggregated weights to model"""
        
        # This is a simplified implementation
        # In production, properly map weights to model parameters
        start_idx = 0
        
        for param in model.parameters():
            param_size = param.numel()
            param_weights = weights[start_idx:start_idx + param_size]
            param.data = param_weights.reshape(param.shape)
            start_idx += param_size
    
    def share_threat_intelligence(self, intel: ThreatIntelligence) -> bool:
        """Share threat intelligence with authorized nodes"""
        
        # Check sharing authorization
        authorized_nodes = self._get_authorized_nodes_for_intel(intel)
        
        if not authorized_nodes:
            logger.warning("No authorized nodes for threat intelligence sharing")
            return False
        
        # Store in local database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        intel_data = {
            "intel_id": intel.intel_id,
            "source_agency": intel.source_agency,
            "classification": intel.classification.value,
            "threat_type": intel.threat_type,
            "indicators": intel.indicators,
            "attribution": intel.attribution,
            "confidence_score": intel.confidence_score,
            "sharing_restrictions": intel.sharing_restrictions
        }
        
        cursor.execute('''
            INSERT OR REPLACE INTO threat_intelligence
            (intel_id, source_agency, classification, threat_data, timestamp, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            intel.intel_id,
            intel.source_agency,
            intel.classification.value,
            json.dumps(intel_data),
            datetime.now(),
            intel.expires_at
        ))
        
        conn.commit()
        conn.close()
        
        # Simulate sharing with authorized nodes
        for node_id in authorized_nodes:
            logger.info(f"Sharing threat intelligence {intel.intel_id} with {node_id}")
        
        self.stats["intel_shared"] += 1
        
        return True
    
    def _get_authorized_nodes_for_intel(self, intel: ThreatIntelligence) -> List[str]:
        """Get list of nodes authorized to receive threat intelligence"""
        
        authorized_nodes = []
        
        for node_id, node in self.connected_nodes.items():
            # Check security clearance
            if not self._validate_security_clearance(node.security_clearance):
                continue
            
            # Check sharing restrictions
            if "no_sharing" in intel.sharing_restrictions:
                continue
            
            if "allies_only" in intel.sharing_restrictions and node.agency_type not in [
                AgencyType.DEFENSE, AgencyType.INTELLIGENCE, AgencyType.CYBERSECURITY
            ]:
                continue
            
            # Check classification level
            classification_hierarchy = {
                SecurityClearance.UNCLASSIFIED: 0,
                SecurityClearance.CONFIDENTIAL: 1,
                SecurityClearance.SECRET: 2,
                SecurityClearance.TOP_SECRET: 3,
                SecurityClearance.COMPARTMENTED: 4
            }
            
            if classification_hierarchy[node.security_clearance] >= classification_hierarchy[intel.classification]:
                authorized_nodes.append(node_id)
        
        return authorized_nodes
    
    def start_collaborative_session(self, session_type: str, participants: List[str]) -> str:
        """Start collaborative defense session"""
        
        session_id = f"SESSION_{int(time.time())}_{len(participants)}"
        
        # Validate participants
        valid_participants = []
        for participant in participants:
            if participant in self.connected_nodes:
                valid_participants.append(participant)
            else:
                logger.warning(f"Unknown participant: {participant}")
        
        if len(valid_participants) < 2:
            raise ValueError("Insufficient valid participants for collaboration")
        
        # Log session start
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO collaboration_log
            (session_id, participants, session_type, start_time, end_time, outcomes)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            json.dumps(valid_participants),
            session_type,
            datetime.now(),
            None,
            None
        ))
        
        conn.commit()
        conn.close()
        
        self.stats["collaboration_sessions"] += 1
        
        logger.info(f"Collaborative session started: {session_id} ({session_type})")
        
        return session_id
    
    def get_federation_status(self) -> Dict[str, Any]:
        """Get current federation status and metrics"""
        
        return {
            "node_info": {
                "node_id": self.node_id,
                "agency_name": self.agency_name,
                "agency_type": self.agency_type.value,
                "security_clearance": self.security_clearance.value
            },
            "network_status": {
                "connected_nodes": len(self.connected_nodes),
                "trusted_nodes": len([n for n in self.connected_nodes.values() if n.trust_score > 0.8]),
                "active_models": len(self.global_models),
                "local_models": len(self.local_models)
            },
            "privacy_metrics": {
                "differential_privacy_enabled": True,
                "epsilon_budget": self.differential_privacy.epsilon,
                "secure_aggregation_active": True,
                "cryptographic_protection": "AES-256 + RSA-2048"
            },
            "collaboration_metrics": {
                "total_collaborations": self.stats["collaboration_sessions"],
                "intelligence_shared": self.stats["intel_shared"],
                "intelligence_received": self.stats["intel_received"],
                "model_updates_exchanged": self.stats["updates_shared"] + self.stats["updates_received"]
            },
            "last_update": datetime.now().isoformat(),
            "operational_status": "ACTIVE"
        }

# Factory function
def create_federated_defense_grid(node_id: str, agency_name: str, agency_type: AgencyType, 
                                security_clearance: SecurityClearance) -> FederatedDefenseGrid:
    """Create a federated defense grid node"""
    return FederatedDefenseGrid(node_id, agency_name, agency_type, security_clearance)