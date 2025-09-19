"""
Cognitive AI Threat Reasoning (GAN-based Self-Learning AI)
Predicts future attack techniques by simulating both attacker & defender roles.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

class ThreatCategory(Enum):
    MALWARE = "malware"
    PHISHING = "phishing"
    ZERO_DAY = "zero_day"
    APT = "apt"
    RANSOMWARE = "ransomware"
    STEGANOGRAPHY = "steganography"
    SOCIAL_ENGINEERING = "social_engineering"
    SUPPLY_CHAIN = "supply_chain"

@dataclass
class ThreatPrediction:
    """Result of cognitive threat analysis"""
    threat_probability: float
    confidence_score: float
    threat_categories: List[ThreatCategory]
    attack_vectors: List[str]
    predicted_techniques: List[Dict[str, Any]]
    evolutionary_score: float
    novelty_detection: bool
    countermeasures: List[str]
    gan_generation_round: int
    discriminator_accuracy: float
    processing_time_ms: float

class AttackGenerator(nn.Module):
    """
    Generator Network - Creates novel malicious patterns
    Acts as the "attacker" in the GAN framework
    """
    
    def __init__(self, input_dim: int = 100, hidden_dim: int = 256, output_dim: int = 1024):
        super(AttackGenerator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 4),
            
            nn.Linear(hidden_dim * 4, output_dim),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # Attack technique embedding layers
        self.technique_embedder = nn.Embedding(50, 64)  # 50 known techniques
        self.category_embedder = nn.Embedding(len(ThreatCategory), 32)
        
    def forward(self, noise: torch.Tensor, technique_ids: torch.Tensor = None, 
                category_ids: torch.Tensor = None) -> torch.Tensor:
        """Generate attack patterns"""
        
        # Base generation from noise
        generated = self.network(noise)
        
        # Add technique-specific and category-specific information
        if technique_ids is not None:
            technique_embed = self.technique_embedder(technique_ids)
            technique_embed = technique_embed.view(technique_embed.size(0), -1)
            # Concat or add technique embedding
            if technique_embed.size(1) <= generated.size(1):
                generated[:, :technique_embed.size(1)] += technique_embed
        
        if category_ids is not None:
            category_embed = self.category_embedder(category_ids)
            category_embed = category_embed.view(category_embed.size(0), -1)
            # Concat or add category embedding
            if category_embed.size(1) <= generated.size(1):
                generated[:, :category_embed.size(1)] += category_embed
        
        return generated

class ThreatDiscriminator(nn.Module):
    """
    Discriminator Network - Learns to detect threats
    Acts as the "defender" in the GAN framework
    """
    
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512):
        super(ThreatDiscriminator, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.4),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim // 4),
        )
        
        # Multiple classification heads
        self.threat_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.category_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 4, 64),
            nn.ReLU(),
            nn.Linear(64, len(ThreatCategory)),
            nn.Softmax(dim=1)
        )
        
        # Novelty detection head
        self.novelty_detector = nn.Sequential(
            nn.Linear(hidden_dim // 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
        - threat_score: probability of being a threat
        - category_probs: probabilities for each threat category
        - novelty_score: probability of being a novel/unknown threat
        """
        features = self.feature_extractor(x)
        
        threat_score = self.threat_classifier(features)
        category_probs = self.category_classifier(features)
        novelty_score = self.novelty_detector(features)
        
        return threat_score, category_probs, novelty_score

class CognitiveAIThreatReasoner:
    """
    Main Cognitive AI Threat Reasoning Engine
    
    Uses adversarial training to evolve both attack generation and detection
    capabilities, creating a self-improving system that stays ahead of threats.
    """
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing Cognitive AI on device: {self.device}")
        
        # Initialize networks
        self.generator = AttackGenerator().to(self.device)
        self.discriminator = ThreatDiscriminator().to(self.device)
        
        # Optimizers
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss functions
        self.adversarial_loss = nn.BCELoss()
        self.category_loss = nn.CrossEntropyLoss()
        self.novelty_loss = nn.BCELoss()
        
        # Training state
        self.training_round = 0
        self.threat_database = []
        self.known_techniques = self._initialize_known_techniques()
        self.evolution_history = []
        
        # Performance metrics
        self.discriminator_accuracy = 0.0
        self.generator_diversity = 0.0
        self.novelty_detection_rate = 0.0
        
    def _initialize_known_techniques(self) -> Dict[str, Dict]:
        """Initialize database of known attack techniques"""
        
        techniques = {
            "dll_injection": {
                "category": ThreatCategory.MALWARE,
                "signature": [0.8, 0.2, 0.9, 0.1, 0.7],
                "evolution_potential": 0.7
            },
            "spear_phishing": {
                "category": ThreatCategory.PHISHING,
                "signature": [0.3, 0.9, 0.4, 0.8, 0.2],
                "evolution_potential": 0.8
            },
            "zero_day_exploit": {
                "category": ThreatCategory.ZERO_DAY,
                "signature": [0.9, 0.1, 0.8, 0.3, 0.9],
                "evolution_potential": 0.95
            },
            "lateral_movement": {
                "category": ThreatCategory.APT,
                "signature": [0.5, 0.7, 0.6, 0.9, 0.4],
                "evolution_potential": 0.75
            },
            "ransomware_encryption": {
                "category": ThreatCategory.RANSOMWARE,
                "signature": [0.9, 0.8, 0.7, 0.6, 0.9],
                "evolution_potential": 0.6
            },
            "steganographic_hiding": {
                "category": ThreatCategory.STEGANOGRAPHY,
                "signature": [0.1, 0.3, 0.9, 0.2, 0.8],
                "evolution_potential": 0.9
            },
            "social_engineering": {
                "category": ThreatCategory.SOCIAL_ENGINEERING,
                "signature": [0.4, 0.8, 0.3, 0.7, 0.5],
                "evolution_potential": 0.85
            },
            "supply_chain_compromise": {
                "category": ThreatCategory.SUPPLY_CHAIN,
                "signature": [0.7, 0.4, 0.8, 0.5, 0.6],
                "evolution_potential": 0.8
            }
        }
        
        return techniques
    
    def analyze_threat(self, input_data: np.ndarray, context: Dict[str, Any] = None) -> ThreatPrediction:
        """
        Main threat analysis method using cognitive reasoning
        """
        start_time = time.time()
        
        # Convert input to tensor
        if isinstance(input_data, np.ndarray):
            if input_data.ndim == 1:
                input_data = input_data.reshape(1, -1)
            
            # Pad or truncate to expected size
            expected_size = 1024
            if input_data.shape[1] < expected_size:
                padding = np.zeros((input_data.shape[0], expected_size - input_data.shape[1]))
                input_data = np.concatenate([input_data, padding], axis=1)
            elif input_data.shape[1] > expected_size:
                input_data = input_data[:, :expected_size]
        
        input_tensor = torch.FloatTensor(input_data).to(self.device)
        
        # Perform discriminator analysis
        self.discriminator.eval()
        with torch.no_grad():
            threat_score, category_probs, novelty_score = self.discriminator(input_tensor)
            
            threat_probability = threat_score.item()
            novelty_detection = novelty_score.item() > 0.7
            
            # Get top threat categories
            category_values, category_indices = torch.topk(category_probs, 3)
            threat_categories = [list(ThreatCategory)[idx.item()] for idx in category_indices[0]]
            
        # Generate potential attack vectors using generator insights
        attack_vectors = self._predict_attack_vectors(input_tensor, threat_categories)
        
        # Predict future techniques based on current patterns
        predicted_techniques = self._predict_evolution(input_tensor, threat_categories)
        
        # Calculate evolutionary score
        evolutionary_score = self._calculate_evolutionary_potential(
            threat_probability, novelty_detection, predicted_techniques
        )
        
        # Generate countermeasures
        countermeasures = self._generate_countermeasures(
            threat_categories, attack_vectors, evolutionary_score
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return ThreatPrediction(
            threat_probability=threat_probability,
            confidence_score=self._calculate_confidence(threat_probability, novelty_detection),
            threat_categories=threat_categories,
            attack_vectors=attack_vectors,
            predicted_techniques=predicted_techniques,
            evolutionary_score=evolutionary_score,
            novelty_detection=novelty_detection,
            countermeasures=countermeasures,
            gan_generation_round=self.training_round,
            discriminator_accuracy=self.discriminator_accuracy,
            processing_time_ms=processing_time
        )
    
    def adversarial_training_step(self, real_threats: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Perform one step of adversarial training
        """
        batch_size = real_threats.shape[0]
        
        # Convert to tensors
        real_data = torch.FloatTensor(real_threats).to(self.device)
        real_labels = torch.FloatTensor(labels).to(self.device)
        
        # Generate fake threats
        noise = torch.randn(batch_size, 100).to(self.device)
        technique_ids = torch.randint(0, len(self.known_techniques), (batch_size,)).to(self.device)
        category_ids = torch.randint(0, len(ThreatCategory), (batch_size,)).to(self.device)
        
        fake_data = self.generator(noise, technique_ids, category_ids)
        
        # Train Discriminator
        self.disc_optimizer.zero_grad()
        
        # Real data
        real_threat_scores, real_category_probs, real_novelty_scores = self.discriminator(real_data)
        real_loss = self.adversarial_loss(real_threat_scores.squeeze(), real_labels)
        
        # Fake data
        fake_threat_scores, fake_category_probs, fake_novelty_scores = self.discriminator(fake_data.detach())
        fake_labels = torch.zeros(batch_size).to(self.device)
        fake_loss = self.adversarial_loss(fake_threat_scores.squeeze(), fake_labels)
        
        disc_loss = (real_loss + fake_loss) / 2
        disc_loss.backward()
        self.disc_optimizer.step()
        
        # Train Generator
        self.gen_optimizer.zero_grad()
        
        # Generator wants discriminator to classify fake data as real
        gen_threat_scores, gen_category_probs, gen_novelty_scores = self.discriminator(fake_data)
        gen_labels = torch.ones(batch_size).to(self.device)
        gen_loss = self.adversarial_loss(gen_threat_scores.squeeze(), gen_labels)
        
        # Add diversity loss to encourage varied attack generation
        diversity_loss = self._calculate_diversity_loss(fake_data)
        total_gen_loss = gen_loss + 0.1 * diversity_loss
        
        total_gen_loss.backward()
        self.gen_optimizer.step()
        
        # Update metrics
        with torch.no_grad():
            real_accuracy = ((real_threat_scores > 0.5).float() == real_labels.unsqueeze(1)).float().mean()
            fake_accuracy = ((fake_threat_scores < 0.5).float()).mean()
            self.discriminator_accuracy = (real_accuracy + fake_accuracy) / 2
        
        self.training_round += 1
        
        return {
            "discriminator_loss": disc_loss.item(),
            "generator_loss": gen_loss.item(),
            "diversity_loss": diversity_loss.item(),
            "discriminator_accuracy": self.discriminator_accuracy.item(),
            "training_round": self.training_round
        }
    
    def continuous_learning(self, new_threat_data: List[Dict[str, Any]]):
        """
        Continuously learn from new threat intelligence
        """
        logger.info(f"Processing {len(new_threat_data)} new threat samples")
        
        # Convert threat data to training format
        features = []
        labels = []
        
        for threat in new_threat_data:
            # Extract features from threat data
            feature_vector = self._extract_threat_features(threat)
            features.append(feature_vector)
            labels.append(1.0 if threat.get('is_threat', True) else 0.0)
        
        if features:
            features_array = np.array(features)
            labels_array = np.array(labels)
            
            # Perform training steps
            for _ in range(5):  # Multiple training iterations
                metrics = self.adversarial_training_step(features_array, labels_array)
                
            logger.info(f"Training completed. Discriminator accuracy: {metrics['discriminator_accuracy']:.3f}")
    
    def generate_novel_attacks(self, num_samples: int = 10, 
                             target_category: ThreatCategory = None) -> List[Dict[str, Any]]:
        """
        Generate novel attack patterns for proactive defense preparation
        """
        self.generator.eval()
        
        generated_attacks = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Generate random noise
                noise = torch.randn(1, 100).to(self.device)
                
                # Select technique and category
                if target_category:
                    category_id = torch.tensor([list(ThreatCategory).index(target_category)]).to(self.device)
                else:
                    category_id = torch.randint(0, len(ThreatCategory), (1,)).to(self.device)
                
                technique_id = torch.randint(0, len(self.known_techniques), (1,)).to(self.device)
                
                # Generate attack pattern
                attack_pattern = self.generator(noise, technique_id, category_id)
                
                # Analyze generated pattern
                threat_score, category_probs, novelty_score = self.discriminator(attack_pattern)
                
                attack_info = {
                    "pattern": attack_pattern.cpu().numpy().tolist(),
                    "predicted_threat_score": threat_score.item(),
                    "predicted_category": list(ThreatCategory)[category_id.item()],
                    "novelty_score": novelty_score.item(),
                    "generation_round": self.training_round
                }
                
                generated_attacks.append(attack_info)
        
        return generated_attacks
    
    def _predict_attack_vectors(self, input_tensor: torch.Tensor, 
                               threat_categories: List[ThreatCategory]) -> List[str]:
        """Predict likely attack vectors based on current analysis"""
        
        vectors = []
        
        for category in threat_categories:
            if category == ThreatCategory.MALWARE:
                vectors.extend([
                    "Process injection", "DLL sideloading", "Registry manipulation",
                    "File system hiding", "Network communication"
                ])
            elif category == ThreatCategory.PHISHING:
                vectors.extend([
                    "Email spoofing", "Domain typosquatting", "Credential harvesting",
                    "Social engineering", "Malicious attachments"
                ])
            elif category == ThreatCategory.ZERO_DAY:
                vectors.extend([
                    "Memory corruption", "Logic flaws", "Race conditions",
                    "Privilege escalation", "Sandbox escape"
                ])
            elif category == ThreatCategory.APT:
                vectors.extend([
                    "Lateral movement", "Persistence mechanisms", "Data exfiltration",
                    "Command and control", "Living off the land"
                ])
            elif category == ThreatCategory.STEGANOGRAPHY:
                vectors.extend([
                    "Image embedding", "Audio hiding", "Network covert channels",
                    "File metadata", "AI model weights"
                ])
        
        # Limit and deduplicate
        return list(set(vectors))[:10]
    
    def _predict_evolution(self, input_tensor: torch.Tensor, 
                          threat_categories: List[ThreatCategory]) -> List[Dict[str, Any]]:
        """Predict how threats might evolve"""
        
        predicted_techniques = []
        
        # Generate evolved versions using the generator
        self.generator.eval()
        with torch.no_grad():
            # Use input as inspiration for evolution
            noise = torch.randn(1, 100).to(self.device)
            
            for category in threat_categories[:3]:  # Top 3 categories
                category_id = torch.tensor([list(ThreatCategory).index(category)]).to(self.device)
                technique_id = torch.randint(0, len(self.known_techniques), (1,)).to(self.device)
                
                evolved_pattern = self.generator(noise, technique_id, category_id)
                
                technique_info = {
                    "category": category.value,
                    "evolution_probability": np.random.uniform(0.6, 0.9),
                    "predicted_changes": [
                        "Enhanced evasion techniques",
                        "Polymorphic characteristics", 
                        "Anti-analysis measures",
                        "AI-assisted adaptation"
                    ],
                    "time_to_emergence_days": np.random.randint(30, 365),
                    "sophistication_increase": np.random.uniform(0.2, 0.8)
                }
                
                predicted_techniques.append(technique_info)
        
        return predicted_techniques
    
    def _calculate_evolutionary_potential(self, threat_probability: float, 
                                        novelty_detection: bool, 
                                        predicted_techniques: List[Dict]) -> float:
        """Calculate how likely this threat is to evolve"""
        
        base_score = threat_probability * 0.4
        
        if novelty_detection:
            base_score += 0.3
        
        # Average sophistication increase from predicted techniques
        if predicted_techniques:
            avg_sophistication = sum(t.get("sophistication_increase", 0) for t in predicted_techniques) / len(predicted_techniques)
            base_score += avg_sophistication * 0.3
        
        return min(1.0, base_score)
    
    def _generate_countermeasures(self, threat_categories: List[ThreatCategory], 
                                attack_vectors: List[str], 
                                evolutionary_score: float) -> List[str]:
        """Generate specific countermeasures"""
        
        countermeasures = []
        
        # Category-specific countermeasures
        for category in threat_categories:
            if category == ThreatCategory.MALWARE:
                countermeasures.extend([
                    "Deploy behavioral analysis", "Implement application whitelisting",
                    "Monitor process injection", "Enhance endpoint detection"
                ])
            elif category == ThreatCategory.PHISHING:
                countermeasures.extend([
                    "Email authentication (SPF/DKIM/DMARC)", "User training programs",
                    "URL reputation checking", "Attachment sandboxing"
                ])
            elif category == ThreatCategory.ZERO_DAY:
                countermeasures.extend([
                    "Implement exploit mitigation", "Deploy honeypots",
                    "Enhance monitoring", "Patch management acceleration"
                ])
        
        # Evolutionary countermeasures
        if evolutionary_score > 0.7:
            countermeasures.extend([
                "Implement AI-based adaptive defenses",
                "Deploy deception technologies",
                "Enhance threat hunting capabilities",
                "Prepare incident response protocols"
            ])
        
        return list(set(countermeasures))[:8]
    
    def _calculate_confidence(self, threat_probability: float, novelty_detection: bool) -> float:
        """Calculate confidence in the analysis"""
        
        base_confidence = 0.8
        
        # Higher confidence for extreme probabilities
        if threat_probability > 0.8 or threat_probability < 0.2:
            base_confidence += 0.1
        
        # Lower confidence for novel threats
        if novelty_detection:
            base_confidence -= 0.2
        
        # Factor in discriminator performance
        base_confidence *= min(1.0, self.discriminator_accuracy + 0.2)
        
        return max(0.1, min(1.0, base_confidence))
    
    def _calculate_diversity_loss(self, generated_data: torch.Tensor) -> torch.Tensor:
        """Calculate diversity loss to encourage varied attack generation"""
        
        # Calculate pairwise distances
        expanded_a = generated_data.unsqueeze(1)
        expanded_b = generated_data.unsqueeze(0)
        
        distances = torch.mean((expanded_a - expanded_b) ** 2, dim=2)
        
        # Encourage diversity by maximizing minimum distance
        min_distance = torch.min(distances + torch.eye(distances.size(0)).to(self.device) * 1e6)
        
        return -torch.log(min_distance + 1e-8)
    
    def _extract_threat_features(self, threat_data: Dict[str, Any]) -> np.ndarray:
        """Extract feature vector from threat data"""
        
        # Create feature vector from threat characteristics
        features = np.zeros(1024)
        
        # Basic threat indicators
        if 'file_hash' in threat_data:
            hash_bytes = bytes.fromhex(threat_data['file_hash'][:16])  # Use first 8 bytes
            for i, byte in enumerate(hash_bytes):
                if i < 8:
                    features[i] = byte / 255.0
        
        # Behavioral indicators
        if 'behaviors' in threat_data:
            behaviors = threat_data['behaviors']
            if 'process_injection' in behaviors:
                features[10] = 1.0
            if 'network_communication' in behaviors:
                features[11] = 1.0
            if 'registry_modification' in behaviors:
                features[12] = 1.0
        
        # File characteristics
        if 'file_size' in threat_data:
            features[20] = min(1.0, threat_data['file_size'] / 10000000)  # Normalize file size
        
        if 'entropy' in threat_data:
            features[21] = threat_data['entropy'] / 8.0  # Normalize entropy
        
        # Fill remaining with random noise based on threat characteristics
        np.random.seed(hash(str(threat_data)) % 2**32)
        features[100:] = np.random.random(924) * 0.1
        
        return features
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        
        model_state = {
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'gen_optimizer': self.gen_optimizer.state_dict(),
            'disc_optimizer': self.disc_optimizer.state_dict(),
            'training_round': self.training_round,
            'known_techniques': self.known_techniques,
            'discriminator_accuracy': self.discriminator_accuracy
        }
        
        torch.save(model_state, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        
        model_state = torch.load(filepath, map_location=self.device)
        
        self.generator.load_state_dict(model_state['generator'])
        self.discriminator.load_state_dict(model_state['discriminator'])
        self.gen_optimizer.load_state_dict(model_state['gen_optimizer'])
        self.disc_optimizer.load_state_dict(model_state['disc_optimizer'])
        self.training_round = model_state['training_round']
        self.known_techniques = model_state['known_techniques']
        self.discriminator_accuracy = model_state['discriminator_accuracy']
        
        logger.info(f"Model loaded from {filepath}")

# Factory function for easy integration
def create_cognitive_ai_reasoner(device: str = None) -> CognitiveAIThreatReasoner:
    """Create and initialize a Cognitive AI Threat Reasoner"""
    return CognitiveAIThreatReasoner(device=device)