"""
Multi-Domain Steganography Detection
Detects payloads hidden in images, audio, video, text, fonts, metadata, and AI models.
"""

import numpy as np
import cv2
import librosa
import scipy.stats as stats
import torch
import torch.nn as nn
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import pandas as pd
import json
import logging
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import io
import struct
import zipfile
import pickle
from pathlib import Path
import re
import string

logger = logging.getLogger(__name__)

class SteganographyType(Enum):
    IMAGE_LSB = "image_lsb"
    IMAGE_DCT = "image_dct"
    AUDIO_LSB = "audio_lsb"
    AUDIO_SPECTRUM = "audio_spectrum"
    TEXT_SEMANTIC = "text_semantic"
    TEXT_SYNTACTIC = "text_syntactic"
    FONT_GEOMETRIC = "font_geometric"
    METADATA_HIDDEN = "metadata_hidden"
    AI_MODEL_WEIGHTS = "ai_model_weights"
    VIDEO_FRAME = "video_frame"
    NETWORK_COVERT = "network_covert"

class MediaType(Enum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"
    FONT = "font"
    DOCUMENT = "document"
    AI_MODEL = "ai_model"
    NETWORK_PACKET = "network_packet"

@dataclass
class SteganographyDetectionResult:
    """Result of steganography analysis"""
    media_type: MediaType
    detected_techniques: List[SteganographyType]
    confidence_scores: Dict[SteganographyType, float]
    suspicious_regions: List[Dict[str, Any]]
    entropy_analysis: Dict[str, float]
    statistical_anomalies: List[str]
    payload_probability: float
    estimated_payload_size: int
    extraction_possible: bool
    countermeasures: List[str]
    processing_time_ms: float

class ImageSteganographyDetector:
    """Detects steganography in images using multiple techniques"""
    
    def __init__(self):
        self.block_size = 8  # For DCT analysis
        
    def detect_lsb_steganography(self, image: np.ndarray) -> Tuple[float, List[Dict]]:
        """Detect LSB steganography using statistical analysis"""
        
        if len(image.shape) == 3:
            # Analyze each color channel
            channels = [image[:, :, i] for i in range(image.shape[2])]
        else:
            channels = [image]
        
        channel_scores = []
        suspicious_regions = []
        
        for ch_idx, channel in enumerate(channels):
            # LSB plane analysis
            lsb_plane = channel & 1
            
            # Calculate chi-square test for randomness
            chi2_score = self._chi_square_test(lsb_plane)
            
            # Analyze bit patterns
            pattern_score = self._analyze_bit_patterns(lsb_plane)
            
            # Look for non-random regions
            regions = self._find_suspicious_regions(lsb_plane, channel)
            
            combined_score = (chi2_score + pattern_score) / 2
            channel_scores.append(combined_score)
            
            if regions:
                for region in regions:
                    region['channel'] = ch_idx
                    suspicious_regions.append(region)
        
        overall_score = max(channel_scores) if channel_scores else 0.0
        return overall_score, suspicious_regions
    
    def detect_dct_steganography(self, image: np.ndarray) -> Tuple[float, List[Dict]]:
        """Detect DCT-based steganography"""
        
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        
        # Divide image into 8x8 blocks for DCT analysis
        h, w = gray_image.shape
        blocks = []
        
        for i in range(0, h - self.block_size + 1, self.block_size):
            for j in range(0, w - self.block_size + 1, self.block_size):
                block = gray_image[i:i+self.block_size, j:j+self.block_size]
                blocks.append((block, i, j))
        
        suspicious_blocks = []
        anomaly_scores = []
        
        for block, y, x in blocks:
            # Perform DCT
            dct_block = cv2.dct(np.float32(block))
            
            # Analyze high-frequency coefficients
            high_freq_score = self._analyze_high_frequency_coefficients(dct_block)
            
            # Check for regular patterns that might indicate embedding
            pattern_score = self._detect_dct_patterns(dct_block)
            
            combined_score = (high_freq_score + pattern_score) / 2
            anomaly_scores.append(combined_score)
            
            if combined_score > 0.7:
                suspicious_blocks.append({
                    'type': 'dct_anomaly',
                    'position': (x, y),
                    'size': (self.block_size, self.block_size),
                    'confidence': combined_score
                })
        
        overall_score = np.mean(anomaly_scores) if anomaly_scores else 0.0
        return overall_score, suspicious_blocks
    
    def _chi_square_test(self, data: np.ndarray) -> float:
        """Perform chi-square test for randomness"""
        
        flat_data = data.flatten()
        observed_freq, _ = np.histogram(flat_data, bins=2, range=(0, 2))
        
        expected_freq = len(flat_data) / 2
        
        if expected_freq == 0:
            return 0.0
        
        chi2 = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)
        
        # Normalize to [0, 1] range
        return min(1.0, chi2 / 10.0)
    
    def _analyze_bit_patterns(self, lsb_plane: np.ndarray) -> float:
        """Analyze patterns in LSB plane"""
        
        # Look for regular patterns that shouldn't exist in natural images
        h, w = lsb_plane.shape
        
        # Horizontal pattern analysis
        h_transitions = np.sum(np.abs(np.diff(lsb_plane, axis=1)))
        expected_h_transitions = h * (w - 1) * 0.5
        h_score = abs(h_transitions - expected_h_transitions) / expected_h_transitions
        
        # Vertical pattern analysis
        v_transitions = np.sum(np.abs(np.diff(lsb_plane, axis=0)))
        expected_v_transitions = (h - 1) * w * 0.5
        v_score = abs(v_transitions - expected_v_transitions) / expected_v_transitions
        
        return min(1.0, (h_score + v_score) / 2)
    
    def _find_suspicious_regions(self, lsb_plane: np.ndarray, original_channel: np.ndarray) -> List[Dict]:
        """Find regions with suspicious characteristics"""
        
        regions = []
        h, w = lsb_plane.shape
        
        # Divide into regions and analyze each
        region_size = 32
        for i in range(0, h - region_size + 1, region_size):
            for j in range(0, w - region_size + 1, region_size):
                region_lsb = lsb_plane[i:i+region_size, j:j+region_size]
                region_orig = original_channel[i:i+region_size, j:j+region_size]
                
                # Calculate entropy
                entropy = self._calculate_entropy(region_lsb.flatten())
                
                # Calculate variance
                variance = np.var(region_orig)
                
                # Suspicious if high entropy in LSB but low variance in original
                if entropy > 0.9 and variance < 100:
                    regions.append({
                        'type': 'entropy_variance_mismatch',
                        'position': (j, i),
                        'size': (region_size, region_size),
                        'entropy': entropy,
                        'variance': variance,
                        'confidence': entropy * (1 - variance / 1000)
                    })
        
        return regions
    
    def _analyze_high_frequency_coefficients(self, dct_block: np.ndarray) -> float:
        """Analyze high-frequency DCT coefficients for anomalies"""
        
        # High-frequency coefficients are in the bottom-right corner
        hf_coeffs = dct_block[4:, 4:]
        
        # Calculate variance of high-frequency coefficients
        hf_variance = np.var(hf_coeffs)
        
        # Natural images should have low high-frequency variance
        # Hidden data might increase this variance
        normalized_variance = min(1.0, hf_variance / 1000.0)
        
        return normalized_variance
    
    def _detect_dct_patterns(self, dct_block: np.ndarray) -> float:
        """Detect regular patterns in DCT coefficients"""
        
        # Look for patterns that might indicate systematic embedding
        coeffs = dct_block.flatten()
        
        # Check for regular intervals
        pattern_score = 0.0
        
        for step in [2, 3, 4, 5]:
            stepped_coeffs = coeffs[::step]
            if len(stepped_coeffs) > 1:
                correlation = np.corrcoef(stepped_coeffs[:-1], stepped_coeffs[1:])[0, 1]
                if not np.isnan(correlation):
                    pattern_score = max(pattern_score, abs(correlation))
        
        return pattern_score
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy"""
        
        if len(data) == 0:
            return 0.0
        
        _, counts = np.unique(data, return_counts=True)
        probabilities = counts / len(data)
        
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        # Normalize to [0, 1]
        max_entropy = np.log2(len(np.unique(data)))
        if max_entropy == 0:
            return 0.0
        
        return entropy / max_entropy

class AudioSteganographyDetector:
    """Detects steganography in audio files"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def detect_lsb_audio(self, audio_data: np.ndarray) -> Tuple[float, List[Dict]]:
        """Detect LSB steganography in audio"""
        
        # Ensure audio is integer format
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            audio_data = (audio_data * 32767).astype(np.int16)
        
        # Extract LSB
        lsb_bits = audio_data & 1
        
        # Statistical analysis of LSB sequence
        chi2_score = self._audio_chi_square_test(lsb_bits)
        entropy_score = self._calculate_entropy(lsb_bits)
        
        # Analyze for patterns
        pattern_score = self._analyze_audio_patterns(lsb_bits)
        
        overall_score = (chi2_score + entropy_score + pattern_score) / 3
        
        suspicious_regions = []
        if overall_score > 0.6:
            suspicious_regions.append({
                'type': 'lsb_audio_anomaly',
                'confidence': overall_score,
                'entropy': entropy_score,
                'chi2': chi2_score,
                'pattern': pattern_score
            })
        
        return overall_score, suspicious_regions
    
    def detect_spectral_steganography(self, audio_data: np.ndarray) -> Tuple[float, List[Dict]]:
        """Detect steganography in frequency domain"""
        
        # Perform STFT
        stft = librosa.stft(audio_data, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Analyze magnitude spectrum for anomalies
        mag_score = self._analyze_spectral_anomalies(magnitude)
        
        # Analyze phase spectrum for hidden data
        phase_score = self._analyze_phase_anomalies(phase)
        
        overall_score = (mag_score + phase_score) / 2
        
        suspicious_regions = []
        if overall_score > 0.5:
            suspicious_regions.append({
                'type': 'spectral_anomaly',
                'confidence': overall_score,
                'magnitude_score': mag_score,
                'phase_score': phase_score
            })
        
        return overall_score, suspicious_regions
    
    def _audio_chi_square_test(self, lsb_bits: np.ndarray) -> float:
        """Chi-square test for audio LSB randomness"""
        
        observed_freq, _ = np.histogram(lsb_bits, bins=2, range=(0, 2))
        expected_freq = len(lsb_bits) / 2
        
        if expected_freq == 0:
            return 0.0
        
        chi2 = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)
        return min(1.0, chi2 / 5.0)
    
    def _analyze_audio_patterns(self, lsb_bits: np.ndarray) -> float:
        """Analyze patterns in audio LSB sequence"""
        
        # Look for periodic patterns
        autocorr = np.correlate(lsb_bits, lsb_bits, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Find peaks that might indicate repeating patterns
        peak_score = 0.0
        for lag in range(2, min(1000, len(autocorr))):
            if autocorr[lag] > len(lsb_bits) * 0.6:  # Strong correlation
                peak_score = max(peak_score, autocorr[lag] / len(lsb_bits))
        
        return min(1.0, peak_score)
    
    def _analyze_spectral_anomalies(self, magnitude: np.ndarray) -> float:
        """Analyze magnitude spectrum for anomalies"""
        
        # Calculate variance across frequency bins
        freq_variance = np.var(magnitude, axis=0)
        
        # Look for bins with unusual variance
        mean_variance = np.mean(freq_variance)
        std_variance = np.std(freq_variance)
        
        # Count outliers
        outliers = np.sum(np.abs(freq_variance - mean_variance) > 2 * std_variance)
        outlier_ratio = outliers / len(freq_variance)
        
        return min(1.0, outlier_ratio * 5)
    
    def _analyze_phase_anomalies(self, phase: np.ndarray) -> float:
        """Analyze phase spectrum for hidden data"""
        
        # Phase should be relatively random in natural audio
        # Systematic modifications might indicate steganography
        
        phase_diff = np.diff(phase, axis=1)
        phase_variance = np.var(phase_diff)
        
        # Very low variance might indicate phase manipulation
        if phase_variance < 0.1:
            return 0.8
        elif phase_variance > 2.0:
            return 0.6
        else:
            return 0.2

class TextSteganographyDetector:
    """Detects steganography in text documents"""
    
    def __init__(self):
        self.common_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    
    def detect_semantic_steganography(self, text: str) -> Tuple[float, List[Dict]]:
        """Detect semantic steganography using NLP techniques"""
        
        # Analyze word frequency distribution
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Check for unusual word frequency patterns
        freq_score = self._analyze_word_frequency_anomalies(word_freq)
        
        # Analyze sentence structure
        sentences = re.split(r'[.!?]+', text)
        structure_score = self._analyze_sentence_structure(sentences)
        
        # Check for synonym substitution patterns
        synonym_score = self._detect_synonym_patterns(words)
        
        overall_score = (freq_score + structure_score + synonym_score) / 3
        
        suspicious_regions = []
        if overall_score > 0.5:
            suspicious_regions.append({
                'type': 'semantic_anomaly',
                'confidence': overall_score,
                'word_frequency_score': freq_score,
                'structure_score': structure_score,
                'synonym_score': synonym_score
            })
        
        return overall_score, suspicious_regions
    
    def detect_syntactic_steganography(self, text: str) -> Tuple[float, List[Dict]]:
        """Detect syntactic steganography (whitespace, punctuation patterns)"""
        
        # Analyze whitespace patterns
        whitespace_score = self._analyze_whitespace_patterns(text)
        
        # Analyze punctuation patterns
        punctuation_score = self._analyze_punctuation_patterns(text)
        
        # Analyze invisible characters
        invisible_score = self._detect_invisible_characters(text)
        
        overall_score = (whitespace_score + punctuation_score + invisible_score) / 3
        
        suspicious_regions = []
        if overall_score > 0.4:
            suspicious_regions.append({
                'type': 'syntactic_anomaly',
                'confidence': overall_score,
                'whitespace_score': whitespace_score,
                'punctuation_score': punctuation_score,
                'invisible_score': invisible_score
            })
        
        return overall_score, suspicious_regions
    
    def _analyze_word_frequency_anomalies(self, word_freq: Dict[str, int]) -> float:
        """Analyze word frequency for anomalies"""
        
        if not word_freq:
            return 0.0
        
        # Compare against expected frequency distribution (Zipf's law)
        sorted_freqs = sorted(word_freq.values(), reverse=True)
        
        # Calculate how much the distribution deviates from Zipf's law
        expected_freqs = [sorted_freqs[0] / (i + 1) for i in range(len(sorted_freqs))]
        
        # Calculate chi-square goodness of fit
        chi2 = 0.0
        for observed, expected in zip(sorted_freqs, expected_freqs):
            if expected > 0:
                chi2 += (observed - expected) ** 2 / expected
        
        # Normalize and return
        return min(1.0, chi2 / (len(sorted_freqs) * 10))
    
    def _analyze_sentence_structure(self, sentences: List[str]) -> float:
        """Analyze sentence structure for anomalies"""
        
        if not sentences:
            return 0.0
        
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        
        if not sentence_lengths:
            return 0.0
        
        # Check for unusual length patterns
        mean_length = np.mean(sentence_lengths)
        std_length = np.std(sentence_lengths)
        
        # Very uniform sentence lengths might indicate steganography
        if std_length < 1.0 and mean_length > 5:
            return 0.8
        
        # Very regular patterns (e.g., alternating lengths)
        if len(sentence_lengths) > 3:
            pattern_score = 0.0
            for i in range(len(sentence_lengths) - 2):
                if sentence_lengths[i] == sentence_lengths[i + 2]:
                    pattern_score += 1
            
            pattern_ratio = pattern_score / (len(sentence_lengths) - 2)
            if pattern_ratio > 0.7:
                return 0.7
        
        return 0.2
    
    def _detect_synonym_patterns(self, words: List[str]) -> float:
        """Detect patterns that might indicate synonym substitution"""
        
        # Simple heuristic: look for unusual word choices
        unusual_words = 0
        
        for word in words:
            if len(word) > 8 and word not in self.common_words:
                unusual_words += 1
        
        if len(words) == 0:
            return 0.0
        
        unusual_ratio = unusual_words / len(words)
        
        # High ratio of unusual words might indicate synonym substitution
        return min(1.0, unusual_ratio * 3)
    
    def _analyze_whitespace_patterns(self, text: str) -> float:
        """Analyze whitespace patterns for hidden data"""
        
        # Look for multiple spaces that might encode bits
        multiple_spaces = re.findall(r' {2,}', text)
        
        if not multiple_spaces:
            return 0.0
        
        # Analyze the pattern of space counts
        space_counts = [len(spaces) for spaces in multiple_spaces]
        
        # Check for binary-like patterns (e.g., only 2s and 3s)
        unique_counts = set(space_counts)
        if len(unique_counts) == 2 and all(c in [2, 3] for c in unique_counts):
            return 0.9
        
        # Check for other regular patterns
        if len(unique_counts) <= 3:
            return 0.6
        
        return 0.2
    
    def _analyze_punctuation_patterns(self, text: str) -> float:
        """Analyze punctuation patterns for steganography"""
        
        # Look for unusual punctuation sequences
        punct_sequences = re.findall(r'[^\w\s]+', text)
        
        if not punct_sequences:
            return 0.0
        
        # Check for patterns that might encode data
        pattern_score = 0.0
        
        for seq in punct_sequences:
            if len(seq) > 1:
                # Multiple punctuation marks might encode bits
                pattern_score += 1
        
        pattern_ratio = pattern_score / len(punct_sequences)
        return min(1.0, pattern_ratio * 2)
    
    def _detect_invisible_characters(self, text: str) -> float:
        """Detect invisible or unusual Unicode characters"""
        
        invisible_chars = 0
        
        for char in text:
            # Check for various invisible Unicode characters
            if ord(char) in [0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF]:  # Zero-width chars
                invisible_chars += 1
            elif ord(char) > 0x1000 and char not in string.printable:
                invisible_chars += 1
        
        if len(text) == 0:
            return 0.0
        
        invisible_ratio = invisible_chars / len(text)
        return min(1.0, invisible_ratio * 10)

class AIModelSteganographyDetector:
    """Detects steganography hidden in AI model weights"""
    
    def __init__(self):
        pass
    
    def detect_weight_steganography(self, model_weights: Dict[str, np.ndarray]) -> Tuple[float, List[Dict]]:
        """Detect hidden data in neural network weights"""
        
        layer_scores = []
        suspicious_layers = []
        
        for layer_name, weights in model_weights.items():
            # Analyze weight distribution
            weight_score = self._analyze_weight_distribution(weights, layer_name)
            layer_scores.append(weight_score)
            
            if weight_score > 0.6:
                suspicious_layers.append({
                    'type': 'weight_anomaly',
                    'layer': layer_name,
                    'confidence': weight_score,
                    'shape': weights.shape
                })
        
        overall_score = max(layer_scores) if layer_scores else 0.0
        return overall_score, suspicious_layers
    
    def _analyze_weight_distribution(self, weights: np.ndarray, layer_name: str) -> float:
        """Analyze weight distribution for anomalies"""
        
        flat_weights = weights.flatten()
        
        # Calculate statistical properties
        mean_weight = np.mean(flat_weights)
        std_weight = np.std(flat_weights)
        skewness = stats.skew(flat_weights)
        kurtosis = stats.kurtosis(flat_weights)
        
        # Check for unusual patterns
        anomaly_score = 0.0
        
        # Very high or low standard deviation
        if std_weight > 2.0 or std_weight < 0.001:
            anomaly_score += 0.3
        
        # Unusual skewness or kurtosis
        if abs(skewness) > 2.0:
            anomaly_score += 0.2
        if abs(kurtosis) > 10.0:
            anomaly_score += 0.2
        
        # Check for regular patterns in LSBs
        if weights.dtype in [np.float32, np.float64]:
            # Convert to binary representation and check LSBs
            binary_weights = weights.view(np.uint32 if weights.dtype == np.float32 else np.uint64)
            lsb_pattern_score = self._check_lsb_patterns(binary_weights.flatten())
            anomaly_score += lsb_pattern_score * 0.3
        
        return min(1.0, anomaly_score)
    
    def _check_lsb_patterns(self, binary_weights: np.ndarray) -> float:
        """Check for patterns in LSBs of weights"""
        
        lsbs = binary_weights & 1
        
        # Check for non-random patterns
        entropy = self._calculate_entropy(lsbs)
        
        # Entropy should be close to 1 for random data
        if entropy < 0.8:
            return 1.0 - entropy
        
        return 0.0

class MultiDomainSteganographyDetector:
    """Main steganography detection engine for all media types"""
    
    def __init__(self):
        self.image_detector = ImageSteganographyDetector()
        self.audio_detector = AudioSteganographyDetector()
        self.text_detector = TextSteganographyDetector()
        self.ai_detector = AIModelSteganographyDetector()
    
    def analyze_media(self, data: bytes, filename: str = "", 
                     media_type: MediaType = None) -> SteganographyDetectionResult:
        """
        Main analysis method for detecting steganography across all media types
        """
        start_time = time.time()
        
        # Auto-detect media type if not provided
        if not media_type:
            media_type = self._detect_media_type(data, filename)
        
        logger.info(f"Analyzing {media_type.value} for steganography: {filename}")
        
        detected_techniques = []
        confidence_scores = {}
        suspicious_regions = []
        entropy_analysis = {}
        statistical_anomalies = []
        
        try:
            if media_type == MediaType.IMAGE:
                result = self._analyze_image(data)
            elif media_type == MediaType.AUDIO:
                result = self._analyze_audio(data)
            elif media_type == MediaType.TEXT:
                result = self._analyze_text(data)
            elif media_type == MediaType.AI_MODEL:
                result = self._analyze_ai_model(data)
            else:
                result = self._analyze_generic(data)
            
            detected_techniques = result['techniques']
            confidence_scores = result['confidence_scores']
            suspicious_regions = result['suspicious_regions']
            entropy_analysis = result['entropy_analysis']
            statistical_anomalies = result['statistical_anomalies']
            
        except Exception as e:
            logger.error(f"Steganography analysis failed: {e}")
            statistical_anomalies.append(f"Analysis error: {str(e)}")
        
        # Calculate overall payload probability
        payload_probability = self._calculate_payload_probability(confidence_scores)
        
        # Estimate payload size
        estimated_payload_size = self._estimate_payload_size(
            len(data), detected_techniques, confidence_scores
        )
        
        # Determine if extraction is possible
        extraction_possible = any(score > 0.7 for score in confidence_scores.values())
        
        # Generate countermeasures
        countermeasures = self._generate_countermeasures(
            detected_techniques, media_type, payload_probability
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return SteganographyDetectionResult(
            media_type=media_type,
            detected_techniques=detected_techniques,
            confidence_scores=confidence_scores,
            suspicious_regions=suspicious_regions,
            entropy_analysis=entropy_analysis,
            statistical_anomalies=statistical_anomalies,
            payload_probability=payload_probability,
            estimated_payload_size=estimated_payload_size,
            extraction_possible=extraction_possible,
            countermeasures=countermeasures,
            processing_time_ms=processing_time
        )
    
    def _detect_media_type(self, data: bytes, filename: str) -> MediaType:
        """Detect media type from data and filename"""
        
        # Check magic numbers
        if data.startswith((b'\xff\xd8\xff', b'\x89PNG', b'GIF8', b'BM')):
            return MediaType.IMAGE
        elif data.startswith(b'RIFF') or data.startswith(b'ID3'):
            return MediaType.AUDIO
        elif filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return MediaType.VIDEO
        elif filename.endswith(('.pth', '.pkl', '.h5', '.pb')):
            return MediaType.AI_MODEL
        elif filename.endswith(('.ttf', '.otf', '.woff')):
            return MediaType.FONT
        else:
            # Try to decode as text
            try:
                data.decode('utf-8')
                return MediaType.TEXT
            except:
                return MediaType.DOCUMENT
    
    def _analyze_image(self, data: bytes) -> Dict[str, Any]:
        """Analyze image for steganography"""
        
        # Load image
        image = Image.open(io.BytesIO(data))
        image_array = np.array(image)
        
        # Detect LSB steganography
        lsb_score, lsb_regions = self.image_detector.detect_lsb_steganography(image_array)
        
        # Detect DCT steganography
        dct_score, dct_regions = self.image_detector.detect_dct_steganography(image_array)
        
        techniques = []
        confidence_scores = {}
        suspicious_regions = lsb_regions + dct_regions
        
        if lsb_score > 0.5:
            techniques.append(SteganographyType.IMAGE_LSB)
            confidence_scores[SteganographyType.IMAGE_LSB] = lsb_score
        
        if dct_score > 0.5:
            techniques.append(SteganographyType.IMAGE_DCT)
            confidence_scores[SteganographyType.IMAGE_DCT] = dct_score
        
        # Calculate entropy analysis
        entropy_analysis = {
            'overall_entropy': self.image_detector._calculate_entropy(image_array.flatten()),
            'lsb_entropy': self.image_detector._calculate_entropy((image_array & 1).flatten()) if len(image_array.shape) > 1 else 0
        }
        
        statistical_anomalies = []
        if entropy_analysis['lsb_entropy'] > 0.8:
            statistical_anomalies.append("High LSB entropy detected")
        
        return {
            'techniques': techniques,
            'confidence_scores': confidence_scores,
            'suspicious_regions': suspicious_regions,
            'entropy_analysis': entropy_analysis,
            'statistical_anomalies': statistical_anomalies
        }
    
    def _analyze_audio(self, data: bytes) -> Dict[str, Any]:
        """Analyze audio for steganography"""
        
        try:
            # Load audio using librosa
            audio_data, sr = librosa.load(io.BytesIO(data), sr=None)
        except:
            # Fallback: assume raw PCM data
            audio_data = np.frombuffer(data, dtype=np.int16)
            sr = 44100
        
        # Detect LSB steganography
        lsb_score, lsb_regions = self.audio_detector.detect_lsb_audio(audio_data)
        
        # Detect spectral steganography
        spectral_score, spectral_regions = self.audio_detector.detect_spectral_steganography(audio_data)
        
        techniques = []
        confidence_scores = {}
        suspicious_regions = lsb_regions + spectral_regions
        
        if lsb_score > 0.5:
            techniques.append(SteganographyType.AUDIO_LSB)
            confidence_scores[SteganographyType.AUDIO_LSB] = lsb_score
        
        if spectral_score > 0.5:
            techniques.append(SteganographyType.AUDIO_SPECTRUM)
            confidence_scores[SteganographyType.AUDIO_SPECTRUM] = spectral_score
        
        # Calculate entropy analysis
        entropy_analysis = {
            'audio_entropy': self.audio_detector._calculate_entropy(audio_data.astype(np.int16) & 1),
            'spectral_entropy': np.mean([self.audio_detector._calculate_entropy(frame) 
                                       for frame in np.array_split(audio_data, 10)])
        }
        
        statistical_anomalies = []
        if entropy_analysis['audio_entropy'] > 0.8:
            statistical_anomalies.append("High audio LSB entropy")
        
        return {
            'techniques': techniques,
            'confidence_scores': confidence_scores,
            'suspicious_regions': suspicious_regions,
            'entropy_analysis': entropy_analysis,
            'statistical_anomalies': statistical_anomalies
        }
    
    def _analyze_text(self, data: bytes) -> Dict[str, Any]:
        """Analyze text for steganography"""
        
        try:
            text = data.decode('utf-8')
        except:
            text = data.decode('utf-8', errors='ignore')
        
        # Detect semantic steganography
        semantic_score, semantic_regions = self.text_detector.detect_semantic_steganography(text)
        
        # Detect syntactic steganography
        syntactic_score, syntactic_regions = self.text_detector.detect_syntactic_steganography(text)
        
        techniques = []
        confidence_scores = {}
        suspicious_regions = semantic_regions + syntactic_regions
        
        if semantic_score > 0.5:
            techniques.append(SteganographyType.TEXT_SEMANTIC)
            confidence_scores[SteganographyType.TEXT_SEMANTIC] = semantic_score
        
        if syntactic_score > 0.4:
            techniques.append(SteganographyType.TEXT_SYNTACTIC)
            confidence_scores[SteganographyType.TEXT_SYNTACTIC] = syntactic_score
        
        # Calculate entropy analysis
        entropy_analysis = {
            'character_entropy': self.text_detector._calculate_entropy(np.array([ord(c) for c in text])),
            'word_entropy': len(set(text.split())) / len(text.split()) if text.split() else 0
        }
        
        statistical_anomalies = []
        if entropy_analysis['character_entropy'] < 0.3:
            statistical_anomalies.append("Low character entropy (possible encoding)")
        
        return {
            'techniques': techniques,
            'confidence_scores': confidence_scores,
            'suspicious_regions': suspicious_regions,
            'entropy_analysis': entropy_analysis,
            'statistical_anomalies': statistical_anomalies
        }
    
    def _analyze_ai_model(self, data: bytes) -> Dict[str, Any]:
        """Analyze AI model for steganography"""
        
        try:
            # Try to load as PyTorch model
            model_data = torch.load(io.BytesIO(data), map_location='cpu')
            if isinstance(model_data, dict):
                weights = {k: v.numpy() if hasattr(v, 'numpy') else v 
                          for k, v in model_data.items() if hasattr(v, 'shape')}
            else:
                weights = {'model': model_data.numpy() if hasattr(model_data, 'numpy') else model_data}
        except:
            # Fallback: try pickle
            try:
                weights = pickle.loads(data)
                if not isinstance(weights, dict):
                    weights = {'data': weights}
            except:
                weights = {}
        
        if not weights:
            return {
                'techniques': [],
                'confidence_scores': {},
                'suspicious_regions': [],
                'entropy_analysis': {},
                'statistical_anomalies': ['Could not parse model data']
            }
        
        # Detect weight steganography
        weight_score, weight_regions = self.ai_detector.detect_weight_steganography(weights)
        
        techniques = []
        confidence_scores = {}
        suspicious_regions = weight_regions
        
        if weight_score > 0.6:
            techniques.append(SteganographyType.AI_MODEL_WEIGHTS)
            confidence_scores[SteganographyType.AI_MODEL_WEIGHTS] = weight_score
        
        # Calculate entropy analysis
        all_weights = np.concatenate([w.flatten() for w in weights.values() if isinstance(w, np.ndarray)])
        entropy_analysis = {
            'weight_entropy': self.ai_detector._calculate_entropy(all_weights) if len(all_weights) > 0 else 0,
            'layer_count': len(weights)
        }
        
        statistical_anomalies = []
        if entropy_analysis['weight_entropy'] < 0.7:
            statistical_anomalies.append("Low weight entropy (possible hidden data)")
        
        return {
            'techniques': techniques,
            'confidence_scores': confidence_scores,
            'suspicious_regions': suspicious_regions,
            'entropy_analysis': entropy_analysis,
            'statistical_anomalies': statistical_anomalies
        }
    
    def _analyze_generic(self, data: bytes) -> Dict[str, Any]:
        """Generic analysis for unknown media types"""
        
        # Basic entropy and statistical analysis
        entropy = self._calculate_entropy(data)
        
        techniques = []
        confidence_scores = {}
        suspicious_regions = []
        
        # High entropy might indicate hidden data
        if entropy > 0.9:
            techniques.append(SteganographyType.METADATA_HIDDEN)
            confidence_scores[SteganographyType.METADATA_HIDDEN] = entropy
        
        entropy_analysis = {
            'data_entropy': entropy,
            'size': len(data)
        }
        
        statistical_anomalies = []
        if entropy > 0.95:
            statistical_anomalies.append("Very high entropy - possible encryption or steganography")
        elif entropy < 0.1:
            statistical_anomalies.append("Very low entropy - possible structured hiding")
        
        return {
            'techniques': techniques,
            'confidence_scores': confidence_scores,
            'suspicious_regions': suspicious_regions,
            'entropy_analysis': entropy_analysis,
            'statistical_anomalies': statistical_anomalies
        }
    
    def _calculate_entropy(self, data: Union[bytes, np.ndarray]) -> float:
        """Calculate Shannon entropy"""
        
        if isinstance(data, bytes):
            data_array = np.frombuffer(data, dtype=np.uint8)
        else:
            data_array = data.flatten()
        
        if len(data_array) == 0:
            return 0.0
        
        _, counts = np.unique(data_array, return_counts=True)
        probabilities = counts / len(data_array)
        
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        # Normalize to [0, 1]
        max_entropy = np.log2(len(np.unique(data_array)))
        if max_entropy == 0:
            return 0.0
        
        return entropy / max_entropy
    
    def _calculate_payload_probability(self, confidence_scores: Dict[SteganographyType, float]) -> float:
        """Calculate overall probability of hidden payload"""
        
        if not confidence_scores:
            return 0.0
        
        # Use maximum confidence as primary indicator
        max_confidence = max(confidence_scores.values())
        
        # Adjust based on number of detected techniques
        technique_count = len(confidence_scores)
        multiplier = min(1.2, 1 + (technique_count - 1) * 0.1)
        
        return min(1.0, max_confidence * multiplier)
    
    def _estimate_payload_size(self, data_size: int, techniques: List[SteganographyType], 
                              confidence_scores: Dict[SteganographyType, float]) -> int:
        """Estimate size of hidden payload"""
        
        if not techniques:
            return 0
        
        # Estimate based on technique and data size
        max_payload = 0
        
        for technique in techniques:
            confidence = confidence_scores.get(technique, 0)
            
            if technique == SteganographyType.IMAGE_LSB:
                # LSB can hide 1 bit per byte per channel
                estimated = int(data_size * 0.125 * confidence)  # 1/8 for 1 bit per byte
            elif technique == SteganographyType.AUDIO_LSB:
                estimated = int(data_size * 0.0625 * confidence)  # Less capacity in audio
            elif technique == SteganographyType.TEXT_SEMANTIC:
                estimated = int(data_size * 0.01 * confidence)  # Very small capacity
            elif technique == SteganographyType.AI_MODEL_WEIGHTS:
                estimated = int(data_size * 0.001 * confidence)  # Minimal capacity
            else:
                estimated = int(data_size * 0.05 * confidence)  # Default estimate
            
            max_payload = max(max_payload, estimated)
        
        return max_payload
    
    def _generate_countermeasures(self, techniques: List[SteganographyType], 
                                 media_type: MediaType, payload_probability: float) -> List[str]:
        """Generate countermeasures based on detected techniques"""
        
        countermeasures = []
        
        # General countermeasures
        if payload_probability > 0.7:
            countermeasures.extend([
                "Quarantine file immediately",
                "Perform forensic analysis",
                "Trace file origin and distribution"
            ])
        elif payload_probability > 0.5:
            countermeasures.extend([
                "Isolate file for further analysis",
                "Monitor for similar patterns",
                "Consider content reconstruction"
            ])
        
        # Technique-specific countermeasures
        if SteganographyType.IMAGE_LSB in techniques:
            countermeasures.extend([
                "Apply LSB noise injection",
                "Perform lossy recompression",
                "Strip and rebuild image structure"
            ])
        
        if SteganographyType.AUDIO_LSB in techniques:
            countermeasures.extend([
                "Apply audio resampling",
                "Use lossy compression",
                "Add minimal audio noise"
            ])
        
        if SteganographyType.TEXT_SEMANTIC in techniques:
            countermeasures.extend([
                "Normalize text formatting",
                "Remove unusual punctuation",
                "Verify content authenticity"
            ])
        
        if SteganographyType.AI_MODEL_WEIGHTS in techniques:
            countermeasures.extend([
                "Retrain model from clean data",
                "Apply weight quantization",
                "Verify model provenance"
            ])
        
        # Media-specific countermeasures
        if media_type == MediaType.IMAGE:
            countermeasures.append("Consider CDR++ image reconstruction")
        elif media_type == MediaType.TEXT:
            countermeasures.append("Extract and verify text content only")
        
        return list(set(countermeasures))[:8]  # Limit and deduplicate

# Factory function
def create_steganography_detector() -> MultiDomainSteganographyDetector:
    """Create a multi-domain steganography detector"""
    return MultiDomainSteganographyDetector()