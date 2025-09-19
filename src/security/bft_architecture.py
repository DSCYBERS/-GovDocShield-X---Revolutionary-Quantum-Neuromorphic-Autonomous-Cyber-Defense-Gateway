"""
Byzantine Fault Tolerance (BFT) Architecture
Keeps the system operational even if nodes fail or act maliciously.
Uses consensus algorithms to ensure majority rules while ignoring malicious/conflicting nodes.
"""

import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import socket
import struct
from datetime import datetime, timezone
import numpy as np
from collections import defaultdict, deque
import random

logger = logging.getLogger(__name__)

class NodeState(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"
    OBSERVER = "observer"
    FAULTY = "faulty"

class MessageType(Enum):
    HEARTBEAT = "heartbeat"
    VOTE_REQUEST = "vote_request"
    VOTE_RESPONSE = "vote_response"
    APPEND_ENTRIES = "append_entries"
    APPEND_RESPONSE = "append_response"
    CONSENSUS_REQUEST = "consensus_request"
    CONSENSUS_RESPONSE = "consensus_response"
    FAULT_DETECTION = "fault_detection"
    NODE_STATUS = "node_status"

class FaultType(Enum):
    CRASH_FAULT = "crash_fault"
    BYZANTINE_FAULT = "byzantine_fault"
    NETWORK_PARTITION = "network_partition"
    TIMING_FAULT = "timing_fault"
    OMISSION_FAULT = "omission_fault"

@dataclass
class BFTMessage:
    """Message structure for BFT communication"""
    message_id: str
    message_type: MessageType
    sender_id: str
    receiver_id: str
    term: int
    timestamp: str
    payload: Dict[str, Any]
    signature: str
    message_hash: str

@dataclass
class NodeInfo:
    """Information about a BFT node"""
    node_id: str
    address: str
    port: int
    public_key: str
    state: NodeState
    last_heartbeat: float
    reputation_score: float
    fault_count: int
    is_trusted: bool

@dataclass
class ConsensusProposal:
    """Proposal for consensus decision"""
    proposal_id: str
    proposer_id: str
    term: int
    data: Dict[str, Any]
    timestamp: str
    signatures: Dict[str, str]
    votes: Dict[str, bool]
    finalized: bool

class ByzantineFaultDetector:
    """Detects Byzantine faults in network nodes"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.node_behaviors = defaultdict(list)
        self.message_history = defaultdict(deque)
        self.fault_threshold = 0.3  # 30% suspicious behavior triggers fault detection
        self.max_history_size = 1000
        
    def record_node_behavior(self, node_id: str, behavior: Dict[str, Any]):
        """Record behavior for fault analysis"""
        
        behavior['timestamp'] = time.time()
        self.node_behaviors[node_id].append(behavior)
        
        # Limit history size
        if len(self.node_behaviors[node_id]) > self.max_history_size:
            self.node_behaviors[node_id].pop(0)
    
    def detect_byzantine_behavior(self, node_id: str) -> Tuple[bool, List[str]]:
        """Detect Byzantine behavior patterns"""
        
        if node_id not in self.node_behaviors:
            return False, []
        
        behaviors = self.node_behaviors[node_id]
        if len(behaviors) < 10:  # Need sufficient data
            return False, []
        
        suspicious_patterns = []
        recent_behaviors = behaviors[-50:]  # Analyze recent behaviors
        
        # Pattern 1: Inconsistent voting
        votes = [b.get('vote') for b in recent_behaviors if 'vote' in b]
        if len(votes) > 5:
            vote_consistency = self._analyze_vote_consistency(votes)
            if vote_consistency < 0.6:
                suspicious_patterns.append("Inconsistent voting pattern")
        
        # Pattern 2: Message timing anomalies
        response_times = [b.get('response_time') for b in recent_behaviors if 'response_time' in b]
        if len(response_times) > 5:
            timing_anomaly = self._detect_timing_anomalies(response_times)
            if timing_anomaly:
                suspicious_patterns.append("Timing attack detected")
        
        # Pattern 3: Contradictory messages
        messages = [b.get('message_content') for b in recent_behaviors if 'message_content' in b]
        if len(messages) > 3:
            contradiction_score = self._detect_contradictions(messages)
            if contradiction_score > 0.3:
                suspicious_patterns.append("Contradictory messages detected")
        
        # Pattern 4: Selective message dropping
        dropped_messages = [b.get('dropped_message') for b in recent_behaviors if 'dropped_message' in b]
        if len(dropped_messages) > len(recent_behaviors) * 0.2:
            suspicious_patterns.append("Excessive message dropping")
        
        # Pattern 5: Invalid signatures
        invalid_signatures = [b for b in recent_behaviors if b.get('invalid_signature', False)]
        if len(invalid_signatures) > 2:
            suspicious_patterns.append("Multiple invalid signatures")
        
        is_byzantine = len(suspicious_patterns) >= 2 or any(
            pattern in suspicious_patterns for pattern in [
                "Contradictory messages detected", 
                "Multiple invalid signatures"
            ]
        )
        
        return is_byzantine, suspicious_patterns
    
    def _analyze_vote_consistency(self, votes: List[Any]) -> float:
        """Analyze consistency of voting behavior"""
        
        if not votes:
            return 1.0
        
        # Calculate entropy of votes
        vote_counts = {}
        for vote in votes:
            vote_str = str(vote)
            vote_counts[vote_str] = vote_counts.get(vote_str, 0) + 1
        
        # High entropy indicates inconsistent voting
        if len(vote_counts) <= 1:
            return 1.0
        
        total_votes = len(votes)
        entropy = 0
        for count in vote_counts.values():
            probability = count / total_votes
            entropy -= probability * np.log2(probability)
        
        max_entropy = np.log2(len(vote_counts))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Consistency is inverse of entropy
        return 1.0 - normalized_entropy
    
    def _detect_timing_anomalies(self, response_times: List[float]) -> bool:
        """Detect timing attack patterns"""
        
        if len(response_times) < 5:
            return False
        
        mean_time = np.mean(response_times)
        std_time = np.std(response_times)
        
        # Detect suspicious patterns
        outliers = [t for t in response_times if abs(t - mean_time) > 3 * std_time]
        
        # Too many outliers or very regular timing might indicate attack
        return len(outliers) > len(response_times) * 0.3 or std_time < 0.01
    
    def _detect_contradictions(self, messages: List[Any]) -> float:
        """Detect contradictory messages"""
        
        if len(messages) < 2:
            return 0.0
        
        contradiction_count = 0
        total_comparisons = 0
        
        for i in range(len(messages)):
            for j in range(i + 1, len(messages)):
                total_comparisons += 1
                
                msg1, msg2 = messages[i], messages[j]
                if self._messages_contradict(msg1, msg2):
                    contradiction_count += 1
        
        return contradiction_count / total_comparisons if total_comparisons > 0 else 0.0
    
    def _messages_contradict(self, msg1: Any, msg2: Any) -> bool:
        """Check if two messages contradict each other"""
        
        if not isinstance(msg1, dict) or not isinstance(msg2, dict):
            return False
        
        # Check for direct contradictions in similar message types
        if msg1.get('type') == msg2.get('type'):
            # Same type messages with different conclusions
            if ('decision' in msg1 and 'decision' in msg2 and 
                msg1['decision'] != msg2['decision']):
                return True
            
            # Same proposal with different content
            if ('proposal_id' in msg1 and 'proposal_id' in msg2 and 
                msg1['proposal_id'] == msg2['proposal_id'] and
                msg1.get('content') != msg2.get('content')):
                return True
        
        return False

class PBFTConsensus:
    """Practical Byzantine Fault Tolerance consensus algorithm"""
    
    def __init__(self, node_id: str, total_nodes: int):
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.f = (total_nodes - 1) // 3  # Maximum Byzantine nodes tolerated
        self.min_nodes_for_consensus = 2 * self.f + 1
        
        # Consensus state
        self.view_number = 0
        self.sequence_number = 0
        self.current_proposals = {}
        self.prepare_messages = defaultdict(dict)
        self.commit_messages = defaultdict(dict)
        
        # Performance tracking
        self.consensus_latency = deque(maxlen=100)
        self.consensus_success_rate = deque(maxlen=100)
        
    def initiate_consensus(self, data: Dict[str, Any]) -> str:
        """Initiate new consensus round"""
        
        proposal_id = f"prop_{self.node_id}_{self.sequence_number}_{int(time.time())}"
        
        proposal = ConsensusProposal(
            proposal_id=proposal_id,
            proposer_id=self.node_id,
            term=self.view_number,
            data=data,
            timestamp=datetime.now(timezone.utc).isoformat(),
            signatures={},
            votes={},
            finalized=False
        )
        
        self.current_proposals[proposal_id] = proposal
        self.sequence_number += 1
        
        logger.info(f"Initiated consensus proposal: {proposal_id}")
        return proposal_id
    
    def process_prepare_message(self, proposal_id: str, sender_id: str, signature: str) -> bool:
        """Process PREPARE message in PBFT"""
        
        if proposal_id not in self.current_proposals:
            return False
        
        # Verify signature (simplified)
        if not self._verify_signature(sender_id, signature):
            return False
        
        # Record prepare message
        self.prepare_messages[proposal_id][sender_id] = signature
        
        # Check if we have enough prepare messages
        if len(self.prepare_messages[proposal_id]) >= self.min_nodes_for_consensus:
            return self._advance_to_commit_phase(proposal_id)
        
        return True
    
    def process_commit_message(self, proposal_id: str, sender_id: str, signature: str) -> bool:
        """Process COMMIT message in PBFT"""
        
        if proposal_id not in self.current_proposals:
            return False
        
        # Verify signature
        if not self._verify_signature(sender_id, signature):
            return False
        
        # Record commit message
        self.commit_messages[proposal_id][sender_id] = signature
        
        # Check if we have enough commit messages
        if len(self.commit_messages[proposal_id]) >= self.min_nodes_for_consensus:
            return self._finalize_consensus(proposal_id)
        
        return True
    
    def _advance_to_commit_phase(self, proposal_id: str) -> bool:
        """Advance proposal to commit phase"""
        
        proposal = self.current_proposals[proposal_id]
        
        # Generate commit message
        commit_data = {
            'proposal_id': proposal_id,
            'phase': 'commit',
            'node_id': self.node_id,
            'view': self.view_number
        }
        
        # Sign commit
        commit_signature = self._sign_data(commit_data)
        
        # Add our own commit
        self.commit_messages[proposal_id][self.node_id] = commit_signature
        
        logger.info(f"Advanced proposal to commit phase: {proposal_id}")
        return True
    
    def _finalize_consensus(self, proposal_id: str) -> bool:
        """Finalize consensus decision"""
        
        start_time = time.time()
        
        proposal = self.current_proposals[proposal_id]
        proposal.finalized = True
        
        # Record consensus metrics
        latency = (time.time() - start_time) * 1000
        self.consensus_latency.append(latency)
        self.consensus_success_rate.append(1.0)
        
        logger.info(f"Consensus finalized: {proposal_id}")
        return True
    
    def _verify_signature(self, sender_id: str, signature: str) -> bool:
        """Verify message signature (simplified)"""
        # In production, use proper cryptographic verification
        return len(signature) > 10 and sender_id in signature
    
    def _sign_data(self, data: Dict[str, Any]) -> str:
        """Sign data (simplified)"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(f"{self.node_id}{data_str}".encode()).hexdigest()
    
    def get_consensus_metrics(self) -> Dict[str, Any]:
        """Get consensus performance metrics"""
        
        return {
            'average_latency_ms': np.mean(self.consensus_latency) if self.consensus_latency else 0,
            'success_rate': np.mean(self.consensus_success_rate) if self.consensus_success_rate else 1.0,
            'active_proposals': len(self.current_proposals),
            'finalized_proposals': sum(1 for p in self.current_proposals.values() if p.finalized),
            'current_view': self.view_number,
            'max_byzantine_nodes_tolerated': self.f
        }

class BFTNode:
    """Individual node in Byzantine Fault Tolerant network"""
    
    def __init__(self, node_id: str, address: str = "localhost", port: int = 8000):
        self.node_id = node_id
        self.address = address
        self.port = port
        self.state = NodeState.FOLLOWER
        
        # Network and consensus
        self.peers = {}
        self.consensus = None
        self.fault_detector = ByzantineFaultDetector(node_id)
        
        # Operational state
        self.is_running = False
        self.message_queue = asyncio.Queue()
        self.reputation = 1.0
        self.fault_count = 0
        
        # Performance metrics
        self.message_count = 0
        self.consensus_participated = 0
        self.uptime_start = time.time()
        
        # Threading
        self.lock = threading.Lock()
        
        logger.info(f"BFT Node initialized: {node_id}")
    
    def join_network(self, peer_nodes: List[Dict[str, Any]]):
        """Join BFT network with given peers"""
        
        total_nodes = len(peer_nodes) + 1  # Including self
        self.consensus = PBFTConsensus(self.node_id, total_nodes)
        
        # Add peer information
        for peer in peer_nodes:
            node_info = NodeInfo(
                node_id=peer['node_id'],
                address=peer['address'],
                port=peer['port'],
                public_key=peer.get('public_key', ''),
                state=NodeState.FOLLOWER,
                last_heartbeat=time.time(),
                reputation_score=1.0,
                fault_count=0,
                is_trusted=True
            )
            self.peers[peer['node_id']] = node_info
        
        logger.info(f"Joined BFT network with {len(self.peers)} peers")
    
    async def start(self):
        """Start BFT node operations"""
        
        self.is_running = True
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._message_processor()),
            asyncio.create_task(self._fault_detection_loop()),
            asyncio.create_task(self._network_monitor())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"BFT node error: {e}")
        finally:
            self.is_running = False
    
    async def stop(self):
        """Stop BFT node operations"""
        self.is_running = False
        logger.info(f"BFT node stopped: {self.node_id}")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to peers"""
        
        while self.is_running:
            heartbeat_message = BFTMessage(
                message_id=f"hb_{self.node_id}_{int(time.time())}",
                message_type=MessageType.HEARTBEAT,
                sender_id=self.node_id,
                receiver_id="broadcast",
                term=self.consensus.view_number if self.consensus else 0,
                timestamp=datetime.now(timezone.utc).isoformat(),
                payload={
                    'state': self.state.value,
                    'reputation': self.reputation,
                    'uptime': time.time() - self.uptime_start
                },
                signature="",
                message_hash=""
            )
            
            # Sign and hash message
            heartbeat_message.signature = self._sign_message(heartbeat_message)
            heartbeat_message.message_hash = self._hash_message(heartbeat_message)
            
            # Broadcast to all peers
            await self._broadcast_message(heartbeat_message)
            
            await asyncio.sleep(5)  # Heartbeat every 5 seconds
    
    async def _message_processor(self):
        """Process incoming messages"""
        
        while self.is_running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self._handle_message(message)
                self.message_count += 1
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Message processing error: {e}")
    
    async def _fault_detection_loop(self):
        """Monitor for Byzantine faults"""
        
        while self.is_running:
            current_time = time.time()
            
            # Check peer heartbeats
            for peer_id, peer_info in self.peers.items():
                time_since_heartbeat = current_time - peer_info.last_heartbeat
                
                if time_since_heartbeat > 30:  # 30 seconds timeout
                    await self._handle_suspected_fault(peer_id, FaultType.CRASH_FAULT)
                
                # Check for Byzantine behavior
                is_byzantine, patterns = self.fault_detector.detect_byzantine_behavior(peer_id)
                if is_byzantine:
                    await self._handle_suspected_fault(peer_id, FaultType.BYZANTINE_FAULT, patterns)
            
            await asyncio.sleep(10)  # Check every 10 seconds
    
    async def _network_monitor(self):
        """Monitor network health and performance"""
        
        while self.is_running:
            # Calculate network health metrics
            healthy_peers = sum(1 for peer in self.peers.values() 
                              if time.time() - peer.last_heartbeat < 15)
            
            network_health = healthy_peers / len(self.peers) if self.peers else 1.0
            
            # Log network status
            if network_health < 0.7:
                logger.warning(f"Network health degraded: {network_health:.2f}")
            
            await asyncio.sleep(30)  # Monitor every 30 seconds
    
    async def _handle_message(self, message: BFTMessage):
        """Handle incoming BFT message"""
        
        # Verify message integrity
        if not self._verify_message(message):
            logger.warning(f"Invalid message from {message.sender_id}")
            self._record_fault_behavior(message.sender_id, {'invalid_signature': True})
            return
        
        # Update sender's last seen time
        if message.sender_id in self.peers:
            self.peers[message.sender_id].last_heartbeat = time.time()
        
        # Route message based on type
        if message.message_type == MessageType.HEARTBEAT:
            await self._handle_heartbeat(message)
        elif message.message_type == MessageType.CONSENSUS_REQUEST:
            await self._handle_consensus_request(message)
        elif message.message_type == MessageType.VOTE_REQUEST:
            await self._handle_vote_request(message)
        elif message.message_type == MessageType.FAULT_DETECTION:
            await self._handle_fault_detection(message)
        else:
            logger.debug(f"Unhandled message type: {message.message_type}")
    
    async def _handle_heartbeat(self, message: BFTMessage):
        """Handle heartbeat message"""
        
        sender_id = message.sender_id
        if sender_id in self.peers:
            peer = self.peers[sender_id]
            peer.last_heartbeat = time.time()
            peer.state = NodeState(message.payload.get('state', 'follower'))
            
            # Update reputation based on uptime
            uptime = message.payload.get('uptime', 0)
            if uptime > 3600:  # Bonus for staying online > 1 hour
                peer.reputation_score = min(1.0, peer.reputation_score + 0.01)
    
    async def _handle_consensus_request(self, message: BFTMessage):
        """Handle consensus request"""
        
        if not self.consensus:
            return
        
        proposal_data = message.payload.get('data', {})
        proposal_id = self.consensus.initiate_consensus(proposal_data)
        
        # Participate in consensus
        self.consensus_participated += 1
        
        # Record behavior for fault detection
        self._record_fault_behavior(message.sender_id, {
            'consensus_request': True,
            'response_time': time.time()
        })
    
    async def _handle_vote_request(self, message: BFTMessage):
        """Handle voting request"""
        
        proposal_id = message.payload.get('proposal_id')
        vote_type = message.payload.get('vote_type', 'prepare')
        
        if not proposal_id or not self.consensus:
            return
        
        # Cast vote based on proposal validity
        vote_signature = self._sign_data({
            'proposal_id': proposal_id,
            'vote': True,
            'voter': self.node_id
        })
        
        if vote_type == 'prepare':
            self.consensus.process_prepare_message(proposal_id, self.node_id, vote_signature)
        elif vote_type == 'commit':
            self.consensus.process_commit_message(proposal_id, self.node_id, vote_signature)
        
        # Record voting behavior
        self._record_fault_behavior(message.sender_id, {
            'vote': True,
            'vote_type': vote_type,
            'proposal_id': proposal_id
        })
    
    async def _handle_fault_detection(self, message: BFTMessage):
        """Handle fault detection message"""
        
        suspected_node = message.payload.get('suspected_node')
        fault_type = message.payload.get('fault_type')
        evidence = message.payload.get('evidence', [])
        
        if suspected_node in self.peers:
            # Verify fault claims
            if self._verify_fault_evidence(suspected_node, fault_type, evidence):
                await self._isolate_faulty_node(suspected_node)
    
    async def _handle_suspected_fault(self, node_id: str, fault_type: FaultType, evidence: List[str] = None):
        """Handle suspected node fault"""
        
        if node_id not in self.peers:
            return
        
        peer = self.peers[node_id]
        peer.fault_count += 1
        peer.reputation_score = max(0.0, peer.reputation_score - 0.2)
        
        logger.warning(f"Suspected fault in node {node_id}: {fault_type.value}")
        
        # If multiple faults detected, consider isolation
        if peer.fault_count >= 3 or fault_type == FaultType.BYZANTINE_FAULT:
            await self._isolate_faulty_node(node_id)
        
        # Broadcast fault detection to network
        fault_message = BFTMessage(
            message_id=f"fault_{self.node_id}_{int(time.time())}",
            message_type=MessageType.FAULT_DETECTION,
            sender_id=self.node_id,
            receiver_id="broadcast",
            term=self.consensus.view_number if self.consensus else 0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload={
                'suspected_node': node_id,
                'fault_type': fault_type.value,
                'evidence': evidence or [],
                'detector_reputation': self.reputation
            },
            signature="",
            message_hash=""
        )
        
        fault_message.signature = self._sign_message(fault_message)
        fault_message.message_hash = self._hash_message(fault_message)
        
        await self._broadcast_message(fault_message)
    
    async def _isolate_faulty_node(self, node_id: str):
        """Isolate faulty node from network"""
        
        if node_id in self.peers:
            self.peers[node_id].state = NodeState.FAULTY
            self.peers[node_id].is_trusted = False
            
            logger.info(f"Node isolated due to faults: {node_id}")
    
    def _record_fault_behavior(self, node_id: str, behavior: Dict[str, Any]):
        """Record node behavior for fault analysis"""
        
        if node_id in self.peers:
            self.fault_detector.record_node_behavior(node_id, behavior)
    
    def _verify_message(self, message: BFTMessage) -> bool:
        """Verify message integrity and signature"""
        
        # Verify hash
        expected_hash = self._hash_message(message)
        if expected_hash != message.message_hash:
            return False
        
        # Verify signature (simplified)
        return len(message.signature) > 10
    
    def _verify_fault_evidence(self, node_id: str, fault_type: str, evidence: List[str]) -> bool:
        """Verify fault evidence claims"""
        
        # Simple verification - in production, use cryptographic proofs
        return len(evidence) >= 2 and node_id in self.peers
    
    def _sign_message(self, message: BFTMessage) -> str:
        """Sign BFT message"""
        
        message_dict = asdict(message)
        message_dict['signature'] = ""  # Exclude signature from signing
        message_dict['message_hash'] = ""  # Exclude hash from signing
        
        message_str = json.dumps(message_dict, sort_keys=True)
        return hashlib.sha256(f"{self.node_id}{message_str}".encode()).hexdigest()
    
    def _hash_message(self, message: BFTMessage) -> str:
        """Calculate message hash"""
        
        message_dict = asdict(message)
        message_dict['message_hash'] = ""  # Exclude hash from hashing
        
        message_str = json.dumps(message_dict, sort_keys=True)
        return hashlib.sha256(message_str.encode()).hexdigest()
    
    def _sign_data(self, data: Dict[str, Any]) -> str:
        """Sign arbitrary data"""
        
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(f"{self.node_id}{data_str}".encode()).hexdigest()
    
    async def _broadcast_message(self, message: BFTMessage):
        """Broadcast message to all peers"""
        
        # In production, implement actual network communication
        logger.debug(f"Broadcasting {message.message_type.value} from {self.node_id}")
    
    def get_node_status(self) -> Dict[str, Any]:
        """Get comprehensive node status"""
        
        return {
            'node_id': self.node_id,
            'state': self.state.value,
            'reputation': self.reputation,
            'fault_count': self.fault_count,
            'uptime_seconds': time.time() - self.uptime_start,
            'peers_count': len(self.peers),
            'healthy_peers': sum(1 for peer in self.peers.values() 
                               if time.time() - peer.last_heartbeat < 15),
            'byzantine_peers': sum(1 for peer in self.peers.values() 
                                 if peer.state == NodeState.FAULTY),
            'messages_processed': self.message_count,
            'consensus_participated': self.consensus_participated,
            'consensus_metrics': self.consensus.get_consensus_metrics() if self.consensus else {}
        }

class BFTNetwork:
    """Byzantine Fault Tolerant Network Manager"""
    
    def __init__(self, network_id: str = "govdocshield-bft"):
        self.network_id = network_id
        self.nodes = {}
        self.network_config = {
            'max_byzantine_nodes': 0,
            'consensus_timeout': 30,
            'heartbeat_interval': 5,
            'fault_detection_threshold': 3
        }
        
    def add_node(self, node: BFTNode):
        """Add node to BFT network"""
        
        self.nodes[node.node_id] = node
        
        # Update network configuration
        total_nodes = len(self.nodes)
        self.network_config['max_byzantine_nodes'] = (total_nodes - 1) // 3
        
        logger.info(f"Added node to BFT network: {node.node_id}")
        logger.info(f"Network can tolerate {self.network_config['max_byzantine_nodes']} Byzantine nodes")
    
    def remove_node(self, node_id: str):
        """Remove node from network"""
        
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"Removed node from BFT network: {node_id}")
    
    async def start_network(self):
        """Start all nodes in the network"""
        
        # Prepare peer lists for each node
        for node_id, node in self.nodes.items():
            peers = []
            for peer_id, peer_node in self.nodes.items():
                if peer_id != node_id:
                    peers.append({
                        'node_id': peer_id,
                        'address': peer_node.address,
                        'port': peer_node.port,
                        'public_key': f"pubkey_{peer_id}"
                    })
            
            node.join_network(peers)
        
        # Start all nodes
        tasks = [node.start() for node in self.nodes.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status"""
        
        healthy_nodes = sum(1 for node in self.nodes.values() if node.is_running)
        total_faults = sum(node.fault_count for node in self.nodes.values())
        
        return {
            'network_id': self.network_id,
            'total_nodes': len(self.nodes),
            'healthy_nodes': healthy_nodes,
            'max_byzantine_tolerance': self.network_config['max_byzantine_nodes'],
            'total_faults_detected': total_faults,
            'network_health': healthy_nodes / len(self.nodes) if self.nodes else 0,
            'consensus_active': any(node.consensus for node in self.nodes.values()),
            'nodes': {node_id: node.get_node_status() for node_id, node in self.nodes.items()}
        }

# Factory functions
def create_bft_node(node_id: str, address: str = "localhost", port: int = 8000) -> BFTNode:
    """Create a BFT node"""
    return BFTNode(node_id, address, port)

def create_bft_network(network_id: str = "govdocshield-bft") -> BFTNetwork:
    """Create a BFT network"""
    return BFTNetwork(network_id)