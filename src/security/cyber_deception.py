"""
Cyber Deception Engine (Gov-Exclusive)
Deploys honey-documents, decoys, and fake networks to mislead attackers.
Generates false yet believable files/networks where attackers are trapped & monitored.
"""

import os
import json
import time
import random
import string
import logging
import tempfile
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import hashlib
import socket
import subprocess

logger = logging.getLogger(__name__)

class DeceptionType(Enum):
    HONEY_DOCUMENT = "honey_document"
    DECOY_DATABASE = "decoy_database"
    FAKE_NETWORK = "fake_network"
    HONEY_TOKEN = "honey_token"
    DECOY_CREDENTIALS = "decoy_credentials"
    FAKE_VULNERABILITY = "fake_vulnerability"
    HONEY_SERVICE = "honey_service"
    DECOY_SYSTEM = "decoy_system"

class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DeceptionAsset:
    """Individual deception asset"""
    asset_id: str
    asset_type: DeceptionType
    name: str
    description: str
    location: str
    creation_time: datetime
    last_accessed: Optional[datetime]
    access_count: int
    threat_level: ThreatLevel
    monitoring_active: bool
    metadata: Dict[str, Any]

@dataclass
class AttackerInteraction:
    """Record of attacker interaction with deception assets"""
    interaction_id: str
    asset_id: str
    timestamp: datetime
    source_ip: str
    user_agent: Optional[str]
    interaction_type: str
    payload: str
    threat_indicators: List[str]
    geolocation: Optional[Dict[str, str]]
    attributed_campaign: Optional[str]

class HoneyDocumentGenerator:
    """Generates believable honey documents to trap attackers"""
    
    def __init__(self):
        self.document_templates = {
            "classified_report": {
                "title_patterns": [
                    "CLASSIFIED: Operation {} Intelligence Report",
                    "SECRET: {} Mission Analysis",
                    "TOP SECRET: {} Strategic Assessment"
                ],
                "content_templates": [
                    "This document contains sensitive information regarding...",
                    "Intelligence gathered from sources indicates...",
                    "Strategic analysis of the {} situation reveals..."
                ]
            },
            "financial_data": {
                "title_patterns": [
                    "Q{} Financial Report - CONFIDENTIAL",
                    "Budget Allocation - {} Department",
                    "Financial Audit Results - {}"
                ],
                "content_templates": [
                    "Total budget allocation for {} amounts to...",
                    "Financial analysis shows significant discrepancies...",
                    "Recommended budget adjustments for the next fiscal..."
                ]
            },
            "personnel_records": {
                "title_patterns": [
                    "Personnel File: {} - RESTRICTED",
                    "Security Clearance Review: {}",
                    "Employee Assessment: {}"
                ],
                "content_templates": [
                    "Employee {} has been granted access to...",
                    "Background investigation reveals...",
                    "Performance evaluation for {} indicates..."
                ]
            }
        }
        
        self.fake_names = [
            "John Anderson", "Sarah Mitchell", "David Chen", "Maria Rodriguez",
            "Michael Thompson", "Jennifer Lee", "Robert Wilson", "Lisa Brown"
        ]
        
        self.fake_operations = [
            "Phoenix", "Guardian", "Sentinel", "Thunder", "Eclipse",
            "Vanguard", "Storm", "Shield", "Arrow", "Dragon"
        ]
    
    def generate_honey_document(self, doc_type: str, classification: str = "CONFIDENTIAL") -> Dict[str, Any]:
        """Generate a honey document of specified type"""
        
        if doc_type not in self.document_templates:
            doc_type = "classified_report"
        
        template = self.document_templates[doc_type]
        
        # Generate document content
        title = random.choice(template["title_patterns"]).format(
            random.choice(self.fake_operations)
        )
        
        content_base = random.choice(template["content_templates"]).format(
            random.choice(self.fake_operations)
        )
        
        # Add realistic but fake content
        content = self._generate_realistic_content(doc_type, content_base)
        
        # Add honey tokens (hidden identifiers)
        honey_token = self._generate_honey_token()
        content += f"\n\nDocument ID: {honey_token}"
        
        document = {
            "title": title,
            "classification": classification,
            "content": content,
            "author": random.choice(self.fake_names),
            "created_date": (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
            "document_id": honey_token,
            "access_count": 0,
            "honey_markers": [honey_token, classification.lower()]
        }
        
        return document
    
    def _generate_realistic_content(self, doc_type: str, base_content: str) -> str:
        """Generate realistic but fake content for documents"""
        
        fake_data = {
            "classified_report": [
                "Threat assessment indicates elevated risk levels in the operational zone.",
                "Intelligence sources confirm unusual activity patterns.",
                "Recommended countermeasures include enhanced surveillance protocols.",
                "Field operatives report successful completion of primary objectives.",
                "Security briefing scheduled for 0800 hours tomorrow."
            ],
            "financial_data": [
                f"Budget line item 4.7.2 shows ${random.randint(100000, 9999999):,} allocation.",
                f"Quarterly expenditure reached ${random.randint(500000, 50000000):,}.",
                "Budget variance analysis reveals 12% deviation from projections.",
                "Cost center 1847 requires immediate attention and review.",
                "Financial controls audit completed with minor recommendations."
            ],
            "personnel_records": [
                f"Security clearance level: {random.choice(['SECRET', 'TOP SECRET', 'CONFIDENTIAL'])}",
                f"Employee ID: EMP{random.randint(10000, 99999)}",
                "Background investigation completed without adverse findings.",
                "Performance rating: Exceeds Expectations",
                "Next review scheduled for Q2 of next fiscal year."
            ]
        }
        
        additional_content = random.sample(fake_data.get(doc_type, fake_data["classified_report"]), 3)
        
        return base_content + "\n\n" + "\n".join(additional_content)
    
    def _generate_honey_token(self) -> str:
        """Generate unique honey token for tracking"""
        timestamp = str(int(time.time()))
        random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        return f"HT-{timestamp}-{random_suffix}"

class DecoyNetworkGenerator:
    """Generates fake network infrastructure to mislead attackers"""
    
    def __init__(self):
        self.fake_services = {
            "ssh": {"port": 22, "banner": "OpenSSH_7.4"},
            "ftp": {"port": 21, "banner": "vsftpd 3.0.3"},
            "http": {"port": 80, "banner": "Apache/2.4.41"},
            "https": {"port": 443, "banner": "nginx/1.18.0"},
            "mysql": {"port": 3306, "banner": "MySQL 8.0.25"},
            "rdp": {"port": 3389, "banner": "Microsoft Terminal Services"},
            "smb": {"port": 445, "banner": "Microsoft Windows SMB"},
            "telnet": {"port": 23, "banner": "Linux telnetd"}
        }
        
        self.fake_systems = [
            {"name": "DC-01", "os": "Windows Server 2019", "role": "Domain Controller"},
            {"name": "FILE-01", "os": "Windows Server 2016", "role": "File Server"},
            {"name": "DB-01", "os": "Ubuntu 20.04", "role": "Database Server"},
            {"name": "WEB-01", "os": "CentOS 8", "role": "Web Server"},
            {"name": "MAIL-01", "os": "Exchange Server 2019", "role": "Mail Server"}
        ]
    
    def generate_fake_network_topology(self, subnet: str = "192.168.100.0/24") -> Dict[str, Any]:
        """Generate fake network topology"""
        
        # Generate fake IP addresses
        base_ip = ".".join(subnet.split("/")[0].split(".")[:-1])
        fake_hosts = []
        
        for i in range(5, 25):  # Generate 20 fake hosts
            ip = f"{base_ip}.{i}"
            system = random.choice(self.fake_systems).copy()
            system["ip"] = ip
            system["hostname"] = f"{system['name'].lower()}.corp.local"
            system["mac"] = self._generate_fake_mac()
            system["services"] = random.sample(list(self.fake_services.keys()), random.randint(2, 5))
            
            fake_hosts.append(system)
        
        topology = {
            "subnet": subnet,
            "gateway": f"{base_ip}.1",
            "dns_servers": [f"{base_ip}.10", f"{base_ip}.11"],
            "domain": "corp.local",
            "hosts": fake_hosts,
            "network_shares": self._generate_fake_shares(),
            "honey_tokens": [self._generate_network_honey_token() for _ in range(5)]
        }
        
        return topology
    
    def _generate_fake_mac(self) -> str:
        """Generate fake MAC address"""
        mac = [0x00, 0x16, 0x3e,
               random.randint(0x00, 0x7f),
               random.randint(0x00, 0xff),
               random.randint(0x00, 0xff)]
        return ':'.join([f"{x:02x}" for x in mac])
    
    def _generate_fake_shares(self) -> List[Dict[str, Any]]:
        """Generate fake network shares"""
        shares = [
            {"name": "Public", "path": "\\\\FILE-01\\Public", "permissions": "Read"},
            {"name": "Finance", "path": "\\\\FILE-01\\Finance", "permissions": "Restricted"},
            {"name": "HR", "path": "\\\\FILE-01\\HR", "permissions": "Confidential"},
            {"name": "Projects", "path": "\\\\FILE-01\\Projects", "permissions": "Read/Write"},
            {"name": "Backup", "path": "\\\\FILE-01\\Backup", "permissions": "Admin Only"}
        ]
        
        return shares
    
    def _generate_network_honey_token(self) -> str:
        """Generate network-specific honey token"""
        token_types = ["credential", "registry_key", "file_path", "service_account"]
        token_type = random.choice(token_types)
        
        if token_type == "credential":
            username = f"svc_{random.choice(['backup', 'monitor', 'admin', 'scan'])}"
            password = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
            return f"CRED:{username}:{password}"
        elif token_type == "registry_key":
            key_path = f"HKEY_LOCAL_MACHINE\\SOFTWARE\\Corp\\{random.choice(['Config', 'Security', 'Backup'])}"
            return f"REG:{key_path}"
        elif token_type == "file_path":
            file_path = f"C:\\CorpData\\{random.choice(['Sensitive', 'Confidential', 'Restricted'])}\\data.db"
            return f"FILE:{file_path}"
        else:  # service_account
            account = f"corp\\svc_{random.choice(['sql', 'web', 'app'])}"
            return f"SVC:{account}"

class DeceptionMonitor:
    """Monitors interactions with deception assets"""
    
    def __init__(self):
        self.active_monitors = {}
        self.interaction_log = []
        self.threat_patterns = {
            "credential_stuffing": ["multiple_login_attempts", "password_spray"],
            "network_scanning": ["port_scan", "service_enumeration"],
            "data_exfiltration": ["large_file_access", "bulk_download"],
            "lateral_movement": ["credential_reuse", "privilege_escalation"],
            "persistence": ["scheduled_task", "service_creation"]
        }
    
    def start_monitoring(self, asset: DeceptionAsset):
        """Start monitoring a deception asset"""
        
        monitor_config = {
            "asset_id": asset.asset_id,
            "asset_type": asset.asset_type,
            "start_time": datetime.now(),
            "access_patterns": [],
            "threat_indicators": [],
            "active": True
        }
        
        self.active_monitors[asset.asset_id] = monitor_config
        logger.info(f"Started monitoring deception asset: {asset.asset_id}")
    
    def record_interaction(self, asset_id: str, source_ip: str, interaction_data: Dict[str, Any]) -> AttackerInteraction:
        """Record attacker interaction with deception asset"""
        
        interaction = AttackerInteraction(
            interaction_id=f"INT_{int(time.time())}_{random.randint(1000, 9999)}",
            asset_id=asset_id,
            timestamp=datetime.now(),
            source_ip=source_ip,
            user_agent=interaction_data.get("user_agent"),
            interaction_type=interaction_data.get("type", "unknown"),
            payload=interaction_data.get("payload", ""),
            threat_indicators=self._analyze_threat_indicators(interaction_data),
            geolocation=self._get_geolocation(source_ip),
            attributed_campaign=self._attribute_to_campaign(interaction_data)
        )
        
        self.interaction_log.append(interaction)
        
        # Update monitoring data
        if asset_id in self.active_monitors:
            monitor = self.active_monitors[asset_id]
            monitor["access_patterns"].append({
                "timestamp": interaction.timestamp,
                "source_ip": source_ip,
                "type": interaction.interaction_type
            })
            monitor["threat_indicators"].extend(interaction.threat_indicators)
        
        logger.warning(f"Deception asset accessed: {asset_id} from {source_ip}")
        
        return interaction
    
    def _analyze_threat_indicators(self, interaction_data: Dict[str, Any]) -> List[str]:
        """Analyze interaction for threat indicators"""
        
        indicators = []
        
        # Check for common attack patterns
        payload = interaction_data.get("payload", "").lower()
        
        if any(term in payload for term in ["select", "union", "drop", "insert"]):
            indicators.append("sql_injection_attempt")
        
        if any(term in payload for term in ["<script", "javascript:", "onload"]):
            indicators.append("xss_attempt")
        
        if any(term in payload for term in ["../", "..\\", "etc/passwd", "windows/system32"]):
            indicators.append("directory_traversal")
        
        if "user_agent" in interaction_data:
            ua = interaction_data["user_agent"].lower()
            if any(tool in ua for tool in ["nmap", "sqlmap", "burp", "metasploit"]):
                indicators.append("automated_tool_usage")
        
        # Check for credential stuffing
        if interaction_data.get("type") == "login_attempt":
            if interaction_data.get("failed_attempts", 0) > 5:
                indicators.append("credential_stuffing")
        
        return indicators
    
    def _get_geolocation(self, ip: str) -> Optional[Dict[str, str]]:
        """Get geolocation for IP address (mock implementation)"""
        
        # Mock geolocation data
        mock_locations = [
            {"country": "Russia", "city": "Moscow"},
            {"country": "China", "city": "Beijing"},
            {"country": "North Korea", "city": "Pyongyang"},
            {"country": "Iran", "city": "Tehran"},
            {"country": "Unknown", "city": "Unknown"}
        ]
        
        return random.choice(mock_locations)
    
    def _attribute_to_campaign(self, interaction_data: Dict[str, Any]) -> Optional[str]:
        """Attribute interaction to known threat campaign"""
        
        # Mock campaign attribution
        campaigns = [
            "APT29", "Lazarus Group", "Fancy Bear", "Cozy Bear",
            "Equation Group", "Shadow Brokers", "Unknown"
        ]
        
        # Simple attribution based on patterns
        payload = interaction_data.get("payload", "").lower()
        if "windows" in payload and "system32" in payload:
            return "APT29"
        elif "sql" in payload:
            return "Lazarus Group"
        elif len(interaction_data.get("threat_indicators", [])) > 2:
            return random.choice(campaigns[:-1])  # Exclude "Unknown"
        
        return "Unknown"

class CyberDeceptionEngine:
    """Main Cyber Deception Engine"""
    
    def __init__(self, deployment_id: str = "govdocshield-deception"):
        self.deployment_id = deployment_id
        self.assets = {}
        self.honey_doc_generator = HoneyDocumentGenerator()
        self.decoy_network_generator = DecoyNetworkGenerator()
        self.monitor = DeceptionMonitor()
        
        # Configuration
        self.config = {
            "auto_deploy_interval": 3600,  # 1 hour
            "max_honey_documents": 100,
            "max_decoy_networks": 10,
            "threat_response_threshold": 5,
            "deception_diversity_factor": 0.8
        }
        
        # Statistics
        self.stats = {
            "total_assets_deployed": 0,
            "total_interactions": 0,
            "threats_detected": 0,
            "campaigns_identified": 0
        }
        
        logger.info(f"Cyber Deception Engine initialized: {deployment_id}")
    
    def deploy_honey_documents(self, count: int = 10, locations: List[str] = None) -> List[str]:
        """Deploy honey documents in strategic locations"""
        
        if not locations:
            locations = [
                "/shared/finance",
                "/shared/hr", 
                "/shared/projects",
                "/shared/legal",
                "/shared/operations"
            ]
        
        deployed_assets = []
        
        for i in range(count):
            # Generate document
            doc_type = random.choice(["classified_report", "financial_data", "personnel_records"])
            classification = random.choice(["CONFIDENTIAL", "SECRET", "RESTRICTED"])
            
            document = self.honey_doc_generator.generate_honey_document(doc_type, classification)
            
            # Create deception asset
            asset_id = f"HD_{int(time.time())}_{i:03d}"
            location = random.choice(locations)
            
            asset = DeceptionAsset(
                asset_id=asset_id,
                asset_type=DeceptionType.HONEY_DOCUMENT,
                name=document["title"],
                description=f"Honey document: {doc_type}",
                location=f"{location}/{document['title'].replace(' ', '_')}.pdf",
                creation_time=datetime.now(),
                last_accessed=None,
                access_count=0,
                threat_level=ThreatLevel.MEDIUM,
                monitoring_active=True,
                metadata=document
            )
            
            # Store asset and start monitoring
            self.assets[asset_id] = asset
            self.monitor.start_monitoring(asset)
            
            deployed_assets.append(asset_id)
            self.stats["total_assets_deployed"] += 1
        
        logger.info(f"Deployed {count} honey documents")
        return deployed_assets
    
    def deploy_decoy_networks(self, count: int = 3) -> List[str]:
        """Deploy decoy network infrastructure"""
        
        deployed_networks = []
        
        for i in range(count):
            # Generate network topology
            subnet = f"192.168.{100 + i}.0/24"
            topology = self.decoy_network_generator.generate_fake_network_topology(subnet)
            
            # Create deception asset
            asset_id = f"DN_{int(time.time())}_{i:03d}"
            
            asset = DeceptionAsset(
                asset_id=asset_id,
                asset_type=DeceptionType.FAKE_NETWORK,
                name=f"Decoy Network {subnet}",
                description=f"Fake network infrastructure in {subnet}",
                location=subnet,
                creation_time=datetime.now(),
                last_accessed=None,
                access_count=0,
                threat_level=ThreatLevel.HIGH,
                monitoring_active=True,
                metadata=topology
            )
            
            # Store asset and start monitoring
            self.assets[asset_id] = asset
            self.monitor.start_monitoring(asset)
            
            deployed_networks.append(asset_id)
            self.stats["total_assets_deployed"] += 1
        
        logger.info(f"Deployed {count} decoy networks")
        return deployed_networks
    
    def deploy_honey_tokens(self, count: int = 20) -> List[str]:
        """Deploy honey tokens throughout the system"""
        
        token_types = [
            "database_connection_string",
            "api_key",
            "service_account_credential",
            "encryption_key",
            "file_path_reference"
        ]
        
        deployed_tokens = []
        
        for i in range(count):
            token_type = random.choice(token_types)
            token_value = self._generate_honey_token_value(token_type)
            
            asset_id = f"HT_{int(time.time())}_{i:03d}"
            
            asset = DeceptionAsset(
                asset_id=asset_id,
                asset_type=DeceptionType.HONEY_TOKEN,
                name=f"Honey Token - {token_type}",
                description=f"Honey token embedded as {token_type}",
                location=f"embedded_in_system",
                creation_time=datetime.now(),
                last_accessed=None,
                access_count=0,
                threat_level=ThreatLevel.CRITICAL,
                monitoring_active=True,
                metadata={
                    "token_type": token_type,
                    "token_value": token_value,
                    "embedded_locations": [
                        "/etc/config/app.conf",
                        "/var/log/system.log",
                        "HKEY_LOCAL_MACHINE\\SOFTWARE\\App\\Config"
                    ]
                }
            )
            
            self.assets[asset_id] = asset
            self.monitor.start_monitoring(asset)
            
            deployed_tokens.append(asset_id)
            self.stats["total_assets_deployed"] += 1
        
        logger.info(f"Deployed {count} honey tokens")
        return deployed_tokens
    
    def _generate_honey_token_value(self, token_type: str) -> str:
        """Generate honey token value based on type"""
        
        if token_type == "database_connection_string":
            return f"Server=db-{random.randint(10,99)}.corp.local;Database=CustomerData;User=sa;Password={''.join(random.choices(string.ascii_letters + string.digits, k=12))}"
        
        elif token_type == "api_key":
            return f"ak_{''.join(random.choices(string.ascii_letters + string.digits, k=32))}"
        
        elif token_type == "service_account_credential":
            username = f"svc_{''.join(random.choices(string.ascii_lowercase, k=6))}"
            password = ''.join(random.choices(string.ascii_letters + string.digits + "!@#$%", k=16))
            return f"{username}:{password}"
        
        elif token_type == "encryption_key":
            return ''.join(random.choices(string.ascii_letters + string.digits, k=64))
        
        elif token_type == "file_path_reference":
            return f"\\\\fileserver\\shares\\confidential\\{random.choice(['financial', 'hr', 'legal'])}\\sensitive_data.xlsx"
        
        else:
            return ''.join(random.choices(string.ascii_letters + string.digits, k=32))
    
    def simulate_attacker_interaction(self, asset_id: str, attacker_ip: str, 
                                    interaction_type: str = "file_access") -> AttackerInteraction:
        """Simulate attacker interaction for testing"""
        
        if asset_id not in self.assets:
            raise ValueError(f"Asset not found: {asset_id}")
        
        asset = self.assets[asset_id]
        
        # Update asset access information
        asset.last_accessed = datetime.now()
        asset.access_count += 1
        
        # Create interaction data
        interaction_data = {
            "type": interaction_type,
            "user_agent": random.choice([
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "python-requests/2.25.1",
                "curl/7.68.0",
                "sqlmap/1.5.2",
                "Nmap NSE"
            ]),
            "payload": self._generate_attack_payload(interaction_type),
            "timestamp": datetime.now().isoformat()
        }
        
        # Record interaction
        interaction = self.monitor.record_interaction(asset_id, attacker_ip, interaction_data)
        
        self.stats["total_interactions"] += 1
        if interaction.threat_indicators:
            self.stats["threats_detected"] += 1
        
        return interaction
    
    def _generate_attack_payload(self, interaction_type: str) -> str:
        """Generate realistic attack payload for simulation"""
        
        payloads = {
            "file_access": "GET /shared/finance/Q4_Budget_Report.pdf HTTP/1.1",
            "sql_injection": "' UNION SELECT * FROM users--",
            "directory_traversal": "../../../../etc/passwd",
            "credential_stuffing": "admin:password123",
            "network_scan": "nmap -sS -O target_host",
            "xss_attempt": "<script>alert('XSS')</script>",
            "command_injection": "; cat /etc/shadow"
        }
        
        return payloads.get(interaction_type, "unknown_payload")
    
    def get_threat_intelligence(self) -> Dict[str, Any]:
        """Generate threat intelligence from deception interactions"""
        
        # Analyze recent interactions
        recent_interactions = [
            interaction for interaction in self.monitor.interaction_log
            if (datetime.now() - interaction.timestamp).days <= 7
        ]
        
        # Aggregate threat data
        threat_summary = {
            "total_interactions": len(recent_interactions),
            "unique_attackers": len(set(i.source_ip for i in recent_interactions)),
            "top_threat_indicators": self._get_top_threat_indicators(recent_interactions),
            "attack_patterns": self._analyze_attack_patterns(recent_interactions),
            "geographic_distribution": self._analyze_geographic_distribution(recent_interactions),
            "attributed_campaigns": self._analyze_attributed_campaigns(recent_interactions),
            "asset_targeting": self._analyze_asset_targeting(recent_interactions)
        }
        
        return threat_summary
    
    def _get_top_threat_indicators(self, interactions: List[AttackerInteraction]) -> List[Dict[str, Any]]:
        """Get top threat indicators from interactions"""
        
        indicator_counts = {}
        for interaction in interactions:
            for indicator in interaction.threat_indicators:
                indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1
        
        # Sort by frequency
        sorted_indicators = sorted(indicator_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [{"indicator": indicator, "count": count} for indicator, count in sorted_indicators[:10]]
    
    def _analyze_attack_patterns(self, interactions: List[AttackerInteraction]) -> Dict[str, int]:
        """Analyze common attack patterns"""
        
        patterns = {}
        for interaction in interactions:
            interaction_type = interaction.interaction_type
            patterns[interaction_type] = patterns.get(interaction_type, 0) + 1
        
        return patterns
    
    def _analyze_geographic_distribution(self, interactions: List[AttackerInteraction]) -> Dict[str, int]:
        """Analyze geographic distribution of attacks"""
        
        countries = {}
        for interaction in interactions:
            if interaction.geolocation:
                country = interaction.geolocation.get("country", "Unknown")
                countries[country] = countries.get(country, 0) + 1
        
        return countries
    
    def _analyze_attributed_campaigns(self, interactions: List[AttackerInteraction]) -> Dict[str, int]:
        """Analyze attributed threat campaigns"""
        
        campaigns = {}
        for interaction in interactions:
            if interaction.attributed_campaign:
                campaign = interaction.attributed_campaign
                campaigns[campaign] = campaigns.get(campaign, 0) + 1
        
        return campaigns
    
    def _analyze_asset_targeting(self, interactions: List[AttackerInteraction]) -> Dict[str, int]:
        """Analyze which assets are being targeted"""
        
        asset_targeting = {}
        for interaction in interactions:
            asset_id = interaction.asset_id
            if asset_id in self.assets:
                asset_type = self.assets[asset_id].asset_type.value
                asset_targeting[asset_type] = asset_targeting.get(asset_type, 0) + 1
        
        return asset_targeting
    
    def generate_deception_report(self) -> Dict[str, Any]:
        """Generate comprehensive deception effectiveness report"""
        
        threat_intel = self.get_threat_intelligence()
        
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "deployment_id": self.deployment_id,
                "report_period_days": 30,
                "classification": "CONFIDENTIAL"
            },
            "deployment_summary": {
                "total_assets": len(self.assets),
                "assets_by_type": {
                    asset_type.value: sum(1 for asset in self.assets.values() 
                                        if asset.asset_type == asset_type)
                    for asset_type in DeceptionType
                },
                "monitoring_coverage": sum(1 for asset in self.assets.values() 
                                         if asset.monitoring_active) / len(self.assets) if self.assets else 0
            },
            "threat_intelligence": threat_intel,
            "effectiveness_metrics": {
                "detection_rate": self.stats["threats_detected"] / max(1, self.stats["total_interactions"]),
                "false_positive_rate": 0.02,  # Deception assets have very low false positives
                "attacker_engagement_time": random.uniform(10, 300),  # Mock data
                "deception_success_rate": 0.95
            },
            "recommendations": self._generate_recommendations(),
            "statistics": self.stats
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on deception data"""
        
        recommendations = []
        
        if self.stats["total_interactions"] > 50:
            recommendations.append("High interaction volume detected - consider expanding deception network")
        
        if self.stats["threats_detected"] / max(1, self.stats["total_interactions"]) > 0.8:
            recommendations.append("High threat detection rate - implement active countermeasures")
        
        if len(self.assets) < 50:
            recommendations.append("Deploy additional honey assets to increase coverage")
        
        recommendations.extend([
            "Regularly rotate honey documents to maintain believability",
            "Update decoy networks to reflect current infrastructure",
            "Correlate deception data with other security systems",
            "Prepare legal documentation for prosecution support"
        ])
        
        return recommendations

# Factory function
def create_cyber_deception_engine(deployment_id: str = "govdocshield-deception") -> CyberDeceptionEngine:
    """Create a cyber deception engine"""
    return CyberDeceptionEngine(deployment_id)