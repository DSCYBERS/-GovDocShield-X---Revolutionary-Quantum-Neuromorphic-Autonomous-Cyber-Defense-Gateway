#!/usr/bin/env python3
"""
GovDocShield X Enhanced - Air-Gapped System Validator
Validates system compliance for government deployment
"""

import os
import sys
import json
import subprocess
import socket
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class SystemValidator:
    """Comprehensive system validation for government deployment"""
    
    def __init__(self):
        self.validation_results = {
            "validation_timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "compliance_score": 0,
            "critical_failures": [],
            "warnings": [],
            "recommendations": []
        }
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete system validation"""
        
        logger.info("Starting comprehensive system validation...")
        
        # Air-gap validation
        air_gap_results = self.validate_air_gap()
        
        # FIPS compliance validation
        fips_results = self.validate_fips_compliance()
        
        # Hardware security validation
        hardware_results = self.validate_hardware_security()
        
        # Network security validation
        network_results = self.validate_network_security()
        
        # Operating system validation
        os_results = self.validate_operating_system()
        
        # Application security validation
        app_results = self.validate_application_security()
        
        # Compile results
        self.validation_results.update({
            "air_gap_validation": air_gap_results,
            "fips_compliance": fips_results,
            "hardware_security": hardware_results,
            "network_security": network_results,
            "operating_system": os_results,
            "application_security": app_results
        })
        
        # Calculate overall compliance score
        self._calculate_compliance_score()
        
        return self.validation_results
    
    def validate_air_gap(self) -> Dict[str, Any]:
        """Validate air-gapped environment"""
        
        logger.info("Validating air-gapped environment...")
        
        results = {
            "status": "unknown",
            "network_interfaces": [],
            "active_connections": [],
            "wireless_devices": [],
            "external_connectivity": False,
            "compliance_score": 0
        }
        
        try:
            # Check network interfaces
            import psutil
            
            for interface, addrs in psutil.net_if_addrs().items():
                if_stats = psutil.net_if_stats().get(interface)
                
                interface_info = {
                    "name": interface,
                    "is_up": if_stats.isup if if_stats else False,
                    "addresses": []
                }
                
                for addr in addrs:
                    interface_info["addresses"].append({
                        "family": str(addr.family),
                        "address": addr.address,
                        "netmask": getattr(addr, 'netmask', None)
                    })
                
                results["network_interfaces"].append(interface_info)
            
            # Check active connections
            connections = psutil.net_connections(kind='inet')
            for conn in connections:
                if conn.status == 'ESTABLISHED' and conn.raddr:
                    results["active_connections"].append({
                        "local": f"{conn.laddr.ip}:{conn.laddr.port}",
                        "remote": f"{conn.raddr.ip}:{conn.raddr.port}",
                        "status": conn.status,
                        "pid": conn.pid
                    })
            
        except ImportError:
            logger.warning("psutil not available for detailed network analysis")
        
        # Test external connectivity
        external_hosts = [
            ("8.8.8.8", 53),
            ("1.1.1.1", 53),
            ("google.com", 80),
            ("github.com", 443)
        ]
        
        connectivity_count = 0
        for host, port in external_hosts:
            try:
                socket.create_connection((host, port), timeout=3)
                connectivity_count += 1
            except (socket.error, socket.timeout):
                pass
        
        results["external_connectivity"] = connectivity_count > 0
        
        # Check for wireless interfaces
        try:
            wireless_result = subprocess.run(['iwconfig'], capture_output=True, text=True)
            if 'IEEE 802.11' in wireless_result.stdout:
                results["wireless_devices"] = ["wireless_detected"]
                self.validation_results["warnings"].append("Wireless interfaces detected")
        except FileNotFoundError:
            pass
        
        # Determine air-gap status
        if results["external_connectivity"]:
            results["status"] = "compromised"
            results["compliance_score"] = 0
            self.validation_results["critical_failures"].append("External connectivity detected - air-gap compromised")
        elif len(results["active_connections"]) > 5:
            results["status"] = "questionable"
            results["compliance_score"] = 50
            self.validation_results["warnings"].append("High number of active network connections")
        else:
            results["status"] = "compliant"
            results["compliance_score"] = 100
        
        return results
    
    def validate_fips_compliance(self) -> Dict[str, Any]:
        """Validate FIPS 140-2 compliance"""
        
        logger.info("Validating FIPS 140-2 compliance...")
        
        results = {
            "fips_mode_enabled": False,
            "openssl_fips": False,
            "python_crypto_fips": False,
            "approved_algorithms": [],
            "compliance_score": 0
        }
        
        # Check kernel FIPS mode
        try:
            with open('/proc/sys/crypto/fips_enabled', 'r') as f:
                fips_enabled = f.read().strip()
                results["fips_mode_enabled"] = fips_enabled == "1"
        except FileNotFoundError:
            self.validation_results["warnings"].append("Cannot verify kernel FIPS mode")
        
        # Check OpenSSL FIPS
        try:
            openssl_result = subprocess.run(['openssl', 'version'], capture_output=True, text=True)
            results["openssl_fips"] = 'fips' in openssl_result.stdout.lower()
        except FileNotFoundError:
            self.validation_results["critical_failures"].append("OpenSSL not found")
        
        # Check Python cryptography library
        try:
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            
            # Test AES-256
            try:
                cipher = Cipher(algorithms.AES(b'0' * 32), modes.GCM(b'0' * 12), backend=default_backend())
                results["approved_algorithms"].append("AES-256-GCM")
                results["python_crypto_fips"] = True
            except Exception:
                pass
            
        except ImportError:
            self.validation_results["critical_failures"].append("Python cryptography library not available")
        
        # Calculate FIPS compliance score
        score = 0
        if results["fips_mode_enabled"]:
            score += 40
        if results["openssl_fips"]:
            score += 30
        if results["python_crypto_fips"]:
            score += 30
        
        results["compliance_score"] = score
        
        if score < 70:
            self.validation_results["critical_failures"].append("FIPS 140-2 compliance insufficient")
        
        return results
    
    def validate_hardware_security(self) -> Dict[str, Any]:
        """Validate hardware security features"""
        
        logger.info("Validating hardware security features...")
        
        results = {
            "secure_boot": False,
            "tpm_available": False,
            "hsm_detected": False,
            "hardware_rng": False,
            "compliance_score": 0
        }
        
        # Check Secure Boot
        try:
            with open('/sys/firmware/efi/efivars/SecureBoot-8be4df61-93ca-11d2-aa0d-00e098032b8c', 'rb') as f:
                secure_boot_data = f.read()
                results["secure_boot"] = secure_boot_data[-1] == 1
        except (FileNotFoundError, PermissionError):
            self.validation_results["warnings"].append("Cannot verify Secure Boot status")
        
        # Check TPM
        tpm_paths = ['/dev/tpm0', '/dev/tpmrm0']
        for tpm_path in tpm_paths:
            if Path(tpm_path).exists():
                results["tpm_available"] = True
                break
        
        # Check for PKCS#11 HSM
        pkcs11_paths = [
            '/usr/lib/libpkcs11.so',
            '/usr/lib/x86_64-linux-gnu/libpkcs11.so',
            '/opt/nfast/toolkits/pkcs11/libcknfast.so'
        ]
        
        for pkcs11_path in pkcs11_paths:
            if Path(pkcs11_path).exists():
                results["hsm_detected"] = True
                break
        
        # Check hardware RNG
        if Path('/dev/hwrng').exists():
            results["hardware_rng"] = True
        
        # Calculate score
        score = 0
        if results["secure_boot"]:
            score += 25
        if results["tpm_available"]:
            score += 25
        if results["hsm_detected"]:
            score += 30
        if results["hardware_rng"]:
            score += 20
        
        results["compliance_score"] = score
        
        if not results["hsm_detected"]:
            self.validation_results["critical_failures"].append("Hardware Security Module (HSM) not detected")
        
        return results
    
    def validate_network_security(self) -> Dict[str, Any]:
        """Validate network security configuration"""
        
        logger.info("Validating network security configuration...")
        
        results = {
            "firewall_enabled": False,
            "ssh_hardened": False,
            "network_segmentation": False,
            "intrusion_detection": False,
            "compliance_score": 0
        }
        
        # Check firewall
        try:
            ufw_result = subprocess.run(['ufw', 'status'], capture_output=True, text=True)
            results["firewall_enabled"] = 'Status: active' in ufw_result.stdout
        except FileNotFoundError:
            try:
                iptables_result = subprocess.run(['iptables', '-L'], capture_output=True, text=True)
                results["firewall_enabled"] = len(iptables_result.stdout.splitlines()) > 10
            except FileNotFoundError:
                self.validation_results["critical_failures"].append("No firewall detected")
        
        # Check SSH configuration
        ssh_config_path = Path('/etc/ssh/sshd_config')
        if ssh_config_path.exists():
            try:
                with open(ssh_config_path, 'r') as f:
                    ssh_config = f.read()
                    
                    ssh_checks = [
                        'PasswordAuthentication no',
                        'PermitRootLogin no',
                        'Protocol 2'
                    ]
                    
                    hardened_count = sum(1 for check in ssh_checks if check in ssh_config)
                    results["ssh_hardened"] = hardened_count >= 2
                    
            except PermissionError:
                self.validation_results["warnings"].append("Cannot read SSH configuration")
        
        # Check for IDS/IPS
        ids_processes = ['snort', 'suricata', 'ossec', 'aide']
        try:
            ps_result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            for ids_proc in ids_processes:
                if ids_proc in ps_result.stdout:
                    results["intrusion_detection"] = True
                    break
        except Exception:
            pass
        
        # Calculate score
        score = 0
        if results["firewall_enabled"]:
            score += 40
        if results["ssh_hardened"]:
            score += 30
        if results["intrusion_detection"]:
            score += 30
        
        results["compliance_score"] = score
        
        return results
    
    def validate_operating_system(self) -> Dict[str, Any]:
        """Validate operating system security"""
        
        logger.info("Validating operating system security...")
        
        results = {
            "os_version": "",
            "kernel_version": "",
            "selinux_enforcing": False,
            "apparmor_enabled": False,
            "system_hardened": False,
            "compliance_score": 0
        }
        
        # Get OS information
        try:
            with open('/etc/os-release', 'r') as f:
                os_release = f.read()
                for line in os_release.splitlines():
                    if line.startswith('PRETTY_NAME='):
                        results["os_version"] = line.split('=')[1].strip('"')
                        break
        except FileNotFoundError:
            pass
        
        # Get kernel version
        try:
            uname_result = subprocess.run(['uname', '-r'], capture_output=True, text=True)
            results["kernel_version"] = uname_result.stdout.strip()
        except Exception:
            pass
        
        # Check SELinux
        try:
            sestatus_result = subprocess.run(['sestatus'], capture_output=True, text=True)
            results["selinux_enforcing"] = 'Current mode:                   enforcing' in sestatus_result.stdout
        except FileNotFoundError:
            pass
        
        # Check AppArmor
        try:
            aa_status_result = subprocess.run(['aa-status'], capture_output=True, text=True)
            results["apparmor_enabled"] = 'profiles are loaded' in aa_status_result.stdout
        except FileNotFoundError:
            pass
        
        # Check system hardening
        hardening_checks = [
            Path('/etc/security/limits.conf').exists(),
            Path('/etc/sysctl.conf').exists(),
            Path('/etc/audit/auditd.conf').exists()
        ]
        
        results["system_hardened"] = sum(hardening_checks) >= 2
        
        # Calculate score
        score = 0
        if 'fips' in results["kernel_version"].lower():
            score += 30
        if results["selinux_enforcing"] or results["apparmor_enabled"]:
            score += 40
        if results["system_hardened"]:
            score += 30
        
        results["compliance_score"] = score
        
        return results
    
    def validate_application_security(self) -> Dict[str, Any]:
        """Validate application security configuration"""
        
        logger.info("Validating application security configuration...")
        
        results = {
            "govdocshield_installed": False,
            "service_running": False,
            "ssl_configured": False,
            "authentication_enabled": False,
            "audit_logging": False,
            "compliance_score": 0
        }
        
        # Check if GovDocShield is installed
        govdocshield_paths = [
            Path('/opt/govdocshield'),
            Path('/usr/local/govdocshield'),
            Path('./src')  # Development installation
        ]
        
        for path in govdocshield_paths:
            if path.exists():
                results["govdocshield_installed"] = True
                break
        
        # Check if service is running
        try:
            systemctl_result = subprocess.run(['systemctl', 'is-active', 'govdocshield'], 
                                            capture_output=True, text=True)
            results["service_running"] = systemctl_result.stdout.strip() == 'active'
        except FileNotFoundError:
            # Check for process
            try:
                ps_result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
                results["service_running"] = 'govdocshield' in ps_result.stdout or 'deploy_enhanced.py' in ps_result.stdout
            except Exception:
                pass
        
        # Check SSL configuration
        ssl_cert_paths = [
            Path('/etc/govdocshield/server.crt'),
            Path('/opt/govdocshield/ssl/server.crt'),
            Path('./ssl/server.crt')
        ]
        
        for cert_path in ssl_cert_paths:
            if cert_path.exists():
                results["ssl_configured"] = True
                break
        
        # Check configuration files
        config_paths = [
            Path('/etc/govdocshield/config.json'),
            Path('/opt/govdocshield/config/config.json'),
            Path('./config/config.json')
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        
                        auth_config = config.get('authentication', {})
                        results["authentication_enabled"] = auth_config.get('enabled', False)
                        
                        audit_config = config.get('audit', {})
                        results["audit_logging"] = audit_config.get('enabled', False)
                        
                        break
                except (json.JSONDecodeError, PermissionError):
                    pass
        
        # Check log files
        log_paths = [
            Path('/var/log/govdocshield'),
            Path('/opt/govdocshield/logs'),
            Path('./logs')
        ]
        
        for log_path in log_paths:
            if log_path.exists() and any(log_path.iterdir()):
                results["audit_logging"] = True
                break
        
        # Calculate score
        score = 0
        if results["govdocshield_installed"]:
            score += 20
        if results["service_running"]:
            score += 20
        if results["ssl_configured"]:
            score += 20
        if results["authentication_enabled"]:
            score += 20
        if results["audit_logging"]:
            score += 20
        
        results["compliance_score"] = score
        
        return results
    
    def _calculate_compliance_score(self):
        """Calculate overall compliance score"""
        
        # Weight different validation areas
        weights = {
            "air_gap_validation": 0.25,
            "fips_compliance": 0.25,
            "hardware_security": 0.20,
            "network_security": 0.15,
            "operating_system": 0.10,
            "application_security": 0.05
        }
        
        total_score = 0
        for area, weight in weights.items():
            area_data = self.validation_results.get(area, {})
            area_score = area_data.get("compliance_score", 0)
            total_score += area_score * weight
        
        self.validation_results["compliance_score"] = int(total_score)
        
        # Determine overall status
        if total_score >= 90:
            self.validation_results["overall_status"] = "compliant"
        elif total_score >= 70:
            self.validation_results["overall_status"] = "partially_compliant"
        else:
            self.validation_results["overall_status"] = "non_compliant"
        
        # Add recommendations based on score
        if total_score < 90:
            self.validation_results["recommendations"].extend([
                "Review and address all critical failures",
                "Implement missing security controls",
                "Complete FIPS 140-2 compliance requirements",
                "Ensure proper air-gap configuration"
            ])

def main():
    """Main validation function"""
    
    print("=" * 80)
    print("GovDocShield X Enhanced - System Validation")
    print("Government Air-Gapped Deployment Compliance Check")
    print("=" * 80)
    
    validator = SystemValidator()
    
    try:
        results = validator.run_full_validation()
        
        print(f"\n📊 VALIDATION RESULTS")
        print(f"Timestamp: {results['validation_timestamp']}")
        print(f"Overall Status: {results['overall_status'].upper()}")
        print(f"Compliance Score: {results['compliance_score']}/100")
        
        # Display area scores
        print(f"\n📋 DETAILED SCORES:")
        areas = [
            ("Air-Gap Validation", "air_gap_validation"),
            ("FIPS 140-2 Compliance", "fips_compliance"),
            ("Hardware Security", "hardware_security"),
            ("Network Security", "network_security"),
            ("Operating System", "operating_system"),
            ("Application Security", "application_security")
        ]
        
        for area_name, area_key in areas:
            area_data = results.get(area_key, {})
            score = area_data.get("compliance_score", 0)
            status = "✅" if score >= 70 else "❌"
            print(f"  {status} {area_name}: {score}/100")
        
        # Display critical failures
        if results["critical_failures"]:
            print(f"\n🚨 CRITICAL FAILURES:")
            for failure in results["critical_failures"]:
                print(f"  ❌ {failure}")
        
        # Display warnings
        if results["warnings"]:
            print(f"\n⚠️  WARNINGS:")
            for warning in results["warnings"]:
                print(f"  ⚠️  {warning}")
        
        # Display recommendations
        if results["recommendations"]:
            print(f"\n💡 RECOMMENDATIONS:")
            for recommendation in results["recommendations"]:
                print(f"  💡 {recommendation}")
        
        # Save detailed results
        results_file = f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        # Exit code based on compliance
        if results["overall_status"] == "compliant":
            print(f"\n✅ SYSTEM READY FOR GOVERNMENT DEPLOYMENT")
            sys.exit(0)
        elif results["overall_status"] == "partially_compliant":
            print(f"\n⚠️  SYSTEM REQUIRES REMEDIATION BEFORE DEPLOYMENT")
            sys.exit(1)
        else:
            print(f"\n❌ SYSTEM NOT SUITABLE FOR GOVERNMENT DEPLOYMENT")
            sys.exit(2)
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        print(f"\n❌ VALIDATION FAILED: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()