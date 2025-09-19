"""
GovDocShield X - Autonomous Cyber Defense Gateway
Setup and installation configuration
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="govdocshield-x",
    version="1.0.0-alpha",
    author="GovDocShield Development Team",
    author_email="dev@govdocshield.mil",
    description="Autonomous Cyber Defense Gateway combining quantum, neuromorphic, and bio-inspired technologies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/govdocshield/govdocshield-x",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Government",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Filesystems",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "quantum": [
            "qiskit[all]>=0.45.0",
            "pennylane[all]>=0.32.0",
            "cirq[contrib]>=1.3.0",
        ],
        "neuromorphic": [
            "norse>=1.0.0",
            "spikingjelly>=0.0.0.0.14",
            "bindsnet>=0.3.0",
        ],
        "bio-inspired": [
            "deap>=1.4.0",
            "scikit-opt>=0.6.0",
            "pymoo>=0.6.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.10.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
            "coverage>=7.3.0",
        ],
        "all": [
            "qiskit[all]>=0.45.0",
            "pennylane[all]>=0.32.0",
            "norse>=1.0.0",
            "deap>=1.4.0",
            "pytest>=7.4.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "govdocshield=cli.main:main",
            "gds=cli.main:main",
            "govdocshield-quantum=cli.quantum:main",
            "govdocshield-neuromorphic=cli.neuromorphic:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.proto", "*.md"],
        "shared": ["data/*", "models/*", "configs/*"],
        "infrastructure": ["k8s/*", "docker/*", "helm/*"],
    },
    zip_safe=False,
    keywords="cybersecurity quantum neuromorphic government defense",
    project_urls={
        "Bug Reports": "https://github.com/govdocshield/govdocshield-x/issues",
        "Source": "https://github.com/govdocshield/govdocshield-x",
        "Documentation": "https://docs.govdocshield.mil",
    },
)