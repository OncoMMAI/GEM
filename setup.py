# -*- coding: utf-8 -*-

"""
Setup script for glioma embedding package.
"""

from setuptools import setup, find_packages
import os

# Read requirements from file
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f.readlines() if not line.startswith("#")]

# Read README for long description
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="glioma_embedding",
    version="0.1.0",
    description="Fine-tuning JINA embeddings for glioma classification and survival prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Benedict Neo",
    author_email="your.email@example.com",
    url="https://github.com/benedictneo/glioma_embedding",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": ["glioma=glioma:main"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords="nlp, embeddings, medical, glioma, classification, jina, radiology, pathology, survival prediction, mgmt prediction",
)
