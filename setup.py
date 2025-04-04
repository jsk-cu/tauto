from setuptools import setup, find_packages

setup(
    name="tauto",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "pyyaml>=6.0",
        "wandb>=0.15.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.65.0",
        "click>=8.1.0",
        "optuna>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.10.0",
            "mypy>=1.0.0",
            "ruff>=0.0.200",
        ]
    },
    entry_points={
        "console_scripts": [
            "tauto=tauto.cli:main",
        ],
    },
    author="HPML Team",
    author_email="example@columbia.edu",
    description="TAuto: An AutoML optimization suite for PyTorch models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jsk-cu/tauto",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)