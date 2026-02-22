from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="medsecure",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Automated Adversarial Robustness Testing for Medical AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/medsecure",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "medsecure-train=experiments.train_agent:main",
            "medsecure-eval=experiments.evaluate:main",
            "medsecure-baseline=experiments.run_baselines:main",
        ],
    },
)
