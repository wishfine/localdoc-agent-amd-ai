"""Package metadata for localdoc-agent-amd-ai."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="localdoc-agent-amd-ai",
    version="0.1.0",
    author="LocalDoc Agent Team",
    author_email="team@localdoc-agent.com",
    description="A local document agent with AMD AI hardware acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/localdoc-agent/localdoc-agent-amd-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "flake8>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "localdoc-agent=localdoc.app:main",
        ],
    },
)
