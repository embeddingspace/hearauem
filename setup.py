#!/usr/bin/env python3
from setuptools import find_packages, setup

long_description = open("README.md", "r").read()

setup(
    name="hearauem",
    description="Holistic Evaluation of Audio Representations (HEAR) 2021 -- Auem Submission",
    author="Christopher Jacoby & Constantinos Dimitriou",
#    author_email="TODO",
#    url="https://github.com/foobar",
    license="Apache-2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
#    project_urls={
#        "Bug Tracker": "https://github.com/neuralaudio/hear-baseline/issues",
#        "Source Code": "https://github.com/neuralaudio/hear-baseline",
#    },
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.7",
    install_requires=[
        "librosa",
        "numpy",
        "torch>=1.10",
        "nnAudio"
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-cov",
            "pytest-env",
        ],
        "dev": [
            "pre-commit",
            "black",  # Used in pre-commit hooks
            "pytest",
            "pytest-cov",
            "pytest-env",
        ],
    },
)
