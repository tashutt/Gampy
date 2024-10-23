#!/usr/bin/env python

import setuptools

VER = "0.1"

reqs = ["numpy",
        "awkward",
        "astropy",
        "scipy",
        "pandas",
        "seaborn",
        "joblib",
        "scikit-learn",
        "matplotlib",
        "pyproj",
        ]

setuptools.setup(
    name="gampy",
    version=VER,
    author="SLAC National Accelerator Laboratory",
    author_email="tshutt@slac.stanford.edu",
    description="Gampix Readout Simulation",
    url="https://github.com/tashutt/Gampy",
    packages=setuptools.find_packages(),
    install_requires=reqs,
    classifiers=[
        "Development Status :: 1 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    python_requires='>=3.2',
)
