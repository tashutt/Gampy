from setuptools import setup, find_packages

setup(
    name="Gampy",
    version="0.1.0",
    packages=find_packages(),
    description="GammaTPC simulation and analysis tools",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "joblib",
        "scikit-learn",
        "scipy",
        "awkward",
        "tqdm",
        "PyYAML",
        "astropy",
        "h5py",
    ],
)
