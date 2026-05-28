from distutils.core import setup
from setuptools import setup

setup(
    name="solvation_predictor",
    version="1.1.2",
    packages=[
        "solvation_predictor",
        "solvation_predictor.data",
        "solvation_predictor.train",
        "solvation_predictor.models",
        "solvation_predictor.features",
        "solvation_predictor.solubility",
    ],
    package_data={
        "solvation_predictor": ["trained_models/*/*.pt", "solubility/*.json"]
    },
    url="https://gitlab.kuleuven.be/creas/vermeiregroup/SolProp_ML",
    license="KU Leuven",
    author="Simona Buzzi, Roel Leenhouts, Florence Vermeire",
    author_email="simona.buzzi@kuleuven.be",
    description="Package to make solubility predictions in monosolvents and mixtures with ML",
)