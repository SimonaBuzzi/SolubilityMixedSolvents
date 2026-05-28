
import os
import sys

# Step 1: Define the absolute path to the 'solvation_predictor' directory
# Update this path to point directly to the location of 'solvation_predictor'
SOLVATION_PREDICTOR_DIR = '/Users/u0171107/Documents/SolProp/Git_SolProp/SolProp_ML/solvation_predictor'
""" Define the absolute path to solvation_predictor:
You can explicitly set SOLVATION_PREDICTOR_DIR to the absolute path of the solvation_predictor directory. 
In this case, it’s /Users/u0171107/Documents/SolProp/Git_SolProp/SolProp_ML/solvation_predictor."""

# Step 2: Change the current working directory to 'solvation_predictor'
os.chdir(SOLVATION_PREDICTOR_DIR)
""" This line changes the directory to the directory of solvation_predictor"""
# Step 3: Add 'solvation_predictor' to sys.path so the imports work
sys.path.append(SOLVATION_PREDICTOR_DIR)


# Step 4: Ensure 'trained_models' directory exists (mock it for testing)
TRAINED_MODELS_DIR = os.path.join(SOLVATION_PREDICTOR_DIR, 'trained_models')
if not os.path.exists(TRAINED_MODELS_DIR):
    os.makedirs(TRAINED_MODELS_DIR)

# Step 5: Mock the 'Solutev2' directory (as an example)
MOCK_SOLUTEV2_DIR = os.path.join(TRAINED_MODELS_DIR, 'Solutev2')
if not os.path.exists(MOCK_SOLUTEV2_DIR):
    os.makedirs(MOCK_SOLUTEV2_DIR)

from solvation_predictor.solubility.SolubilityModels import SolubilityModels


def test_aq_model_loading():
    instance = SolubilityModels(
        reduced_number=None,  # Adjust this based on your test case
        load_ghsolv=False,  # Load solvation free energy models
        load_g=False,  # Load aqueous solubility models
        load_h=False,  # Do not load solvation enthalpy models for this test
        load_saq=True,  # Load aqueous solubility models
        load_solute=False,  # Load solute models
        logger=None,  # Optionally, provide a logger if needed
        verbose=False  # Adjust based on whether you want to see verbose output
    )
    assert instance.saq_models is not None, "SAQ models should be loaded"
   

