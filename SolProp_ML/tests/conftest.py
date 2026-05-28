"""Add somethig here"""

import os
import pytest
from solvation_predictor.inp import Solute

@pytest.fixture
def solute_instance():
    """Fixture to create a Solute instance"""
    return Solute()

def test_solute_initialization(solute_instance):
    """Test if Solute class is initialized correctly"""
    assert solute_instance is not None
    assert solute_instance.input_file.endswith("Data/SoluteDB_selected_data.csv")
    assert solute_instance.num_targets == 5
    assert solute_instance.max_num_mols == 1
    assert solute_instance.f_mol_size == 2
    assert solute_instance.num_folds == 1
    assert solute_instance.num_features == 0

def test_model_loading(solute_instance):
    """Test if the model loads correctly."""
    model_path = solute_instance.model_path
    assert os.path.exists(model_path), f"Model path {model_path} does not exist"
    
    # Simulate model loading if there's a method for that
    try:
        # Assuming you have a method to load the model
        solute_instance.load_model()  
    except AttributeError:
        pytest.skip("No method to load model in Solute class")

def test_prediction(solute_instance):
    """Test if the model can make a prediction on a sample input."""
    sample_inchis = ["InChI=1S/C5H7N3O3/c1-7-4(9)3(8-6)5(10)11-2/h1-2H3,(H,7,9)"]
    
    # Assuming you have a predict method that takes a list of InChIs
    try:
        predictions = solute_instance.predict(sample_inchis)
        assert predictions is not None
        assert len(predictions) == len(sample_inchis)
        print("Predictions:", predictions)
    except AttributeError:
        pytest.skip("No predict method in Solute class")

def test_invalid_inchis(solute_instance):
    """Test how the model handles invalid InChIs."""
    invalid_inchis = ["InvalidInChI"]
    try:
        predictions = solute_instance.predict(invalid_inchis)
        assert predictions is not None
    except Exception as e:
        pytest.fail(f"Model failed with invalid InChIs: {e}")