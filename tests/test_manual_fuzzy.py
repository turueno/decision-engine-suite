import pytest
import numpy as np
from decision_engine.fuzzy_ahp import FuzzyAHPEngine

def test_manual_fuzzy_input():
    # Manual TFN Matrix for 2 criteria
    # C1 vs C2 = (2, 3, 4)
    # C2 vs C1 = (1/4, 1/3, 1/2)
    
    matrix = np.zeros((2, 2, 3))
    matrix[0, 0] = (1, 1, 1)
    matrix[1, 1] = (1, 1, 1)
    
    matrix[0, 1] = (2, 3, 4)
    matrix[1, 0] = (1/4, 1/3, 1/2)
    
    criteria = ["C1", "C2"]
    
    engine = FuzzyAHPEngine(matrix, criteria, input_type="fuzzy")
    results = engine.get_results()
    
    weights = results['weights']
    
    # Check if weights sum to 1
    assert np.isclose(sum(weights.values()), 1.0)
    
    # C1 should be more important than C2
    assert weights['C1'] > weights['C2']
    
    # Approximate check
    # GeoMean C1: (1*2)^(1/2), (1*3)^(1/2), (1*4)^(1/2) -> (1.41, 1.73, 2.0)
    # GeoMean C2: (1*0.25)^(1/2), (1*0.33)^(1/2), (1*0.5)^(1/2) -> (0.5, 0.57, 0.707)
    # Sum: (1.91, 2.3, 2.707)
    # InvSum: (0.369, 0.434, 0.523)
    # FuzzyWeight C1: (1.41*0.369, ..., 2.0*0.523) -> (0.52, ..., 1.04)
    # FuzzyWeight C2: (0.5*0.369, ..., 0.707*0.523) -> (0.18, ..., 0.37)
    
    # Just ensure it runs and produces valid output
    assert isinstance(weights['C1'], float)
