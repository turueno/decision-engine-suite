import pytest
import numpy as np
from decision_engine.ahp import AHPEngine

def test_ahp_weights():
    # Example matrix (3x3)
    # C1 vs C2 = 3
    # C1 vs C3 = 5
    # C2 vs C3 = 2
    matrix = [
        [1, 3, 5],
        [1/3, 1, 2],
        [1/5, 1/2, 1]
    ]
    
    ahp = AHPEngine(matrix)
    results = ahp.get_results()
    weights = list(results['weights'].values())
    
    # Expected weights approx: 0.648, 0.230, 0.122
    assert pytest.approx(weights[0], 0.01) == 0.648
    assert pytest.approx(weights[1], 0.01) == 0.230
    assert pytest.approx(weights[2], 0.01) == 0.122
    
    assert results['is_consistent'] == True

def test_ahp_inconsistent():
    # Inconsistent matrix
    # A > B, B > C, but C > A (extreme case)
    matrix = [
        [1, 5, 1/5],
        [1/5, 1, 5],
        [5, 1/5, 1]
    ]
    ahp = AHPEngine(matrix)
    results = ahp.get_results()
    
    # Should be inconsistent
    assert results['is_consistent'] == False
