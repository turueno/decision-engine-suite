import pytest
import numpy as np
from decision_engine.fuzzy_ahp import FuzzyAHPEngine

def test_fuzzy_ahp_weights():
    # Example matrix (3x3)
    # Same as crisp example:
    # C1 vs C2 = 3
    # C1 vs C3 = 5
    # C2 vs C3 = 2
    matrix = [
        [1, 3, 5],
        [1/3, 1, 2],
        [1/5, 1/2, 1]
    ]
    
    fahp = FuzzyAHPEngine(matrix)
    results = fahp.get_results()
    weights = list(results['weights'].values())
    
    # Fuzzy weights should be somewhat close to crisp weights but not identical
    # Crisp: 0.648, 0.230, 0.122
    
    assert len(weights) == 3
    assert pytest.approx(sum(weights), 0.001) == 1.0
    
    # Check order is preserved
    assert weights[0] > weights[1]
    assert weights[1] > weights[2]

def test_fuzzy_tfn_conversion():
    fahp = FuzzyAHPEngine([[1]])
    
    # 1 -> (1, 1, 1)
    assert fahp._get_tfn(1) == (1, 1, 1)
    
    # 3 -> (2, 3, 4)
    assert fahp._get_tfn(3) == (2, 3, 4)
    
    # 1/3 -> (1/4, 1/3, 1/2)
    tfn = fahp._get_tfn(1/3)
    assert pytest.approx(tfn[0]) == 0.25
    assert pytest.approx(tfn[1]) == 0.333333
    assert pytest.approx(tfn[2]) == 0.5
