import pytest
import numpy as np
from decision_engine.group_ahp import GroupAHPEngine

def test_group_aggregation():
    # Judge 1: A is much better than B (3)
    m1 = np.array([
        [1, 3],
        [1/3, 1]
    ])
    
    # Judge 2: A is moderately better than B (5)
    m2 = np.array([
        [1, 5],
        [1/5, 1]
    ])
    
    # Geometric Mean:
    # A vs B: (3 * 5)^(1/2) = sqrt(15) ≈ 3.87
    # B vs A: (1/3 * 1/5)^(1/2) = sqrt(1/15) ≈ 0.258
    
    engine = GroupAHPEngine([m1, m2], ["A", "B"])
    results = engine.get_results()
    
    agg_matrix = results['aggregated_matrix']
    
    assert np.isclose(agg_matrix[0, 1], np.sqrt(15))
    assert np.isclose(agg_matrix[1, 0], np.sqrt(1/15))
    
    # Weights should favor A
    weights = results['weights']
    assert weights['A'] > weights['B']
