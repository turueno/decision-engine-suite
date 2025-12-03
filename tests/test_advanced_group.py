import pytest
import numpy as np
from decision_engine.group_ahp import GroupDecisionEngine

def test_fuzzy_aggregation():
    # Fuzzy Matrix 1: A vs B is (1, 3, 5)
    m1 = np.empty((2, 2), dtype=object)
    m1[0, 0] = (1, 1, 1)
    m1[1, 1] = (1, 1, 1)
    m1[0, 1] = (1, 3, 5)
    m1[1, 0] = (1/5, 1/3, 1)
    
    # Fuzzy Matrix 2: A vs B is (2, 4, 6)
    m2 = np.empty((2, 2), dtype=object)
    m2[0, 0] = (1, 1, 1)
    m2[1, 1] = (1, 1, 1)
    m2[0, 1] = (2, 4, 6)
    m2[1, 0] = (1/6, 1/4, 1/2)
    
    engine = GroupDecisionEngine(matrices=[m1, m2], criteria_names=["A", "B"])
    agg_matrix = engine.aggregate_pairwise_matrices()
    
    # Check Geometric Mean of A vs B
    # L: sqrt(1 * 2) = 1.414
    # M: sqrt(3 * 4) = 3.464
    # U: sqrt(5 * 6) = 5.477
    
    val = agg_matrix[0, 1]
    assert np.isclose(val[0], np.sqrt(2))
    assert np.isclose(val[1], np.sqrt(12))
    assert np.isclose(val[2], np.sqrt(30))

def test_decision_matrix_aggregation():
    # Matrix 1 (Judge 1)
    dm1 = np.array([
        [10, 20],
        [30, 40]
    ])
    
    # Matrix 2 (Judge 2)
    dm2 = np.array([
        [20, 30],
        [40, 50]
    ])
    
    engine = GroupDecisionEngine(decision_matrices=[dm1, dm2])
    agg_dm = engine.aggregate_decision_matrices()
    
    # Check Arithmetic Mean
    # [0,0]: (10+20)/2 = 15
    # [1,1]: (40+50)/2 = 45
    
    assert agg_dm[0, 0] == 15
    assert agg_dm[1, 1] == 45
