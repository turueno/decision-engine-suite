import pytest
import numpy as np
from decision_engine.group_ahp import GroupDecisionEngine

def test_weighted_ahp_aggregation():
    # Judge 1: A is 3 times B
    m1 = np.array([
        [1.0, 3.0],
        [1/3.0, 1.0]
    ])
    # Judge 2: A is 9 times B
    m2 = np.array([
        [1.0, 9.0],
        [1/9.0, 1.0]
    ])
    
    # 1. Equal weights (50% / 50%)
    engine_equal = GroupDecisionEngine(matrices=[m1, m2], criteria_names=["A", "B"])
    res_equal = engine_equal.aggregate_pairwise_matrices()
    # Expected: (3^0.5) * (9^0.5) = sqrt(27) ≈ 5.196
    assert np.isclose(res_equal[0, 1], np.sqrt(27))
    assert np.isclose(res_equal[1, 0], 1.0 / np.sqrt(27))

    # 2. Asymmetric weights (Judge 1: 100%, Judge 2: 0%)
    engine_w1 = GroupDecisionEngine(matrices=[m1, m2], criteria_names=["A", "B"], judge_weights=[1.0, 0.0])
    res_w1 = engine_w1.aggregate_pairwise_matrices()
    assert np.isclose(res_w1[0, 1], 3.0)
    assert np.isclose(res_w1[1, 0], 1/3.0)

    # 3. Custom weights (Judge 1: 25%, Judge 2: 75%)
    # Expected: (3^0.25) * (9^0.75) = 3^0.25 * 3^1.5 = 3^1.75 ≈ 6.8385
    engine_custom = GroupDecisionEngine(matrices=[m1, m2], criteria_names=["A", "B"], judge_weights=[0.25, 0.75])
    res_custom = engine_custom.aggregate_pairwise_matrices()
    assert np.isclose(res_custom[0, 1], 3.0 ** 1.75)

def test_weighted_decision_matrix_aggregation():
    dm1 = np.array([
        [10.0, 20.0],
        [30.0, 40.0]
    ])
    dm2 = np.array([
        [100.0, 200.0],
        [300.0, 400.0]
    ])
    
    # 80% Judge 1, 20% Judge 2
    # [0,0] = 0.8 * 10 + 0.2 * 100 = 8 + 20 = 28
    engine = GroupDecisionEngine(decision_matrices=[dm1, dm2], judge_weights=[0.8, 0.2])
    agg_dm = engine.aggregate_decision_matrices()
    
    assert np.isclose(agg_dm[0, 0], 28.0)
    assert np.isclose(agg_dm[1, 1], 0.8 * 40.0 + 0.2 * 400.0)
