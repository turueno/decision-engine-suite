import pytest
import numpy as np
from decision_engine.promethee import PrometheeEngine

def test_promethee_preference_func():
    engine = PrometheeEngine([[1]], [1], ['+'], ['Usual'])
    
    # Usual: 1 if d > 0, else 0
    assert engine._preference_func(5, 'Usual', 0, 0) == 1.0
    assert engine._preference_func(0, 'Usual', 0, 0) == 0.0
    assert engine._preference_func(-5, 'Usual', 0, 0) == 0.0
    
    # Linear: d/p if d <= p, 1 if d > p
    assert engine._preference_func(5, 'Linear', 10, 0) == 0.5
    assert engine._preference_func(15, 'Linear', 10, 0) == 1.0
    
    # Linear (q, p): 0 if d<=q, linear, 1 if d>p
    assert engine._preference_func(2, 'Linear (q, p)', 10, 5) == 0.0 # d < q
    assert engine._preference_func(7.5, 'Linear (q, p)', 10, 5) == 0.5 # mid
    assert engine._preference_func(12, 'Linear (q, p)', 10, 5) == 1.0 # d > p

def test_promethee_ranking():
    # Simple case: A1 > A2
    # C1 (Weight 1)
    # A1: 10
    # A2: 5
    # Usual preference
    
    data = [[10], [5]]
    weights = [1]
    impacts = ['+']
    funcs = ['Usual']
    
    promethee = PrometheeEngine(data, weights, impacts, funcs)
    results = promethee.calculate_flows()
    
    # A1 vs A2: diff = 5 > 0 -> P(A1, A2) = 1
    # A2 vs A1: diff = -5 <= 0 -> P(A2, A1) = 0
    
    # Phi+(A1) = 1 / 1 = 1
    # Phi-(A1) = 0 / 1 = 0
    # Net(A1) = 1
    
    # Phi+(A2) = 0
    # Phi-(A2) = 1
    # Net(A2) = -1
    
    assert results.iloc[0]['Alternative'] == 'A1'
    assert results.iloc[0]['Net Phi'] == 1.0
    assert results.iloc[1]['Alternative'] == 'A2'
    assert results.iloc[1]['Net Phi'] == -1.0
