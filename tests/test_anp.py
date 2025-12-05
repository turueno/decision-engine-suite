import pytest
import numpy as np
from decision_engine.anp import ANPEngine

def test_anp_convergence():
    clusters = ["C1"]
    nodes = {"C1": ["A", "B", "C"]}
    connections = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "A")]
    
    engine = ANPEngine(clusters, nodes, connections)
    
    # Define a known stochastic supermatrix
    #       A    B    C
    # A   0.0  0.0  1.0
    # B   0.5  0.0  0.0
    # C   0.5  1.0  0.0
    
    matrix = np.array([
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ])
    
    engine.set_unweighted_supermatrix(matrix)
    
    limit = engine.calculate_limit_matrix()
    
    # Check if columns are identical (steady state)
    col0 = limit[:, 0]
    col1 = limit[:, 1]
    assert np.allclose(col0, col1)
    
    # Check priorities sum to 1
    priorities = engine.get_priorities()
    total_p = sum(priorities.values())
    assert np.isclose(total_p, 1.0)
    
    # Expected values (approximate for this cyclic matrix)
    # A = 0.4, B = 0.2, C = 0.4 (Stationary distribution of Markov chain)
    # Let's check roughly
    assert np.isclose(priorities["A"], 0.4, atol=0.01)
    assert np.isclose(priorities["B"], 0.2, atol=0.01)
    assert np.isclose(priorities["C"], 0.4, atol=0.01)
