import numpy as np
from .ahp import AHPEngine

class GroupAHPEngine:
    def __init__(self, matrices, criteria_names=None):
        """
        Initialize Group AHP Engine.
        
        Args:
            matrices (list of np.array): List of pairwise comparison matrices from judges.
            criteria_names (list): Names of criteria.
        """
        self.matrices = [np.array(m, dtype=float) for m in matrices]
        self.num_judges = len(matrices)
        self.n = self.matrices[0].shape[0]
        self.criteria_names = criteria_names if criteria_names else [f"C{i+1}" for i in range(self.n)]
        
        # Validate dimensions
        for m in self.matrices:
            if m.shape != (self.n, self.n):
                raise ValueError("All matrices must have the same dimensions.")
                
        self.aggregated_matrix = self._aggregate_matrices()
        self.weights = None
        self.consensus_ratio = None # Future feature?

    def _aggregate_matrices(self):
        """
        Aggregate individual matrices using Geometric Mean (AIJ).
        a_ij_group = (prod(a_ij_k))^(1/K)
        """
        agg_matrix = np.ones((self.n, self.n))
        
        for i in range(self.n):
            for j in range(self.n):
                product = 1.0
                for k in range(self.num_judges):
                    product *= self.matrices[k][i, j]
                
                agg_matrix[i, j] = product ** (1.0 / self.num_judges)
                
        return agg_matrix

    def calculate_weights(self):
        """
        Calculate weights based on the aggregated matrix using standard AHP.
        """
        # Use the standard AHP engine on the aggregated matrix
        engine = AHPEngine(self.aggregated_matrix, self.criteria_names)
        results = engine.get_results()
        self.weights = results['weights']
        self.consistency_ratio = results['consistency_ratio']
        self.is_consistent = results['is_consistent']
        
        return self.weights

    def get_results(self):
        if self.weights is None:
            self.calculate_weights()
            
        return {
            "weights": self.weights,
            "aggregated_matrix": self.aggregated_matrix,
            "consistency_ratio": self.consistency_ratio,
            "is_consistent": self.is_consistent,
            "num_judges": self.num_judges,
            "method": "Group AHP (Geometric Mean Aggregation)"
        }
