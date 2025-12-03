import numpy as np
from .ahp import AHPEngine
from .fuzzy_ahp import FuzzyAHPEngine

class GroupDecisionEngine:
    def __init__(self, matrices=None, decision_matrices=None, criteria_names=None, alternatives=None):
        """
        Initialize Group Decision Engine.
        
        Args:
            matrices (list): List of pairwise comparison matrices (crisp or fuzzy).
            decision_matrices (list): List of decision matrices (alternatives x criteria).
            criteria_names (list): Names of criteria.
            alternatives (list): Names of alternatives.
        """
        self.matrices = matrices
        self.decision_matrices = decision_matrices
        self.criteria_names = criteria_names
        self.alternatives = alternatives
        
        self.aggregated_matrix = None
        self.aggregated_decision_matrix = None
        self.weights = None
        self.consistency_ratio = None
        self.is_consistent = None
        self.is_fuzzy = False

    def aggregate_pairwise_matrices(self):
        """
        Aggregate individual pairwise matrices using Geometric Mean (AIJ).
        Supports both Crisp (float) and Fuzzy (tuple/list) matrices.
        """
        if not self.matrices:
            return None
            
        num_judges = len(self.matrices)
        n = len(self.matrices[0])
        
        # Check if fuzzy (first element of first matrix is a list/tuple/array of size 3)
        first_val = self.matrices[0][0][1] # Check off-diagonal
        if isinstance(first_val, (list, tuple, np.ndarray)) and len(first_val) == 3:
            self.is_fuzzy = True
            agg_matrix = np.empty((n, n), dtype=object)
            
            for i in range(n):
                for j in range(n):
                    l_prod, m_prod, u_prod = 1.0, 1.0, 1.0
                    for k in range(num_judges):
                        val = self.matrices[k][i][j]
                        l_prod *= val[0]
                        m_prod *= val[1]
                        u_prod *= val[2]
                    
                    agg_matrix[i, j] = (
                        l_prod ** (1.0 / num_judges),
                        m_prod ** (1.0 / num_judges),
                        u_prod ** (1.0 / num_judges)
                    )
        else:
            self.is_fuzzy = False
            agg_matrix = np.ones((n, n))
            for i in range(n):
                for j in range(n):
                    product = 1.0
                    for k in range(num_judges):
                        product *= self.matrices[k][i][j]
                    agg_matrix[i, j] = product ** (1.0 / num_judges)
                    
        self.aggregated_matrix = agg_matrix
        return agg_matrix

    def aggregate_decision_matrices(self):
        """
        Aggregate decision matrices using Arithmetic Mean.
        """
        if not self.decision_matrices:
            return None
            
        # Stack matrices along a new axis and take the mean
        # Assuming decision_matrices are already numpy arrays or compatible lists
        stacked = np.array(self.decision_matrices, dtype=float)
        self.aggregated_decision_matrix = np.mean(stacked, axis=0)
        
        return self.aggregated_decision_matrix

    def calculate_weights(self):
        """
        Calculate weights based on the aggregated matrix.
        """
        if self.aggregated_matrix is None:
            self.aggregate_pairwise_matrices()
            
        if self.is_fuzzy:
            engine = FuzzyAHPEngine(self.aggregated_matrix, self.criteria_names, input_type="fuzzy")
            results = engine.get_results()
            self.weights = results['weights']
            self.consistency_ratio = results['consistency_ratio']
            self.is_consistent = results['is_consistent']
        else:
            engine = AHPEngine(self.aggregated_matrix, self.criteria_names)
            results = engine.get_results()
            self.weights = results['weights']
            self.consistency_ratio = results['consistency_ratio']
            self.is_consistent = results['is_consistent']
        
        return self.weights

    def get_results(self):
        results = {
            "method": "Group Decision Aggregator",
            "num_judges_pairwise": len(self.matrices) if self.matrices else 0,
            "num_judges_decision": len(self.decision_matrices) if self.decision_matrices else 0,
        }
        
        if self.matrices:
            if self.weights is None:
                self.calculate_weights()
            results["weights"] = self.weights
            results["aggregated_matrix"] = self.aggregated_matrix
            results["consistency_ratio"] = self.consistency_ratio
            results["is_consistent"] = self.is_consistent
            results["is_fuzzy"] = self.is_fuzzy
            
        if self.decision_matrices:
            if self.aggregated_decision_matrix is None:
                self.aggregate_decision_matrices()
            results["aggregated_decision_matrix"] = self.aggregated_decision_matrix
            
        return results
