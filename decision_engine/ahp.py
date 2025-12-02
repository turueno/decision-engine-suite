import numpy as np

class AHPEngine:
    def __init__(self, criteria_matrix, criteria_names=None):
        """
        Initialize the AHP Engine.
        
        Args:
            criteria_matrix (list or np.array): Square matrix of pairwise comparisons.
            criteria_names (list, optional): List of names for the criteria.
        """
        self.matrix = np.array(criteria_matrix, dtype=float)
        self.n = self.matrix.shape[0]
        self.criteria_names = criteria_names if criteria_names else [f"C{i+1}" for i in range(self.n)]
        
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Criteria matrix must be square.")

        self.weights = None
        self.cr = None
        self.consistency_index = None
        self.random_index = {
            1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
        }

    def calculate_weights(self):
        """
        Calculate weights using the Eigenvector method.
        """
        # Normalize the matrix
        column_sums = self.matrix.sum(axis=0)
        normalized_matrix = self.matrix / column_sums
        
        # Calculate weights (average of rows)
        self.weights = normalized_matrix.mean(axis=1)
        
        # Calculate Consistency Ratio
        self._calculate_consistency(column_sums)
        
        return self.weights

    def _calculate_consistency(self, column_sums):
        """
        Calculate Consistency Index (CI) and Consistency Ratio (CR).
        """
        # Lambda Max
        lambda_max = np.dot(column_sums, self.weights)
        
        # Consistency Index
        self.consistency_index = (lambda_max - self.n) / (self.n - 1) if self.n > 1 else 0
        
        # Random Index
        ri = self.random_index.get(self.n, 1.49) # Default to 1.49 for n > 10 for simplicity
        
        # Consistency Ratio
        if ri == 0:
            self.cr = 0
        else:
            self.cr = self.consistency_index / ri

    def get_results(self):
        """
        Return a dictionary with results.
        """
        if self.weights is None:
            self.calculate_weights()
            
        return {
            "weights": dict(zip(self.criteria_names, self.weights)),
            "consistency_ratio": self.cr,
            "is_consistent": self.cr < 0.1
        }
