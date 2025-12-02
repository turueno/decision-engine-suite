import numpy as np
import pandas as pd

class TOPSISEngine:
    def __init__(self, decision_matrix, weights, impacts):
        """
        Initialize the TOPSIS Engine.
        
        Args:
            decision_matrix (pd.DataFrame or np.array): Matrix of alternatives (rows) vs criteria (cols).
            weights (list or np.array): Weights for each criterion.
            impacts (list): List of '+' (benefit) or '-' (cost) for each criterion.
        """
        if isinstance(decision_matrix, pd.DataFrame):
            self.alternatives = decision_matrix.index.tolist()
            self.criteria = decision_matrix.columns.tolist()
            self.matrix = decision_matrix.values.astype(float)
        else:
            self.matrix = np.array(decision_matrix, dtype=float)
            self.alternatives = [f"A{i+1}" for i in range(self.matrix.shape[0])]
            self.criteria = [f"C{i+1}" for i in range(self.matrix.shape[1])]

        self.weights = np.array(weights, dtype=float)
        self.impacts = impacts
        
        if len(self.weights) != self.matrix.shape[1]:
            raise ValueError("Number of weights must match number of criteria.")
        if len(self.impacts) != self.matrix.shape[1]:
            raise ValueError("Number of impacts must match number of criteria.")

    def rank(self):
        """
        Perform TOPSIS ranking.
        """
        # 1. Normalize the decision matrix
        norm_matrix = self.matrix / np.sqrt((self.matrix**2).sum(axis=0))
        
        # 2. Multiply by weights
        weighted_matrix = norm_matrix * self.weights
        
        # 3. Determine Ideal Best and Ideal Worst
        ideal_best = []
        ideal_worst = []
        
        for i, impact in enumerate(self.impacts):
            if impact == '+':
                ideal_best.append(weighted_matrix[:, i].max())
                ideal_worst.append(weighted_matrix[:, i].min())
            else:
                ideal_best.append(weighted_matrix[:, i].min())
                ideal_worst.append(weighted_matrix[:, i].max())
                
        ideal_best = np.array(ideal_best)
        ideal_worst = np.array(ideal_worst)
        
        # 4. Calculate Euclidean distances
        dist_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
        dist_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))
        
        # 5. Calculate Performance Score
        scores = dist_worst / (dist_best + dist_worst)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Alternative': self.alternatives,
            'Score': scores
        })
        
        results = results.sort_values(by='Score', ascending=False).reset_index(drop=True)
        results.index += 1 # Rank starting from 1
        
        return results
