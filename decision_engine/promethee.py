import numpy as np
import pandas as pd

class PrometheeEngine:
    def __init__(self, decision_matrix, weights, impacts, preference_functions, p_params=None, q_params=None):
        """
        Initialize the PROMETHEE Engine.
        
        Args:
            decision_matrix (pd.DataFrame or np.array): Matrix of alternatives (rows) vs criteria (cols).
            weights (list or np.array): Weights for each criterion.
            impacts (list): List of '+' (benefit) or '-' (cost) for each criterion.
            preference_functions (list): List of function types per criterion: 'Usual', 'Linear', 'V-Shape', 'Level', 'Linear w/ Indifference'.
                                       For simplicity, we'll support 'Usual' and 'Linear'.
            p_params (list): Preference thresholds (p) for each criterion. Required for Linear.
            q_params (list): Indifference thresholds (q) for each criterion. Required for Linear/Level.
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
        # Normalize weights
        if self.weights.sum() != 0:
            self.weights = self.weights / self.weights.sum()
            
        self.impacts = impacts
        self.pref_funcs = preference_functions
        self.p = p_params if p_params else [0]*len(weights)
        self.q = q_params if q_params else [0]*len(weights)
        
        self.n_alt = self.matrix.shape[0]
        self.n_crit = self.matrix.shape[1]

    def _preference_func(self, d, func_type, p, q):
        """
        Calculate preference P(a, b) based on difference d = f(a) - f(b).
        d is already adjusted for impact (benefit/cost).
        """
        if d <= 0:
            return 0.0
        
        if func_type == 'Usual':
            return 1.0
        
        elif func_type == 'Linear':
            # Linear preference (Type 5 in some docs, but often just p)
            # Here we assume simple linear up to p: d/p
            if d <= p:
                return d / p
            else:
                return 1.0
                
        elif func_type == 'Linear (q, p)':
            # Type 5: Indifference up to q, then linear to p
            if d <= q:
                return 0.0
            elif d <= p:
                return (d - q) / (p - q)
            else:
                return 1.0
        
        return 0.0

    def calculate_flows(self):
        """
        Calculate Phi+ (Leaving), Phi- (Entering), and Phi (Net).
        """
        # 1. Calculate Preference Indices Pi(a, b)
        # Pi(a, b) = sum(w_j * P_j(a, b))
        
        pi_matrix = np.zeros((self.n_alt, self.n_alt))
        
        for i in range(self.n_alt):
            for k in range(self.n_alt):
                if i == k:
                    continue
                
                weighted_sum_p = 0
                for j in range(self.n_crit):
                    # Calculate difference based on impact
                    diff = 0
                    if self.impacts[j] == '+':
                        diff = self.matrix[i, j] - self.matrix[k, j]
                    else:
                        diff = self.matrix[k, j] - self.matrix[i, j]
                    
                    p_val = self._preference_func(
                        diff, 
                        self.pref_funcs[j], 
                        self.p[j], 
                        self.q[j]
                    )
                    weighted_sum_p += self.weights[j] * p_val
                
                pi_matrix[i, k] = weighted_sum_p

        # 2. Calculate Flows
        # Phi+ (Leaving): Sum of row / (n-1)
        # Phi- (Entering): Sum of col / (n-1)
        
        phi_plus = pi_matrix.sum(axis=1) / (self.n_alt - 1)
        phi_minus = pi_matrix.sum(axis=0) / (self.n_alt - 1)
        phi_net = phi_plus - phi_minus
        
        results = pd.DataFrame({
            'Alternative': self.alternatives,
            'Phi+': phi_plus,
            'Phi-': phi_minus,
            'Net Phi': phi_net
        })
        
        results = results.sort_values(by='Net Phi', ascending=False).reset_index(drop=True)
        results.index += 1
        
        return results
