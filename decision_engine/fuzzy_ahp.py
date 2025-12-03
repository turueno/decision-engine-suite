import numpy as np

class FuzzyAHPEngine:
    def __init__(self, criteria_matrix, criteria_names=None, input_type="crisp"):
        """
        Initialize the Fuzzy AHP Engine.
        
        Args:
            criteria_matrix (list or np.array): Matrix of comparisons.
                                              If input_type="crisp", expects 1-9 scale.
                                              If input_type="fuzzy", expects (n, n, 3) array of TFNs.
            criteria_names (list, optional): List of names for the criteria.
            input_type (str): "crisp" or "fuzzy".
        """
        self.input_type = input_type
        if input_type == "fuzzy":
            self.fuzzy_matrix = np.array(criteria_matrix, dtype=float)
            self.n = self.fuzzy_matrix.shape[0]
        else:
            self.crisp_matrix = np.array(criteria_matrix, dtype=float)
            self.n = self.crisp_matrix.shape[0]
            
        self.criteria_names = criteria_names if criteria_names else [f"C{i+1}" for i in range(self.n)]
        
        # TFN Scale: (l, m, u)
        # 1: (1, 1, 1) if equal, else (1, 1, 3) ? 
        # Standard scale often used:
        # 1: (1, 1, 1)
        # 3: (2, 3, 4)
        # 5: (4, 5, 6)
        # 7: (6, 7, 8)
        # 9: (9, 9, 9) or (8, 9, 9)
        # Intermediate: 2: (1, 2, 3), 4: (3, 4, 5), etc.
        self.tfn_scale = {
            1: (1, 1, 1),
            2: (1, 2, 3),
            3: (2, 3, 4),
            4: (3, 4, 5),
            5: (4, 5, 6),
            6: (5, 6, 7),
            7: (6, 7, 8),
            8: (7, 8, 9),
            9: (9, 9, 9)
        }
        
        if input_type == "crisp":
            self.fuzzy_matrix = self._fuzzify_matrix()
            
        self.weights = None

    def _get_tfn(self, value):
        """Convert a crisp value to TFN."""
        if value >= 1:
            # Direct comparison
            val = int(round(value))
            return self.tfn_scale.get(val, (val-1, val, val+1))
        else:
            # Reciprocal
            val = int(round(1/value))
            l, m, u = self.tfn_scale.get(val, (val-1, val, val+1))
            return (1/u, 1/m, 1/l)

    def _fuzzify_matrix(self):
        """Convert crisp matrix to fuzzy matrix."""
        fuzzy_matrix = np.zeros((self.n, self.n, 3))
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    fuzzy_matrix[i, j] = (1, 1, 1)
                else:
                    fuzzy_matrix[i, j] = self._get_tfn(self.crisp_matrix[i, j])
        return fuzzy_matrix

    def calculate_weights(self):
        """
        Calculate weights using Buckley's Geometric Mean Method.
        """
        # 1. Geometric Mean of each row (fuzzy geometric mean)
        # r_i = (prod(a_ij))^(1/n)
        # For TFN (l, m, u), prod is (prod(l), prod(m), prod(u))
        # power is (l^(1/n), m^(1/n), u^(1/n))
        
        geo_means = []
        for i in range(self.n):
            row = self.fuzzy_matrix[i]
            l_prod = np.prod(row[:, 0])
            m_prod = np.prod(row[:, 1])
            u_prod = np.prod(row[:, 2])
            
            geo_means.append((
                l_prod**(1/self.n),
                m_prod**(1/self.n),
                u_prod**(1/self.n)
            ))
        
        geo_means = np.array(geo_means)
        
        # 2. Calculate Sum of Geometric Means
        # Sum of TFNs: (sum(l), sum(m), sum(u))
        total_l = np.sum(geo_means[:, 0])
        total_m = np.sum(geo_means[:, 1])
        total_u = np.sum(geo_means[:, 2])
        
        total_tfn = (total_l, total_m, total_u)
        
        # 3. Calculate Fuzzy Weights
        # w_i = r_i * (total)^(-1)
        # Inverse of TFN (L, M, U) is (1/U, 1/M, 1/L)
        inv_total = (1/total_tfn[2], 1/total_tfn[1], 1/total_tfn[0])
        
        fuzzy_weights = []
        for i in range(self.n):
            r = geo_means[i]
            # Multiply TFNs: (l1*l2, m1*m2, u1*u2)
            w = (
                r[0] * inv_total[0],
                r[1] * inv_total[1],
                r[2] * inv_total[2]
            )
            fuzzy_weights.append(w)
            
        # 4. Defuzzify (Center of Area / Mean)
        # M_crisp = (l + m + u) / 3
        self.weights = np.array([(w[0] + w[1] + w[2]) / 3 for w in fuzzy_weights])
        
        # Normalize to ensure sum is 1 (usually it's close but good to normalize)
        self.weights = self.weights / self.weights.sum()
        
        return self.weights

    def get_results(self):
        if self.weights is None:
            self.calculate_weights()
            
        return {
            "weights": dict(zip(self.criteria_names, self.weights)),
            "method": "Fuzzy AHP (Buckley's Geometric Mean)"
        }
