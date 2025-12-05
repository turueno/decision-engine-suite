import numpy as np
import pandas as pd

class ANPEngine:
    def __init__(self, clusters, nodes, connections):
        """
        Initialize ANP Engine.
        
        Args:
            clusters (list): List of cluster names.
            nodes (dict): Dict mapping cluster names to list of node names.
            connections (list): List of tuples (source_node, target_node) indicating influence.
                                Source influences Target.
        """
        self.clusters = clusters
        self.nodes = nodes
        self.connections = connections
        
        # Flatten nodes to a list for indexing
        self.all_nodes = []
        self.node_to_cluster = {}
        self.node_indices = {}
        
        idx = 0
        for cluster in self.clusters:
            if cluster in self.nodes:
                for node in self.nodes[cluster]:
                    self.all_nodes.append(node)
                    self.node_to_cluster[node] = cluster
                    self.node_indices[node] = idx
                    idx += 1
                    
        self.n = len(self.all_nodes)
        self.unweighted_supermatrix = np.zeros((self.n, self.n))
        self.weighted_supermatrix = None
        self.limit_matrix = None
        
    def set_unweighted_supermatrix(self, matrix):
        """
        Set the unweighted supermatrix directly (or populated via pairwise comparisons).
        Matrix should be (n x n).
        """
        if matrix.shape != (self.n, self.n):
            raise ValueError(f"Matrix shape mismatch. Expected ({self.n}, {self.n}), got {matrix.shape}")
        self.unweighted_supermatrix = matrix
        
    def calculate_weighted_supermatrix(self):
        """
        Normalize the unweighted supermatrix by cluster to make it column stochastic.
        Each column in a cluster block should sum to the cluster's weight (usually 1/num_clusters or defined).
        
        Simplification: We will normalize each column to sum to 1.
        This assumes equal weight for clusters for now, or implicit weighting in the inputs.
        Standard ANP requires a separate "Cluster Matrix" comparison. 
        For this implementation, we'll use column normalization (Stochastic).
        """
        self.weighted_supermatrix = np.zeros((self.n, self.n))
        
        # Normalize each column
        for j in range(self.n):
            col_sum = np.sum(self.unweighted_supermatrix[:, j])
            if col_sum > 0:
                self.weighted_supermatrix[:, j] = self.unweighted_supermatrix[:, j] / col_sum
            else:
                # If column is zero (sink node), keep it zero or handle as needed
                self.weighted_supermatrix[:, j] = 0.0
                
        return self.weighted_supermatrix
        
    def calculate_limit_matrix(self, max_iter=100, tol=1e-6):
        """
        Raise the weighted supermatrix to powers until it converges.
        """
        if self.weighted_supermatrix is None:
            self.calculate_weighted_supermatrix()
            
        matrix = self.weighted_supermatrix.copy()
        
        for i in range(max_iter):
            next_matrix = np.dot(matrix, matrix)
            
            # Check convergence
            if np.allclose(matrix, next_matrix, atol=tol):
                self.limit_matrix = next_matrix
                return self.limit_matrix
            
            matrix = next_matrix
            
            # Safety break for very high powers (2k+1 logic often used)
            if i > max_iter - 2:
                 # If not converged, try averaging (Cesaro sum) if cyclic?
                 # For now, return last state
                 pass
                 
        self.limit_matrix = matrix
        return self.limit_matrix
        
    def get_priorities(self):
        """
        Extract global priorities from the limit matrix.
        """
        if self.limit_matrix is None:
            self.calculate_limit_matrix()
            
        # Usually all columns are the same in the limit matrix (if irreducible)
        # We take the average of columns just in case
        priorities = np.mean(self.limit_matrix, axis=1)
        
        # Normalize just to be sure
        s = np.sum(priorities)
        if s > 0:
            priorities = priorities / s
            
        return dict(zip(self.all_nodes, priorities))
