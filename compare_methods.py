import numpy as np
from decision_engine.ahp import AHPEngine
from decision_engine.fuzzy_ahp import FuzzyAHPEngine

# Inconsistent Matrix
# C1 > C2 (3)
# C2 > C3 (3)
# C1 > C3 (5)  <-- Should be 9 for perfect consistency
matrix = [
    [1, 3, 5],
    [1/3, 1, 3],
    [1/5, 1/3, 1]
]
criteria = ["C1", "C2", "C3"]

print("--- Standard AHP ---")
ahp = AHPEngine(matrix, criteria)
res_ahp = ahp.get_results()
print(res_ahp['weights'])

print("\n--- Fuzzy AHP ---")
fahp = FuzzyAHPEngine(matrix, criteria)
res_fahp = fahp.get_results()
print(res_fahp['weights'])

# Calculate difference
diff = 0
for c in criteria:
    diff += abs(res_ahp['weights'][c] - res_fahp['weights'][c])
print(f"\nTotal Difference: {diff}")
