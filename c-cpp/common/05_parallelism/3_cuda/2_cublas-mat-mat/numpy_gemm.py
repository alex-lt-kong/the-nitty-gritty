import numpy as np

# Define the matrices
A = np.array(
[
[-0.70, -0.95, -0.26],
[0.68, -0.92, 0.19],
[0.18, -0.26, -0.87],
[-0.95, -0.26, -0.94],
]

)

B = np.array(
[
[-0.59, 0.02, -0.06, 0.00],
[0.93, -0.78, 0.96, 0.00],
[-0.83, -0.30, 0.85, 0.00],
]
)

# Multiply the matrices
C = np.dot(A, B)

# Print the result
print(C)