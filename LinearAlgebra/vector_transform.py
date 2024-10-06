import numpy as np


def transform(vector):
    T = np.array([[1, 0], [2, 0], [0, 3]])

    print("input vector: \n", vector, "\n")
    print("transform matrix: \n", T, "\n")

    transformed = T @ vector

    print("transformed vector: \n", transformed, "\n")
    return transformed


vector = np.array([[23], [25]])
transformed = transform(vector)

print("for vector: \n", vector, "\n")
print("transformed is: \n", transformed, "\n")
