import numpy as np

PAULI_X = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0]
    ]
)

PAULI_Y = np.array(
    [
        [0.0, -1.0j],
        [1.0j, 0.0]
    ]
)

PAULI_Z = np.array(
    [
        [1.0, 0.0],
        [0.0, -1.0]
    ]
)

XX = np.kron(PAULI_X, PAULI_X)
YY = np.kron(PAULI_Y, PAULI_Y)
ZZ = np.kron(PAULI_Z, PAULI_Z)
