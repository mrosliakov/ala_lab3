import numpy as np

def svd_from_scratch(A):
    S_left = A @ A.T
    S_right = A.T @ A
    
    eigvals_L, U = np.linalg.eigh(S_left)
    eigvals_R, V = np.linalg.eigh(S_right)
    
    # sort descending 
    idx_L = np.argsort(eigvals_L)[::-1]
    idx_R = np.argsort(eigvals_R)[::-1]
    
    U = U[:, idx_L]
    V = V[:, idx_R]
    eigvals_R = eigvals_R[idx_R]
    
    singular_values = np.sqrt(np.abs(eigvals_R))
    
    Sigma = np.zeros_like(A, dtype=float)
    min_dim = min(A.shape)
    Sigma[:min_dim, :min_dim] = np.diag(singular_values[:min_dim])
    
    Vt = V.T

    # sign ambiguity fix
    for i in range(min_dim):
        Av = A @ V[:, i]
        sigma_u = singular_values[i] * U[:, i]
        
        if np.dot(Av, sigma_u) < 0:
            U[:, i] = -U[:, i]

    return U, Sigma, Vt

if __name__ == "__main__":
    A = np.array([
        [3, 2, 2],
        [2, 3, -2]
    ])

    print("Target:\n", A)

    u, s, vt = svd_from_scratch(A)

    print("\nU:\n", np.round(u, 2))
    print("\nSigma:\n", np.round(s, 2))
    print("\nVt:\n", np.round(vt, 2))

    print("\nReconstructed:\n", np.round(u @ s @ vt, 2))