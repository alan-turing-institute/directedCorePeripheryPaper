import numpy as np
import networkx as nx
import scipy



class EqualSingularValues(Exception):
        pass

# tested
def __lowRankHelper__(A: np.ndarray, dimension: int, returnMatrix: bool = True,
                      strict: bool = False) -> int:
        if A.sum()/(A.shape[0]**2) < 0.1:
                # We take extra dimensions here just in case we
                # have tied values
                u, s, vh = scipy.sparse.linalg.svds(
                        A, dimension+5, which='LM')
                temp1 = sorted(zip(-s, range(len(s))))
                perm = list(zip(*temp1))[1]
                u[:, :] = u[:, perm]
                s = s[perm, ]
                vh[:, :] = vh[perm, :]
        else:
                if min(dimension+5,A.shape[0])==A.shape[0]:
                    u, s, vh = np.linalg.svd(A, full_matrices=True)
                else:
                    u, s, vh = scipy.sparse.linalg.svds(
                            A, min(dimension+5,A.shape[0]), which='LM')
                temp1 = sorted(zip(-s, range(len(s))))
                perm = list(zip(*temp1))[1]
                u[:, :] = u[:, perm]
                s = s[perm, ]
                vh[:, :] = vh[perm, :]
        uLowDim = u[:, :dimension]
        sLowDim = np.diag(s[:dimension])
        if strict:
                for i in range(min(dimension, A.shape[0]-1)):
                    if abs(s[i]-s[i+1]) < 10**(-10):
                            raise EqualSingularValues
        vhLowDim = vh[:dimension, :]
        if returnMatrix:
                return (uLowDim@sLowDim)@vhLowDim
        else:
                return [uLowDim, sLowDim, vhLowDim]


## Not Tested
def lowRankApproximation(
        G: nx.DiGraph, dimension: int = 2, returnMatrix: bool = True,
        strict: bool = False) ->np.ndarray:
        A = nx.to_numpy_array(G,sorted(G))
        return __lowRankHelper__(A, dimension, returnMatrix, strict)


## Not Tested
def lowRankApproximationThreshold(
        G: nx.DiGraph, dimension: int = 2, threshold: float = 0.5,
        strict: bool = False) ->np.array:
        L = lowRankApproximation(G, dimension, True, strict)
        return (L > threshold).astype(np.float64)
