import numpy as np
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
import helperFunctions as hf
import networkx as nx
from scipy import sparse
#function [l0,l1,l2, timeDIS] = disum_reg_MC(~, k, A)
def  disum_reg_MC(G, k):
    A = nx.to_numpy_array(G,sorted(G.nodes()))
    # N = size(A,1); m = nnz(A);
    n = len(G)
    m = G.number_of_edges()

    # tau is the average node degree
    tau = m / n

    # P = sum(A,1) + tau; O = sum(A,2) + tau;
    P = A.sum(axis=0) + tau
    O = A.sum(axis=1) + tau
    # P = ones(size(P)) ./ sqrt(P);
    # O = ones(size(O)) ./ sqrt(O);
    # A = (A .* P) .* O;
    divisor = np.outer(1.0 / np.sqrt(O), 1.0 / np.sqrt(P))
#    divisor1 = np.outer(1 / A.sum(axis=1), np.ones(n))
#    divisor2 = np.outer( np.ones(n),1 / A.sum(axis=0))
#    import pdb
#    pdb.set_trace()
    #Asilly= np.zeros([n,n])
    #for i in range(n):
    #    for j in range(n):
    #        Asilly[i,j] = A[i,j]/np.sqrt((G.out_degree(i)+tau)*(G.in_degree(j)+tau))
    A = A*divisor
    #import pdb
    #pdb.set_trace()

    # %% Added at some instances would crash due to Nan and Inf entries...
    # A( isnan(A) ) = 0;
    # A( isinf(A) ) = 0;

    A[np.isnan(A)] = 0
    A[np.isinf(A)] = 0

    #[U,~,V] = svds(A,k,'largest','MaxIterations',10000);
    U,a,V = svds(A,k,which='LM',maxiter=10000)
    V=V.T

    U=np.divide(U,np.sqrt((U*U).sum(axis=0,keepdims=1)))
    V=np.divide(V,np.sqrt((V*V).sum(axis=0,keepdims=1)))
#    print(U1[:5,:])
#    print(V1[:5,:])

#    # % 3 versions of DISUM  (DISUM-L, DISUM-R, DISUM-LR)
#    l2 = kmeanspp([U V]',k);
#    l0 = kmeanspp(U',k);
#    l1 = kmeanspp(V',k);
    combination =  np.concatenate([U,V],axis=1)
    l2 = hf.kmeansp2(combination, k,1000*n)
    l0 = hf.kmeansp2(U, k,1000*n)
    l1 = hf.kmeansp2(V, k,1000*n)
    return l0,l1,l2
#end





#function  [ prec, rec, f1, ARI, RI, MR, objs ] = run_SaPa(G, typeSaPa, PARS)

def run_SaPa(G,version, k):
#Gplus = G;  Gplus(Gplus < 0) = 0;   % G = [];
##A = Gplus;
    A = nx.to_numpy_array(G,sorted(G.nodes()))
#    A = nx.to_scipy_sparse_matrix(G,sorted(G.nodes()))
#n = size(A,1);
#k = PARS.k;
    n = len(G)


# if strcmp(typeSaPa, 'sapa_1') == 1
#      U = A * A' + A' * A;
    if version==1:
        A+=np.eye(n)
        U = A.T@A +A.T@A
#elseif strcmp(typeSaPa, 'sapa_2') == 1
#     % D_o is the diagonal matrix of out- degrees
#     deg_o = sum(A,2); % row sums
#     deg_i = sum(A,1); % column sums
#
#     deg_o_inv_half = deg_o .^ (-0.5);
#     deg_i_inv_half = deg_i .^ (-0.5);
#
#     Dih_o = diag(deg_o_inv_half);
#     Dih_i = diag(deg_i_inv_half);
#
#     % Dih_i is the diagonal matrix of in-degrees .^ {-1/2}
#
#     U = Dih_o * A * Dih_i * A' * Dih_o ...
#             + Dih_i * A' * Dih_o * A * Dih_i;
    elif version==2:
         deg_o = np.array(A.sum(axis=1));
         deg_i = np.array(A.sum(axis=0));

         deg_o_inv_half = np.power(deg_o,-0.5);
         deg_i_inv_half = np.power(deg_i,-0.5);

         ## Setting these to 0 as it does not effect the calculation
         deg_o_inv_half[np.isnan(deg_o_inv_half)]=0
         deg_o_inv_half[np.isinf(deg_o_inv_half)]=0

         deg_i_inv_half[np.isnan(deg_i_inv_half)]=0
         deg_i_inv_half[np.isinf(deg_i_inv_half)]=0

         Dih_o = np.diag(np.array(deg_o_inv_half));
         Dih_i = np.diag(np.array(deg_i_inv_half));


         U = Dih_o@A@Dih_i@A.T@Dih_o + Dih_i@A.T@Dih_o@A@Dih_i

#nrEigs = PARS.k;
#[V,D] = eigs(U, nrEigs, 'largestabs');
#label1 = int64(kmeans_pp(V,k))';
    D,V = sparse.linalg.eigsh(U,k , which = 'LM');
    assert(abs(V.imag.max())==0)
    V=V.real

#    label1 = int64(kmeans_pp(V,k))';
    label1 = hf.kmeansp2(V, k)
    return label1
