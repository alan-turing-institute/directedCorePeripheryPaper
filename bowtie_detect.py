#Python code for bow-tie structure detection by Jeroen van Lidth de Jeude. The code #implements the definition provided in "Bow-tie decomposition in directed graphs" by #Yang et al 2001 (https://ieeexplore.ieee.org/document/5977625).
#For further details see also van Lidth de Jeude et al. "Reconstructing mesoscale #network structures", Complexity (2019). Doi: 10.1155/2019/5120581.
import networkx as nx
import numpy as np
def get_bowtie_components(G):
    # Input should be an adjacency matrix in numpy nd-array format (directed)
#    G = nx.from_numpy_matrix(np.matrix(adjacency_matrix), create_using=nx.DiGraph())
    GT = nx.reverse(G, copy=True)
    
    strongly_con_comp = list(nx.strongly_connected_components(G))
    strongly_con_comp = max(strongly_con_comp, key=len)
    
    S = strongly_con_comp
    
    v_any = list(S)[0]
    DFS_G = set(nx.dfs_tree(G,v_any).nodes())
    DFS_GT = set(nx.dfs_tree(GT,v_any).nodes())
    OUT = DFS_G - S
    IN = DFS_GT - S
    V_rest = set(G.nodes()) - S - OUT - IN
    
    TUBES = set()
    INTENDRILS = set()
    OUTTENDRILS = set()
    OTHER = set()
    for v in V_rest:
        irv = len(IN & set(nx.dfs_tree(GT,v).nodes())) is not 0
        vro = len(OUT & set(nx.dfs_tree(G,v).nodes())) is not 0
        if irv and vro:
            TUBES.add(v)
        elif irv and not vro:
            INTENDRILS.add(v)
        elif not irv and vro:
            OUTTENDRILS.add(v)
        elif not irv and not vro:
            OTHER.add(v)
    
    return S, IN, OUT, TUBES, INTENDRILS, OUTTENDRILS, OTHER
