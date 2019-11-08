import networkx as nx

def HITS_alg(G):
    return nx.hits_scipy(G,100000000000)
