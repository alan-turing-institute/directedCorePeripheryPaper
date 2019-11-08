# This file will contain the graph generation code mostly around the
# networkx graph type
import networkx as nx
import random as rd
from math import floor


class IncorrectParameters(Exception):
        pass

# This is not efficient but leads to less code to test.


def syntheticModel1(n, mP, shuffle=False):
    """ Code to generate the 1-parameter synthetic model.  Note this is a
    slightly different but equivalent parameterisation than in the paper, with
    mP=0.5-p - p the parameter in the paper.  Thus, mP=0 gives a perfect
    directed core-periphery structure.  and mP=0.5 gives an ER graph.  """
    return syntheticModel2(n, 1-mP, mP,shuffle)


# The header is similar between the graphs
def syntheticModel2(n, p1, p2,shuffle=False):
    if p1 <= p2:
            raise IncorrectParameters
    if n % 4 != 0:
            raise IncorrectParameters
    groupSize = int(n/4)
    ns = [groupSize,]*4
    return __syntheticModel2Helper__(ns, p1, p2,shuffle)


def __syntheticModel2Helper__(ns, p1, p2,shuffle=False):
    assert(len(ns)==4)
    n = sum(ns)
    groups = []
    for i in range(4):
        groups += [i,]*ns[i]
    if shuffle:
        rd.shuffle(groups)
    node2group = dict(zip(range(n),groups))
#    groupSize = n/4
    edgeTypes = [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 0]]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.name=groups
    for n1 in range(n):
            # incase this code is ran under python3
            i1 = node2group[n1]
            for n2 in range(n):
                    # incase this code is ran under python3
                    i2 = node2group[n2]
                    if edgeTypes[i1][i2] == 1:
                            if rd.random() < p1:
                                    G.add_edge(n1, n2)
                    else:
                            if rd.random() < p2:
                                    G.add_edge(n1, n2)
    return G

