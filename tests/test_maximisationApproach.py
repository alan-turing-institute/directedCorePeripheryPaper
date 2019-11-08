import networkx as nx
import sys
sys.path.append('../')
import numpy as np
import random as rd
from copy import deepcopy
import maximisationApproach as ma
from collections import Counter
from math import log

# Helper functions


def __considerMovementOld__(G, currentSolution, node):
    # This function will give 4 scores indicating the relative score of each of
    # the groups. Note, that this is not necessarily the actual likelihood, but
    # we guarentee that the differences are preserved.
    oldGroup = currentSolution[node]
    result = []
    for i in range(4):
        # print(i)
        currentSolution[node] = i
        likelihoodTest = ma.__likelihood__(G, currentSolution)
        result.append(likelihoodTest)
    currentSolution[node] = oldGroup
    return result


def getEinMiss(G, coms, node):
    correctStructure = []
    correctStructure.append([0, 1, 0, 0])
    correctStructure.append([0, 1, 0, 0])
    correctStructure.append([0, 1, 1, 1])
    correctStructure.append([0, 0, 0, 0])
    cs = correctStructure
    ein = 0
    for x in G:
        if x == node:
            continue
        if coms[x] != 3:
            for y in G[x]:
                if y == node:
                    continue
                if cs[coms[x]][coms[y]]:
                    ein += 1
    return ein


# Actual tests

def test_considerMovement():
    ### test if consider movement
    ## is the same as the slow version (i.e. calling the likelihood)
    for seeds in [[6, 38],  [26, 32], [8, 56]]:
        np.random.seed(seeds[0])
        rd.seed(seeds[1])
        for n in [100, 400, 800]:
            G = nx.erdos_renyi_graph(n, 0.1, directed=True)
            currentSolution1 = [rd.randint(0, 3) for i in range(n)]
            currentSolution = deepcopy(currentSolution1)
            node = rd.randint(0, n-1)
            result1 = __considerMovementOld__(G, currentSolution, node)
            currentSolution = deepcopy(currentSolution1)
            savedVars = {}
            savedVars['numEdges'] = G.number_of_edges()
            savedVars['ein'] = ma.getEin(G, currentSolution)
            savedVars['forwardBackward'] = {}

            correctStructure = []
            correctStructure.append([0, 1, 0, 0])
            correctStructure.append([0, 1, 0, 0])
            correctStructure.append([0, 1, 1, 1])
            correctStructure.append([0, 0, 0, 0])
            cs = correctStructure
            savedVars['cs'] = cs
            numCats = Counter(currentSolution1)
            savedVars['numCats'] = numCats
            # Make the initial forward and backward arrays
            fwdBack = savedVars['forwardBackward']
            for x in range(len(G)):
                currSol = deepcopy(currentSolution1)
                fwdBack[x] = ma.getForwardBackward(G, x, currSol)
            currSol = deepcopy(currentSolution1)
            result2 = ma.__considerMovement__(G, currSol, node, savedVars)
            for i in range(4):
                assert(abs(result1[i]-result2[i]) < 10**(-8))


def test_getEin_zero():
    G = nx.DiGraph()
    coms = [0, 1, 2, 3]
    G.add_edge(0, 0)
    assert(ma.getEin(G, coms) == 0)
    G.add_edge(0, 2)
    assert(ma.getEin(G, coms) == 0)
    G.add_edge(0, 3)
    assert(ma.getEin(G, coms) == 0)
    G.add_edge(1, 0)
    assert(ma.getEin(G, coms) == 0)
    G.add_edge(1, 2)
    assert(ma.getEin(G, coms) == 0)
    G.add_edge(1, 3)
    assert(ma.getEin(G, coms) == 0)
    G.add_edge(2, 0)
    assert(ma.getEin(G, coms) == 0)
    G.add_edge(3, 0)
    assert(ma.getEin(G, coms) == 0)
    G.add_edge(3, 1)
    assert(ma.getEin(G, coms) == 0)
    G.add_edge(3, 2)
    assert(ma.getEin(G, coms) == 0)
    G.add_edge(3, 3)
    assert(ma.getEin(G, coms) == 0)


def test_getEin_one():
    G = nx.DiGraph()
    coms = [0, 1, 2, 3]
    G.add_edge(0, 1)
    assert(ma.getEin(G, coms) == 1)
    G.add_edge(1, 1)
    assert(ma.getEin(G, coms) == 2)
    G.add_edge(2, 1)
    assert(ma.getEin(G, coms) == 3)
    G.add_edge(2, 2)
    assert(ma.getEin(G, coms) == 4)
    G.add_edge(2, 3)
    assert(ma.getEin(G, coms) == 5)


def test_getEin_both():
    G = nx.DiGraph()
    coms = [0, 1, 2, 3]
    G.add_edge(0, 0)
    assert(ma.getEin(G, coms) == 0)
    G.add_edge(0, 1)
    assert(ma.getEin(G, coms) == 1)
    G.add_edge(0, 2)
    assert(ma.getEin(G, coms) == 1)
    G.add_edge(0, 3)
    assert(ma.getEin(G, coms) == 1)
    G.add_edge(1, 0)
    assert(ma.getEin(G, coms) == 1)
    G.add_edge(1, 1)
    assert(ma.getEin(G, coms) == 2)
    G.add_edge(1, 2)
    assert(ma.getEin(G, coms) == 2)
    G.add_edge(1, 3)
    assert(ma.getEin(G, coms) == 2)
    G.add_edge(2, 0)
    assert(ma.getEin(G, coms) == 2)
    G.add_edge(2, 1)
    assert(ma.getEin(G, coms) == 3)
    G.add_edge(2, 2)
    assert(ma.getEin(G, coms) == 4)
    G.add_edge(2, 3)
    assert(ma.getEin(G, coms) == 5)
    G.add_edge(3, 0)
    assert(ma.getEin(G, coms) == 5)
    G.add_edge(3, 1)
    assert(ma.getEin(G, coms) == 5)
    G.add_edge(3, 2)
    assert(ma.getEin(G, coms) == 5)
    G.add_edge(3, 3)
    assert(ma.getEin(G, coms) == 5)


def test_forwardBackward():
    randSeeds = []
    randSeeds.append([12, 96])
    randSeeds.append([71, 58])
    randSeeds.append([15, 92])
    randSeeds.append([93, 33])
    randSeeds.append([77, 95])
    randSeeds.append([15, 93])
    randSeeds.append([94, 20])
    randSeeds.append([86, 38])
    randSeeds.append([85, 44])
    randSeeds.append([9,  10])
    randSeeds.append([28, 18])
    randSeeds.append([41, 75])
    randSeeds.append([92,  1])
    randSeeds.append([94, 66])
    randSeeds.append([98, 45])
    randSeeds.append([47, 95])
    randSeeds.append([26, 18])
    randSeeds.append([52, 30])
    randSeeds.append([23, 13])
    randSeeds.append([77, 44])
    for seeds in randSeeds:
        np.random.seed(seeds[0])
        rd.seed(seeds[1])
        G = nx.erdos_renyi_graph(4, 0.5, directed=True)
        for i in range(4):
            if rd.random() < 0.5:
                G.add_edge(i, i)
        A = nx.to_numpy_array(G)
        coms = [0, 1, 2, 3]
        for i in range(4):
            f, b = ma.getForwardBackward(G, i, coms)
            for j in range(4):
                if i == j:
                    assert(abs(0 == f[j]))
                else:
                    assert(abs(A[i, j] == f[j]))
            for j in range(4):
                if i == j:
                    assert(abs(0 == b[j]))
                else:
                    assert(abs(A[j, i] == b[j]))


def test_likelihood():
    randomSeeds = []
    randomSeeds.append([44,  17])
    randomSeeds.append([40,  81])
    randomSeeds.append([96,  34])
    randomSeeds.append([25,  17])
    randomSeeds.append([63,  16])
    randomSeeds.append([56,  100])
    randomSeeds.append([56,  36])
    randomSeeds.append([21,  35])
    randomSeeds.append([15,  99])
    randomSeeds.append([39,  21])
    randomSeeds.append([90,  20])
    randomSeeds.append([22,  72])
    randomSeeds.append([63,  95])
    randomSeeds.append([81,  98])
    randomSeeds.append([97,  14])
    randomSeeds.append([90,  71])
    randomSeeds.append([43,  92])
    randomSeeds.append([72,  82])
    randomSeeds.append([46,  42])
    randomSeeds.append([21,  52])
    randomSeeds.append([63,  44])
    randomSeeds.append([81,  74])
    randomSeeds.append([2, 89])
    randomSeeds.append([61,  46])
    randomSeeds.append([50,  88])
    randomSeeds.append([6, 60])
    randomSeeds.append([23,  62])
    randomSeeds.append([75,  92])
    randomSeeds.append([76,  57])
    randomSeeds.append([70,  25])
    randomSeeds.append([85,  12])
    randomSeeds.append([29,  61])
    randomSeeds.append([83,  87])
    randomSeeds.append([16,  11])
    randomSeeds.append([36,  82])
    randomSeeds.append([23,  44])
    randomSeeds.append([26,  42])
    randomSeeds.append([60,  34])
    randomSeeds.append([39,  1])
    randomSeeds.append([32,  6])
    randomSeeds.append([79,  44])
    randomSeeds.append([52,  46])
    randomSeeds.append([4, 83])
    randomSeeds.append([86,  62])
    randomSeeds.append([10,  52])
    randomSeeds.append([45,  98])
    randomSeeds.append([27,  2])
    randomSeeds.append([81,  86])
    randomSeeds.append([17,  76])
    randomSeeds.append([67,  15])
    cs = [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 0]]
    for seeds in randomSeeds:
        np.random.seed(seeds[0])
        rd.seed(seeds[1])
        for n in [200, 300, 400]:
            coms = [rd.randint(0, 3) for i in range(n)]
            G = nx.erdos_renyi_graph(n, 0.1, directed=True)
            ein = 0
            eout = 0
            Tin = 0
            Tout = 0
            for i in range(n):
                for j in range(n):
                    if cs[coms[i]][coms[j]]:
                        Tin += 1
                    else:
                        Tout += 1
                    if G.has_edge(i, j):
                        if cs[coms[i]][coms[j]]:
                            ein += 1
                        else:
                            eout += 1
            p = ein/Tin
            q = eout/Tout
            l1 = 0
            l1 += log(p)*ein
            l1 += log(1-p)*(Tin-ein)
            l1 += log(q)*eout
            l1 += log(1-q)*(Tout-eout)
            assert(abs(l1-ma.__likelihood__(G, coms)) < 10**(-8))


def test_likelihood_certain():
    cs = [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 0]]
    randomSeeds = []
    randomSeeds.append([3, 24])
    randomSeeds.append([17, 27])
    randomSeeds.append([65, 79])
    randomSeeds.append([39, 53])
    randomSeeds.append([92, 44])
    randomSeeds.append([20, 13])
    randomSeeds.append([61, 45])
    randomSeeds.append([38, 29])
    randomSeeds.append([21, 46])
    randomSeeds.append([9, 5])
    randomSeeds.append([24, 42])
    randomSeeds.append([28, 66])
    randomSeeds.append([48, 37])
    randomSeeds.append([17, 74])
    randomSeeds.append([74, 95])
    randomSeeds.append([69, 1])
    randomSeeds.append([21, 41])
    randomSeeds.append([36, 80])
    randomSeeds.append([48, 52])
    randomSeeds.append([30, 99])
    randomSeeds.append([30, 11])
    randomSeeds.append([44, 62])
    randomSeeds.append([81, 60])
    randomSeeds.append([58, 17])
    randomSeeds.append([42, 86])
    randomSeeds.append([9, 27])
    randomSeeds.append([25, 73])
    randomSeeds.append([69, 88])
    randomSeeds.append([86, 78])
    randomSeeds.append([6, 18])
    for seeds in randomSeeds:
        np.random.seed(seeds[0])
        rd.seed(seeds[1])
        for n in [100, 300, 600]:
            G = nx.DiGraph()
            G.add_nodes_from(range(n))
            coms = [rd.randint(0, 3) for i in range(n)]
            for i in range(n):
                for j in range(n):
                    if cs[coms[i]][coms[j]]:
                        G.add_edge(i, j)
            l1 = ma.__likelihood__(G, coms)
            assert(l1 == 0)


def test_likelihood_certainNegative():
    # This tests if the reversal works
    cs = [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 0]]
    randomSeeds = []
    randomSeeds.append([23,  13])
    randomSeeds.append([35,  52])
    randomSeeds.append([79,  87])
    randomSeeds.append([50,  29])
    randomSeeds.append([63,  73])
    randomSeeds.append([100,  10])
    randomSeeds.append([55,  22])
    randomSeeds.append([25,  10])
    randomSeeds.append([60,  31])
    randomSeeds.append([14,  5])
    randomSeeds.append([69,  59])
    randomSeeds.append([78,  34])
    randomSeeds.append([90,  86])
    randomSeeds.append([50,  90])
    randomSeeds.append([65,  21])
    randomSeeds.append([51,  12])
    randomSeeds.append([36,  5])
    randomSeeds.append([90,  2])
    randomSeeds.append([35,  86])
    randomSeeds.append([86,  65])
    randomSeeds.append([85,  47])
    randomSeeds.append([34,  95])
    randomSeeds.append([95,  57])
    randomSeeds.append([85,  29])
    randomSeeds.append([70,  56])
    for seeds in randomSeeds:
        np.random.seed(seeds[0])
        rd.seed(seeds[1])
        for n in [200, 400, 500]:
            G = nx.DiGraph()
            G.add_nodes_from(range(n))
            coms = [rd.randint(0, 3) for i in range(n)]
            for i in range(n):
                for j in range(n):
                    if not cs[coms[i]][coms[j]]:
                        G.add_edge(i, j)
            l1 = ma.__likelihood__(G, coms)
            assert(l1 == 0)


def test_estimatePandQ():
    randomSeeds = []
    randomSeeds.append([18535,   9487])
    randomSeeds.append([2723,   4216])
    randomSeeds.append([20752,  27521])
    randomSeeds.append([4852,  16667])
    randomSeeds.append([1079,  17983])
    randomSeeds.append([4599,   1056])
    randomSeeds.append([15489,  22803])
    randomSeeds.append([13648,  21766])
    randomSeeds.append([21365,  14154])
    randomSeeds.append([14086,   4557])
    randomSeeds.append([23053,   9670])
    randomSeeds.append([5464,  11471])
    randomSeeds.append([29493,  29395])
    randomSeeds.append([9880,  17116])
    randomSeeds.append([31972,  25794])
    randomSeeds.append([7244,  10407])
    randomSeeds.append([19289,  27164])
    randomSeeds.append([215,  16249])
    randomSeeds.append([23262,   4641])
    randomSeeds.append([557,  28016])
    randomSeeds.append([28287,  23070])
    randomSeeds.append([3942,   9568])
    randomSeeds.append([30848,   4749])
    randomSeeds.append([15473,   3352])
    randomSeeds.append([30419,  10184])
    for seeds in randomSeeds:
        np.random.seed(seeds[0])
        rd.seed(seeds[1])
        for n in [40, 100, 160]:
            G = nx.DiGraph()
            G.add_nodes_from(range(n))
            coms = [rd.randint(0, 3) for i in range(n)]
            savedVars = ma.__constructSavedVars__(G,  coms)
            cs = savedVars['cs']
            for i in range(n):
                for j in range(n):
                    if cs[coms[i]][coms[j]]:
                        G.add_edge(i,  j)
            savedVars = ma.__constructSavedVars__(G,  coms)
            p,  q = ma.__estimatePandQ__(G,  coms,  savedVars)
            assert(p == 1)
            assert(q == 0)
            coms = [rd.randint(0, 3) for i in range(n)]
            cs = savedVars['cs']
            G = nx.DiGraph()
            G.add_nodes_from(range(n))
            for i in range(n):
                for j in range(n):
                    if not cs[coms[i]][coms[j]]:
                        G.add_edge(i,  j)
            savedVars = ma.__constructSavedVars__(G,  coms)
            p,  q = ma.__estimatePandQ__(G,  coms,  savedVars)
            assert(p == 0)
            assert(q == 1)


def test_estimatePnQ2():
    randomSeeds = []
    randomSeeds.append([1175,  4293])
    randomSeeds.append([4056,  30671])
    randomSeeds.append([31568,  2138])
    randomSeeds.append([20524,  29417])
    randomSeeds.append([21768,  3601])
    randomSeeds.append([30513,  21145])
    randomSeeds.append([14125,  26498])
    randomSeeds.append([21365,  12721])
    randomSeeds.append([27283,  17934])
    randomSeeds.append([27095,  987])
    randomSeeds.append([7211,  1523])
    randomSeeds.append([16221,  3721])
    randomSeeds.append([29817,  9127])
    randomSeeds.append([3430,  12803])
    randomSeeds.append([16546,  22186])
    randomSeeds.append([22013,  18273])
    randomSeeds.append([10982,  24436])
    randomSeeds.append([10678,  11876])
    randomSeeds.append([7359,  8405])
    randomSeeds.append([16313,  21774])
    randomSeeds.append([12158,  28646])
    randomSeeds.append([18557,  1804])
    randomSeeds.append([24983,  1643])
    randomSeeds.append([12231,  24934])
    randomSeeds.append([234,  18648])
    for seeds in randomSeeds:
        np.random.seed(seeds[0])
        rd.seed(seeds[1])
        for n in [40, 100, 160]:
            for mod1 in [2, 3, 5, 7]:
                for mod2 in [2, 3, 5, 7]:
                    c1 = 0
                    c2 = 0
                    t1 = 0
                    t2 = 0
                    G = nx.DiGraph()
                    G.add_nodes_from(range(n))
                    coms = [rd.randint(0, 3) for i in range(n)]
                    savedVars = ma.__constructSavedVars__(G,  coms)
                    cs = savedVars['cs']
                    for i in range(n):
                        for j in range(n):
                            if cs[coms[i]][coms[j]]:
                                if c1 % mod1 == 1:
                                    G.add_edge(i,  j)
                                    t1 += 1
                                c1 += 1
                            else:
                                if c2 % mod2 == 1:
                                    G.add_edge(i,  j)
                                    t2 += 1
                                c2 += 1
                    savedVars = ma.__constructSavedVars__(G,  coms)
                    p,  q = ma.__estimatePandQ__(G,  coms,  savedVars)
                    assert(abs(p-t1/c1) < 10**(-10))
                    assert(abs(q-t2/c2) < 10**(-10))


def test_updateVarsOnComChange__():
    # this test moves communities and checks if there is any change
    randomSeeds = []
    randomSeeds.append([9385,  446])
    randomSeeds.append([17610,  2126])
    randomSeeds.append([27835,  29491])
    randomSeeds.append([31745,  1385])
    randomSeeds.append([26933,  23067])
    randomSeeds.append([31886,  7319])
    randomSeeds.append([16651,  3271])
    randomSeeds.append([6532,  5411])
    randomSeeds.append([5857,  4711])
    randomSeeds.append([6511,  27949])
    randomSeeds.append([10352,  12361])
    randomSeeds.append([2092,  28826])
    randomSeeds.append([12682,  12871])
    randomSeeds.append([14969,  11786])
    randomSeeds.append([19285,  13911])
    randomSeeds.append([11268,  6828])
    randomSeeds.append([29414,  6330])
    randomSeeds.append([30925,  1888])
    randomSeeds.append([29530,  14127])
    randomSeeds.append([18818,  6635])
    randomSeeds.append([12377,  16494])
    randomSeeds.append([15162,  25450])
    randomSeeds.append([22159,  9859])
    randomSeeds.append([22973,  23493])
    randomSeeds.append([24439,  20761])
    for seeds in randomSeeds:
        np.random.seed(seeds[0])
        rd.seed(seeds[1])
        for n in [40, 100, 200]:
            for changes in [1, 5, 10, 20]:
                G = nx.erdos_renyi_graph(n, 0.1, directed=True)
                coms = [rd.randint(0, 3) for i in range(n)]
                savedVars = ma.__constructSavedVars__(G,  coms)
                coms1 = coms[:]
                for x in range(changes):
                    nde = rd.randint(0, n-1)
                    nCom = rd.randint(0, 3)
                    ma.__updateVarsOnComChange__(savedVars, G, coms, nde, nCom)
                    coms1[nde] = nCom
                savedVars1 = ma.__constructSavedVars__(G, coms1)
                ## check if ein is equal
                assert(savedVars1['ein'] == savedVars['ein'])
                ## check if numcats is equal
                for x in range(4):
                    assert(savedVars1['numCats'][x] == savedVars['numCats'][x])
                ## check if forwardBackward is equal
                sv1 = savedVars1['forwardBackward']
                sv2 = savedVars['forwardBackward']
                for x in range(n):
                    for y in range(4):
                        assert(sv1[x][0][y] == sv2[x][0][y])
                        assert(sv1[x][1][y] == sv2[x][1][y])


def test_likelihoodMaximisationHelper():
    # lets see if it moves an obviously out of place node
    randomSeeds = []
    randomSeeds.append([7689, 20976])
    randomSeeds.append([15496, 4486])
    randomSeeds.append([13618, 9202])
    randomSeeds.append([29661, 10301])
    randomSeeds.append([26463, 11866])
    randomSeeds.append([28511, 28487])
    randomSeeds.append([14730, 4889])
    randomSeeds.append([25214, 6170])
    randomSeeds.append([12996, 20817])
    randomSeeds.append([28007, 1731])
    randomSeeds.append([10107, 16513])
    randomSeeds.append([5278, 9295])
    randomSeeds.append([4990, 1071])
    randomSeeds.append([15233, 7555])
    randomSeeds.append([21541, 17502])
    randomSeeds.append([28841, 11279])
    randomSeeds.append([4321, 9281])
    randomSeeds.append([29605, 19962])
    randomSeeds.append([20434, 20193])
    randomSeeds.append([23008, 31084])
    randomSeeds.append([8756, 15553])
    randomSeeds.append([10718, 14326])
    randomSeeds.append([23555, 3397])
    randomSeeds.append([4450, 26465])
    randomSeeds.append([6891, 20693])
    likeMaxHelper = ma.__likelihoodMaximisationHelper__
    for seeds in randomSeeds:
        np.random.seed(seeds[0])
        rd.seed(seeds[1])
        for n in [40, 100, 120]:
            G = nx.DiGraph()
            G.add_nodes_from(range(n))
            coms = [rd.randint(0, 3) for i in range(n)]
            savedVars = ma.__constructSavedVars__(G,  coms)
            cs = savedVars['cs']
            for i in range(n):
                for j in range(n):
                    if cs[coms[i]][coms[j]]:
                        G.add_edge(i,  j)
            savedVars = ma.__constructSavedVars__(G,  coms)
            for i in range(10):
                n1 = rd.randint(0, n-1)
                coms1 = coms[:]
                coms1[n1] = (coms[n1]+1) % 4
                savedVars = ma.__constructSavedVars__(G, coms)
                l1 = ma.__likelihood__(G, coms1)
                maxSol,  curMax, chge = likeMaxHelper(G, coms1, l1, [n1, ])
                assert(chge)
                assert(abs(curMax) < 10**(-10))
                for i123 in range(n):
                    assert(coms[i123] == maxSol[i123])
            for i in range(3):
                coms1 = coms[:]
                for j in range(i+1):
                    n1 = rd.randint(0, n-1)
                    coms1[n1] = (coms[n1]+1) % 4
                savedVars = ma.__constructSavedVars__(G, coms)
                l1 = ma.__likelihood__(G, coms1)
                order = list(set(range(n)))
                rd.shuffle(order)
                maxSol,  curMax, chge = likeMaxHelper(G, coms1, l1, order)
                assert(chge)
                assert(abs(curMax) < 10**(-10))
                for i123 in range(n):
                    assert(coms[i123] == maxSol[i123])
