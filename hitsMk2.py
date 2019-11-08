import networkx as nx
import numpy as np
from scipy import sparse
import random as rd


# tested
def constructCOOtest(G):
    x1 = []
    y1 = []
    for x in G:
        x1 += [x, ]*len(G[x])
        for y in G[x]:
            y1.append(y)
    ones = [1, ]*G.number_of_edges()
    ones = np.array(ones)
    x1 = np.array(x1)
    y1 = np.array(y1)
    A = sparse.coo_matrix((ones, (x1, y1)), shape=[len(G), len(G)])
    return A


def __setup__(G):
    if nx.density(G) < 0.2:
            A = constructCOOtest(G)
            At = (A.T).tocsr()
            A = sparse.csr_matrix(A)
    else:
            A = nx.to_numpy_array(G, sorted(G.nodes()))
            At = A.T
    C = [[-1,  1, -1, -1],
         [-1,  1, -1, -1],
         [-1,  1,  1,  1],
         [-1, -1, -1, -1]]
    C = np.array(C)
    Ct = C.T
    q1 = G.number_of_edges()
    n = A.shape[0]
    s1 = q1/float(n*n)
    s2 = 1-q1/float(n*n)
    return A, At, C, Ct, n, s1, s2

##########  Take Norm ##########


# tested
def __getNorm__(coms):
    assert(coms.shape[1] == 4)
    coms = coms.astype(np.float).copy()
    min1 = coms.min(axis=1)
    # This step could be changed to improve speed
    coms = coms - np.tensordot(min1, np.ones(4), axes=0)
    sum1 = np.sum(coms, axis=1)
    coms[abs(sum1) < 10**(-10), :] = 0.25
    sum1[abs(sum1) < 10**(-10)] = 1.0
    coms /= np.tensordot(sum1, np.ones(4), axes=0)
    return coms


##########  Run Updates to each score separately ##########

# Unnormalised Version

def __hitsmk2Order__(A, At, C, Ct, oldVer, s1, C2, order):
    temp1 = A@oldVer@Ct + At@oldVer@C
    temp2 = s1*C2@(oldVer.sum(axis=0))
    temp1[:, order] -= temp2[order]
    return temp1


def advancedHitsOrder(G, maxiter=1000):
    A, At, C, Ct, n, s1, s2 = __setup__(G)
    C2 = C+Ct
    rawValues = np.random.random([len(G), 4])
    oldVer = __getNorm__(rawValues)
    diff = 100
    count = 0
    while diff > 10**(-8):
        count += 1
        if count > maxiter:
            print('One time fallback')
            return advancedHitsVectOneAtATime(G, oldVer)
        diff = 0
        for order in range(4):
            rawUpdate = __hitsmk2Order__(A, At, C, Ct, oldVer, s1, C2, order)
            rawValues[:, order] = rawUpdate[:, order]
            newVer = __getNorm__(rawValues)
            diff = max(diff, abs(oldVer[:, order] - newVer[:, order]).max())
            oldVer = newVer
    return oldVer


# Normalised Version
def __hitsmk2GroupOrder__(A, At, C, Ct, oldVer, s1, C2, order):
    divisors = oldVer.sum(axis=0)
    assert((divisors > -10**-(14)).all())
    divisors[divisors < 10**(-14)] = 1
    oldVer = oldVer/divisors
    temp1 = A@oldVer@Ct + At@oldVer@C
    temp2 = s1*(C2)@(oldVer.sum(axis=0))
    temp3 = np.ones(A.shape[0])
    temp4 = np.tensordot(temp3, temp2, axes=0)
    NewQ = temp1-temp4
    return NewQ


# no test needed (simple wrapper)
def advancedHitsOrderGroup(G, maxiter=1000):
    A, At, C, Ct, n, s1, s2 = __setup__(G)
    C2 = C+Ct
    rawValues = np.random.random([len(G), 4])
    oldVer = __getNorm__(rawValues)
    diff = 100
    count = 0
    while diff > 10**(-8):
        count += 1
        if count > maxiter:
            print('One time setup')
            return advancedHitsVectOneAtATime_Group(G, oldVer)
        diff = 0
        for order in range(4):
            rawUpdate = __hitsmk2GroupOrder__(A, At, C, Ct, oldVer, s1, C2, order)
            rawValues[:, order] = rawUpdate[:, order]
            newVer = __getNorm__(rawValues)
            diff = max(diff, abs(oldVer[:, order] - newVer[:, order]).max())
            oldVer = newVer
    return newVer


##########  Non Normalised Version  ##########

#  Original


# tested (compared with the slow version)
def __hitsmk2__(A, At, C, Ct, oldVer, s1, C2):
    temp1 = A@oldVer@Ct + At@oldVer@C
    temp2 = s1*C2@(oldVer.sum(axis=0))
    temp3 = np.ones(A.shape[0])
    temp4 = np.tensordot(temp3, temp2, axes=0)
    NewQ = temp1-temp4
    return NewQ


# simple (doesnt need to be tested)
def advancedHitsVect(G, maxiter=1000):
    A, At, C, Ct, n, s1, s2 = __setup__(G)
    oldVer = __getNorm__(np.random.random([len(G), 4]))
    diff = 100
    count = 0
    C2 = C+Ct
    while diff > 10**(-8):
            count += 1
            if count > maxiter:
                print('did not converge')
                break
            newVersion = __hitsmk2__(A, At, C, Ct, oldVer, s1, C2)
            newVersion = __getNorm__(newVersion)
            diff = np.abs(newVersion - oldVer).max()
            oldVer = newVersion
    return newVersion


# tested
def __hitsmk2OneAtATime__(As, Ats, t1, C, Ct, oldVer, s1, s2, totals):
    NewQ = As[t1]@oldVer@Ct + Ats[t1]@oldVer@C - s1*(C+Ct)@(totals)
    NewQ -= NewQ.min()
    sum1 = NewQ.sum()
    if sum1 > 10**(-14):
        NewQ /= sum1
    else:
        # generate once rather than dynamically
        NewQ = np.array([0.25, 0.25, 0.25, 0.25])
    # see if indexing is faster
    temp3 = (NewQ - oldVer[t1, :]).ravel()
    # check if there is a numpy version
    max1 = abs(temp3).max()
    totals += temp3
    oldVer[t1, :] = NewQ
    return max1


# testing not needed
def __setupOneAtATime__(A, At, oldVer):
    As = [A[i:i+1, :] for i in range(A.shape[0])]
    Ats = [At[i:i+1, :] for i in range(A.shape[0])]
    totals = oldVer.sum(axis=0).astype('float64')
    return As, Ats, totals


# testing not needed (simple wrapper)
def advancedHitsVectFullPlus1AtTime(G):
   A,At,C,Ct,n,s1,s2 = __setup__(G)
   oldVer = np.random.random([len(G), 4])
   oldVer = __getNorm__(oldVer)
   count = 0
   diff=100
   while diff > 10**(-8):
           count += 1
           if count>2000: #len(G)*20:
               print('One time setup')
               return advancedHitsVectOneAtATime(G,oldVer)
           newVersion = __hitsmk2vectorised__(A, At, C, Ct, oldVer,s1,s2)
           newVersion = __getNorm__(newVersion)
           diff = np.abs(newVersion-oldVer).max()
           oldVer = newVersion
   return newVersion


# testing not needed (single wrapper around other functions
def advancedHitsVectOneAtATime(G, oldVer=None):
    A, At, C, Ct, n, s1, s2 = __setup__(G)
    if oldVer is None:
        oldVer = np.random.random([len(G), 4])
    diff = 100
    count = 0
    oldVer = __getNorm__(oldVer)
    # Setup of one at a time variables
    As, Ats, totals = __setupOneAtATime__(A, At, oldVer)
    order = list(range(len(G)))
    updateFunc = __hitsmk2OneAtATime__
    while diff > 10**(-8):
            count += 1
            if count > 2000:
                print('did not converge')
                print(diff)
                break
            rd.shuffle(order)
            diff = 0
            for t1 in order:
                max1 = updateFunc(As, Ats, t1, C, Ct, oldVer, s1, s2, totals)
                diff = max(max1, diff)
    return oldVer

# Original Plus One at a Time


# testing not needed (simple wrapper)
def advancedHitsVectFullPlus1AtTime(G):
    A, At, C, Ct, n, s1, s2 = __setup__(G)
    oldVer = np.random.random([len(G), 4])
    oldVer = __getNorm__(oldVer)
    count = 0
    diff = 100
    C2 = C+Ct
    while diff > 10**(-8):
            count += 1
            if count > len(G)*20:
                print('One time setup')
                return advancedHitsVectOneAtATime(G, oldVer)
            newVersion = __hitsmk2__(A, At, C, Ct, oldVer, s1, C2)
            newVersion = __getNorm__(newVersion)
            diff = np.abs(newVersion-oldVer).max()
            oldVer = newVersion
    return newVersion



##########  Group Normalised Version  ##########

# Orig

# tested
def __hitsmk2Group__(A, At, C, Ct, oldVer, s1, C2):
    divisors = oldVer.sum(axis=0)
    assert((divisors > -10**-(14)).all())
    divisors[divisors < 10**(-14)] = 1
    oldVer = oldVer/divisors
    temp1 = A@oldVer@Ct + At@oldVer@C
    temp2 = s1*(C2)@(oldVer.sum(axis=0))
    temp3 = np.ones(A.shape[0])
    temp4 = np.tensordot(temp3, temp2, axes=0)
    NewQ = temp1-temp4
    return NewQ


# no test needed (simple wrapper)
def advancedHitsVectGroupNorm(G):
    A, At, C, Ct, n, s1, s2 = __setup__(G)
    C2 = C + Ct
    oldVer = __getNorm__(np.random.random([len(G), 4]))
    diff = 100
    count = 0
    while diff > 10**(-8):
            count += 1
            if count > 1000:
                print('did not converge')
                break
            newVer = __hitsmk2Group__(A, At, C, Ct, oldVer, s1, C2)
            newVer = __getNorm__(newVer)
            diff = np.abs(newVer - oldVer).max()
            oldVer = newVer
    return newVer


# One At A Time

def __hitsmk2OneAtATimeGroup__(As, Ats, t1, C, Ct, oldVer, s1, s2, flag=False):
    divisors = oldVer.sum(axis=0)
    assert((divisors > -10**-(14)).all())
    minusTerm = np.array([1, 1, 1, 1])
    minusTerm[divisors < 10**(-14)] = 0
    divisors[divisors < 10**(-14)] = 1
    oldVer1 = oldVer/divisors
    temp1 = As[t1]@oldVer1@Ct + Ats[t1]@oldVer1@C
    temp2 = s1*(C+Ct)@(np.array(minusTerm))
    NewQ = temp1-temp2
    NewQ = NewQ - NewQ.min()
    sum1 = NewQ.sum()
    if sum1 > 10**(-14):
        NewQ = NewQ/sum1
    else:
        NewQ = np.array([0.25, 0.25, 0.25, 0.25])
    temp3 = (NewQ - oldVer[t1, :]).ravel()
    max1 = abs(temp3).max()
    oldVer[t1, :] = NewQ
    return oldVer, max1


def advancedHitsVectOneAtATime_Group(G, oldVer=None):
    A, At, C, Ct, n, s1, s2 = __setup__(G)
    if oldVer is None:
        oldVer = np.random.random([len(G), 4])
    diff = 100
    count = 0
    oldVer = __getNorm__(oldVer)
    # Setup of one at a time variables
    order = list(range(len(G)))
    As = [A[i:i+1, :] for i in range(len(G))]
    Ats = [At[i:i+1, :] for i in range(len(G))]
    updateFunc = __hitsmk2OneAtATimeGroup__
    while diff > 10**(-8):
            count += 1
            if count > 2000:
                print('did not converge')
                print(diff)
                break
#           oldVer1=oldVer.copy()
            rd.shuffle(order)
            diff = 0
            for t1 in order:
                newVer, max1 = updateFunc(As, Ats, t1, C, Ct, oldVer, s1, s2)
                diff = max(max1, diff)
                oldVer = newVer
    return newVer


# Orig + One At A Time


def advancedHitsVect_Group_FullPlus1AtTime(G):
    A, At, C, Ct, n, s1, s2 = __setup__(G)
    C2 = C + Ct
    oldVer = np.random.random([len(G), 4])
    oldVer = __getNorm__(oldVer)
    count = 0
    diff=100
    while diff > 10**(-8):
            count += 1
            if count>2000: #len(G)*20:
                print('One time setup')
                return advancedHitsVectOneAtATime_Group(G,oldVer)
            newVersion = __hitsmk2Group__(A, At, C, Ct, oldVer, s1, C2)
            newVersion = __getNorm__(newVersion)
            diff = np.abs(newVersion-oldVer).max()
            oldVer = newVersion
    return newVersion
