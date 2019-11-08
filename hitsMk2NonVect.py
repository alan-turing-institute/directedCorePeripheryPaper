import numpy as np
import networkx as nx
# tested
def __getNormNonVect__(p_in, p_out, c_in, c_out):
        p_in = np.array(p_in, dtype=np.float)
        c_in = np.array(c_in, dtype=np.float)
        p_out = np.array(p_out, dtype=np.float)
        c_out = np.array(c_out, dtype=np.float)
        x123 = np.minimum(p_in, p_out)
        y123 = np.minimum(c_in, c_out)
        m1 = np.minimum(x123,y123)
        p_in-=m1
        c_in-=m1
        p_out-=m1
        c_out-=m1
#        for i in range(len(p_in)):
#            m1 = float(min([p_in[i],c_in[i],p_out[i],c_out[i]]))
#            print(m1)
#            for x in [p_in,c_in,p_out,c_out]:
#                if x[i]==m1:
#                   x[i]=0
#                else:
#                   x[i]-=m1
#            t1 = min([p_in[i],c_in[i],p_out[i],c_out[i]])
#            if t1<0:
#               import pdb
#               pdb.set_trace()
        m1 = p_in + p_out+c_in + c_out
        mask1 = abs(m1) < (10**(-10))
        # deal with the 0 case
        p_in += mask1*0.25
        c_in += mask1*0.25
        p_out += mask1*0.25
        c_out += mask1*0.25
        m1[mask1] = 1.0

        p_in = np.divide(p_in, m1)
        c_in = np.divide(c_in, m1)
        p_out = np.divide(p_out, m1)
        c_out = np.divide(c_out, m1)
        return p_in, p_out, c_in, c_out

def __makeMatrices__(G):
    A = nx.to_numpy_array(G,sorted(G.nodes()))
    mA = 1-A
    At = A.T
    mAt = mA.T
    return A, mA, At, mAt


def __runOneAdvancedHitsCycle__(p_in, p_out, c_in, c_out,A, mA, At, mAt,q1):

        # update each of the values
        p_inN  = updatePin_mk2(  p_in, p_out, c_in, c_out, A, mA, At, mAt, q1)
        c_outN = updateCout_mk2( p_in, p_out, c_in, c_out, A, mA, At, mAt, q1)
        p_outN = updatePout_mk2( p_in, p_out, c_in, c_out, A, mA, At, mAt, q1)
        c_inN  = updateCin_mk2(  p_in, p_out, c_in, c_out, A, mA, At, mAt, q1)

        # Take norm
        p_inN, p_outN, c_inN, c_outN = __getNormNonVect__(p_inN, p_outN, c_inN, c_outN)
        return p_inN, p_outN, c_inN, c_outN


def advancedHits(G: nx.DiGraph) -> np.array:
        n = len(G)

        # Make the matrices
        A, mA, At, mAt = __makeMatrices__(G)

        # Make inital vectors
        rnd=np.random.random
        p_in, p_out, c_in, c_out  = rnd([n, 1]), rnd([n, 1]), rnd([n, 1]), rnd([n, 1])

        # Take norm
        p_in, p_out, c_in, c_out = __getNormNonVect__(p_in, p_out, c_in, c_out)
        q1 = A.sum()
        for x in range(10000):
                p_inN, p_outN, c_inN, c_outN = __runOneAdvancedHitsCycle__(p_in, p_out, c_in, c_out,A, mA, At, mAt,q1)
                # # update each of the values
                # p_inN  = updatePin_mk2(G,  p_in, p_out, c_in, c_out, A, mA, At, mAt, q1)
                # c_outN = updateCout_mk2(G, p_in, p_out, c_in, c_out, A, mA, At, mAt, q1)
                # p_outN = updatePout_mk2(G, p_in, p_out, c_in, c_out, A, mA, At, mAt, q1)
                # c_inN  = updateCin_mk2(G,  p_in, p_out, c_in, c_out, A, mA, At, mAt, q1)

                # # Take norm
                # p_inN, p_outN, c_inN, c_outN = getNorm(G, p_inN, p_outN, c_inN, c_outN)
#               #  getNorm(G, p_inN, p_outN, c_inN, c_outN)

                diff1 = np.abs(p_in-p_inN).max()
                diff2 = np.abs(c_in-c_inN).max()
                diff3 = np.abs(p_out-p_outN).max()
                diff4 = np.abs(c_out-c_outN).max()
                diff = max([diff1, diff2, diff3, diff4])
                if diff < 10**(-8):
                        #print(diff)
                        break

                p_in, p_out, c_in, c_out = p_inN, p_outN, c_inN, c_outN

        res1 = np.concatenate([p_out, c_in, c_out, p_in], axis=1)
        print(diff)
        return res1


def updateCout_mk2(p_in, p_out, c_in, c_out, A, mA, At, mAt, q1):
        n = A.shape[0]
        s1 = q1/float(n*n)
        s2 = 1-q1/float(n*n)

        result  = s2*np.matmul(A, c_in+c_out-p_out+p_in)
        result += s1*np.matmul(mA, p_out-c_in-c_out-p_in)
        result += s2*np.matmul(At, c_out-c_in-p_out-p_in)
        result += s1*np.matmul(mAt, c_in-c_out+p_out+p_in)
        return result


def updatePout_mk2(p_in, p_out, c_in, c_out, A, mA, At, mAt, q1):
        n = A.shape[0]
        s1 = q1/float(n*n)
        s2 = 1-q1/float(n*n)

        result  = s2*np.matmul(A, c_in-c_out-p_out-p_in)
        result += s1*np.matmul(mA, c_out-c_in+p_out+p_in)
        result += s2*np.matmul(At, -c_in-c_out-p_out-p_in)
        result += s1*np.matmul(mAt, c_in+c_out+p_out+p_in)
        return result


def updatePin_mk2(p_in, p_out, c_in, c_out, A, mA, At, mAt, q1):
        n = A.shape[0]
        s1 = q1/float(n*n)
        s2 = 1-q1/float(n*n)

        result  = s2*np.matmul(A, -c_in-c_out-p_out-p_in)
        result += s1*np.matmul(mA, c_in+c_out+p_out+p_in)
        result += s2*np.matmul(At, -c_in+c_out-p_out-p_in)
        result += s1*np.matmul(mAt, c_in-c_out+p_out+p_in)
        return result


def updateCin_mk2(p_in, p_out, c_in, c_out, A, mA, At, mAt, q1):
        n = A.shape[0]
        s1 = q1/float(n*n)
        s2 = 1-q1/float(n*n)

        result  = s2*np.matmul(A, c_in-c_out-p_out-p_in)
        result += s1*np.matmul(mA, -c_in+c_out+p_out+p_in)
        result += s2*np.matmul(At, c_in+c_out+p_out-p_in)
        result += s1*np.matmul(mAt, -c_in-c_out-p_out+p_in)
        return result
