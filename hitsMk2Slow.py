
def __hitsmk2vectorisedOneAtATime__(A, At, C, Ct, oldVer,s1,s2):
    t1=rd.randint(0,A.shape[0]-1) 
    temp1 = A@oldVer@Ct + At@oldVer@C
    temp2 = s1*(C+Ct)@(oldVer.sum(axis=0))
    temp3 = np.ones(A.shape[0])
    temp4 = np.tensordot(temp3, temp2, axes=0)
    NewQ = temp1-temp2
    result = oldVer.copy()
    result[t1,:]=NewQ[t1,:]
    return result 

def advancedHitsVectOneAtATime(G):
    A,At,C,Ct,n,s1,s2 = __setup__(G)
    oldVer = np.random.random([len(G), 4])
    oldVer = __getNorm__(oldVer)
    count = 0
    while diff > 10**(-8):
            count += 1
            if count>1000:
                print('did not converge')
                print(diff)
                break
            oldVer1=oldVer.copy()
            for i in range(len(G)):
                newVersion = __hitsmk2vectorisedOneAtATime__(A, At, C, Ct, oldVer,s1,s2)
                newVersion = __getNorm__(newVersion)
                oldVer = newVersion
            diff = np.abs(newVersion-oldVer1).max()
            print(diff)
    return newVersion


