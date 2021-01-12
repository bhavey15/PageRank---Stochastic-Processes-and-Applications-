import numpy as np
def PageRank(A):
    n=len(A)
    A=np.concatenate((A,np.ones((1,n))))
    for i in range(n):
        A[i][i]=A[i][i]-1

    # print(p)
    p=np.dot(A.T,A)
    # print(A)
    A_inv=np.dot(np.linalg.inv(p),A.T)
    x=[0]*(n)+[1]

    return A_inv.dot(x)

def find_A(L):
    n=len(L)
    M=find_M(L)
    LM_inv=np.dot(L,np.linalg.inv(M))
    A=np.zeros((n,n))
    # print(LM_inv)
    for i in range(n):
        for j in range(n):
            if L[i][j]==1:
                A[i][j]=0.85*LM_inv[i][j]
            A[i][j] += 0.15*1/n
    return A

def find_M(L):
    n=len(L)
    M=np.zeros((n,n))
    for j in range(n):
        for i in range(n):
            M[j][j]+=L[i][j]
    return M
# n=50
# L = (np.random.rand(n*n) > 0.1).astype(int).reshape(n,n)
# A=A/np.sum(A,axis=1,keepdims=True)
# L=[[0,0,1,0,0],[1,0,0,0,0],[0,1,0,0,0],[0,0,0,0,1],[0,0,0,1,0]]
L=[[0,1,1],[1,0,1],[0,0,0]]
# L=[[1,1,1,0],[1,0,1,0],[0,0,0,1],[1,1,1,1]]
# L=[[1,1],[0,0]]
A=find_A(L)

# print(A)
# print(p)
P=PageRank(A)
print('Page rank \n', P)
print('Sum : ',sum(P))
print(np.argsort(-1*P))





# import numpy as np
# # from scipy.sparse import linalg as la
# #Let number of pages be n
# n=50
# import matplotlib.pyplot as plt
# A = (np.random.rand(n*n) > 0.1).astype(int).reshape(n,n)
# A=A/np.sum(A,axis=1,keepdims=True)
# B = np.ones((n,n))/n
# # df=[0.15,0.25,0.35,0.45,0.5,0.8]
# d=0.15
# # for d in df:
# M = (1-d)*A + d*B
# iters=50
# p=np.ones((1,n))*0.1
# states=np.zeros(iters)
# for i in range(iters):
# 	plt.plot(np.argsort(-1*p)[0],label=str(i))
# 	states[i]=np.argmax(p)
# 	if i<3:
# 		print(p)
# 	p=p@M
# # print(p)
# # plt.plot(states,label='df '+str(d))
# plt.legend()
# plt.show()


# # print(p@M)
# # w,v=la.eigs(M,k=1,sigma=1)
# # print(v)

