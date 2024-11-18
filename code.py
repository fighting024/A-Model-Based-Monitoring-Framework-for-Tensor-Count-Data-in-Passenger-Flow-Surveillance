
'''
This code is for the paper titled "A Model-Based Monitoring Framework for Tensor Count Data in Passenger Flow Surveillance".
'''

'''
Software: Python 3.10.9
Computer Hardware:  MacBook Pro with Apple M2 chip, featuring 8 cores (4 performance and 4 efficiency), 16 GB of RAM.
'''

import math
import time
import numpy as np
import scipy
from scipy.special import logsumexp
from scipy.special import gammaln
from scipy.optimize import minimize
from scipy.optimize import root
import matplotlib.pyplot as plt
import tensorly as tl
from scipy.linalg import sqrtm
from sklearn.utils.extmath import fast_logdet

'''
The code below generates the count tensor data.
'''

def mode_n_product(x, m, mode):
    x = np.asarray(x)
    m = np.asarray(m)
    if mode <= 0 or mode % 1 != 0:
        raise ValueError('`mode` must be a positive interger')
    if x.ndim < mode:
        raise ValueError('Invalid shape of X for mode = {}: {}'.format(mode, x.shape))
    if m.ndim != 2:
        raise ValueError('Invalid shape of M: {}'.format(m.shape))
    return np.swapaxes(np.swapaxes(x, mode - 1, -1).dot(m.T), mode - 1, -1)

def Matrixing(tensor,mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1),order="F")

def generate_poiTTens(Sigma1,Sigma2,Sigma3,Mu,nT):
   
    Tens = np.random.normal(0, 1, size=(d1, d2, d3, nT))
    TTens = mode_n_product(Tens, sqrtm(Sigma1), mode=1)
    TTens = mode_n_product(TTens, sqrtm(Sigma2), mode=2)
    TTens = mode_n_product(TTens, sqrtm(Sigma3), mode=3)
    TTens = (TTens.transpose(3, 0, 1, 2) + Mu).transpose(1, 2, 3, 0)
    eTTens = np.exp(TTens) 
    poiTTens = np.random.poisson(lam=eTTens)
    
    return poiTTens

'''
The code below implements the parameter estimation by VGA in Phase I.
'''

E1 = []
for w in range(d1):
    E1_w = np.zeros((d1, d1))
    E1_w[w, w] = 1
    E1.append(E1_w)
E2 = []
for w in range(d2):
    E2_w = np.zeros((d2, d2))
    E2_w[w, w] = 1
    E2.append(E2_w)
E3 = []
for w in range(d3):
    E3_w = np.zeros((d3, d3))
    E3_w[w, w] = 1
    E3.append(E3_w)

def tplnVGA(poiTTens,nT):

    m = [None]*nT
    vec_m = [None]*nT 
    for i in range(nT):
        m[i] = np.log(poiTTens[:, :, :, i]+1)
        vec_m[i] = m[i].flatten(order='F').reshape(-1, 1)
    mu = np.mean([vec_m[i] for i in range(nT)], axis=0)
    
    delta1 = [np.eye(d1)]*nT
    delta2 = [np.eye(d2)]*nT
    delta3 = [np.eye(d3)]*nT

    Sigma1 = np.eye(d1)
    Sigma2 = np.eye(d2)
    Sigma3 = np.eye(d3)
   
    S2 = [None]*nT
    two=[None]*nT
    it = 1
    check = 0
    elbo = []
    loglik = []
    aloglik = []
    aloglik[:3] = [0]*3
    itMax = 200
    eps = 0.0005
    
    w = [None]*nT
    for i in range(nT):
        w[i] = lmbda * (1 - lmbda)**(nT - i)
    w_sum = np.sum(w)

    while check == 0:
        
        iSigma1 = np.linalg.inv(Sigma1)
        iSigma2 = np.linalg.inv(Sigma2)
        iSigma3 = np.linalg.inv(Sigma3)
        S1 = np.kron(np.kron(Sigma3, Sigma2), Sigma1) 

        for i in range(nT):

            Omega = np.diag(np.array([np.dot(np.exp(vec_m[i].T + 0.5 * np.diag(np.kron(np.kron(delta3[i], delta2[i]), delta1[i])).reshape(1, -1)), np.diag(np.kron(np.kron(delta3[i], delta2[i]), E1[w])).reshape(-1, 1)) for w in range(d1)]).flatten())
            delta1[i] = d2 * d3 * np.linalg.inv(Omega + iSigma1 * np.trace(iSigma2 @ delta2[i]) * np.trace(iSigma3 @ delta3[i]))
            
            Omega = np.diag(np.array([np.dot(np.exp(vec_m[i].T + 0.5 * np.diag(np.kron(np.kron(delta3[i], delta2[i]), delta1[i])).reshape(1, -1)), np.diag(np.kron(np.kron(delta3[i], E2[w]), delta1[i])).reshape(-1, 1)) for w in range(d2)]).flatten())
            delta2[i] = d1 * d3 * np.linalg.inv(Omega + iSigma2 * np.trace(iSigma1 @ delta1[i]) * np.trace(iSigma3 @ delta3[i]))
            
            Omega = np.diag(np.array([np.dot(np.exp(vec_m[i].T + 0.5 * np.diag(np.kron(np.kron(delta3[i], delta2[i]), delta1[i])).reshape(1, -1)), np.diag(np.kron(np.kron(E3[w], delta2[i]), delta1[i])).reshape(-1, 1)) for w in range(d3)]).flatten())
            delta3[i] = d1 * d2 * np.linalg.inv(Omega + iSigma3 * np.trace(iSigma1 @ delta1[i]) * np.trace(iSigma2 @ delta2[i]))
            
            S1 = np.kron(np.kron(Sigma3, Sigma2), Sigma1)   
            S2[i] = np.kron(np.kron(delta3[i], delta2[i]), delta1[i])
            
            vec_m[i] = vec_m[i] - np.linalg.inv(np.diag(np.exp(vec_m[i] + 0.5 * np.diag(S2[i]).reshape(-1,1)).flatten())+np.linalg.inv(S1)) @ (np.exp(vec_m[i] + 0.5 * np.diag(S2[i]).reshape(-1, 1)) + np.linalg.inv(S1)@(vec_m[i]-mu)-vec_dataset[i])  #vec_m[i]是vec的变分均值

        mu = np.sum([w[i]*vec_m[i] for i in range(nT)],axis=0)/w_sum 
        
        tensor_vec_m = [vec_m[i].reshape((d1,d2,d3),order='F') for i in range(nT)] 
        tensor_mu =  mu.reshape((d1,d2,d3),order='F') 
        
        Sigma1 = np.sum([w[i]*delta1[i]*np.trace(iSigma2@delta2[i])*np.trace(iSigma3@delta3[i]) + w[i]*Matrixing(tensor_vec_m[i]-tensor_mu,0)@np.linalg.inv(np.kron(Sigma3,Sigma2))@(Matrixing(tensor_vec_m[i]-tensor_mu,0)).T for i in range(nT)], axis=0)/(d2*d3*w_sum)
        iSigma1 = np.linalg.inv(Sigma1)
        Sigma2 = np.sum([w[i]*delta2[i]*np.trace(iSigma1@delta1[i])*np.trace(iSigma3@delta3[i]) + w[i]*Matrixing(tensor_vec_m[i]-tensor_mu,1)@np.linalg.inv(np.kron(Sigma3,Sigma1))@(Matrixing(tensor_vec_m[i]-tensor_mu,1)).T for i in range(nT)], axis=0)/(d1*d3*w_sum)
        iSigma2 = np.linalg.inv(Sigma2)
        Sigma3 = np.sum([w[i]*delta3[i]*np.trace(iSigma1@delta1[i])*np.trace(iSigma2@delta2[i]) + w[i]*Matrixing(tensor_vec_m[i]-tensor_mu,2)@np.linalg.inv(np.kron(Sigma2,Sigma1))@(Matrixing(tensor_vec_m[i]-tensor_mu,2)).T for i in range(nT)], axis=0)/(d1*d2*w_sum)
        
        Sigma1=Sigma1/Sigma1[0,0]
        Sigma2=Sigma2/Sigma2[0,0]
        Sigma3=Sigma3
        one = d2*d3*fast_logdet(Sigma1) + d1*d3*fast_logdet(Sigma2) + d1*d2 *fast_logdet(Sigma3)
        for i in range(nT):
            two[i] = d2*d3 *fast_logdet(delta1[i]) + d1*d3 * fast_logdet(delta2[i]) + d1*d2 * fast_logdet(delta3[i])   
        
        iSigma1 = np.linalg.inv(Sigma1)
        iSigma2 = np.linalg.inv(Sigma2)
        iSigma3 = np.linalg.inv(Sigma3)
        iS1=np.linalg.inv(S1)
        for i in range(nT):
            elbo_i = w[i]*(vec_m[i].reshape(1, -1) @ vec_dataset[i].reshape(-1, 1) - np.sum(gammaln((vec_dataset[i]) + 1))-np.sum(np.exp(vec_m[i].reshape(-1, 1) + 0.5 * np.diag(S2[i]).reshape(-1, 1))) - 0.5*one + 0.5*two[i] -0.5*((vec_m[i]-mu).T)@iS1@(vec_m[i]-mu) - 0.5*np.trace(iSigma1@delta1[i])*np.trace(iSigma2@delta2[i])*np.trace(iSigma3@delta3[i]) + 0.5*q)  
            elbo.append(elbo_i)  

        elbo_D = np.sum(elbo[-nT:], axis=0)  
        loglik.append(elbo_D)  

        if it > 3:
            if (loglik[-2] - loglik[-3]) == 0:
                check = 1
            else:
                a = (loglik[-1] - loglik[-2]) / (loglik[-2] - loglik[-3])
                aa = (loglik[-1] - loglik[-2])/ (1 - a)
                aloglik.append(loglik[-2] + aa)
                if abs(aloglik[-1] - aloglik[-2]) <= eps:
                    check = 1
                else:
                        check = check

        it = it+1

        if it == itMax:
            check = 1
            print("No convergence")

    Results = {
        'vec_Mu':mu,
        'Mu': tensor_mu,
        'Sigma1': Sigma1,
        'Sigma2': Sigma2,
        'Sigma3': Sigma3,
        'S1':S1,
        'finalELBO':loglik[-1],
        'variationalMu':vec_m,
        'delta1':delta1,
        'delta2':delta2,
        'delta3':delta3,
        'S2':S2
    }

    return Results

'''
 The code below implements the Laplace approximation approach in Phase II.
'''

def H0_laplace(poiTTens,nT):
    
    loglik0=[]
    loglik=[]
    vec_dataset1 = [None]*nT
    for i in range(nT):
        vec_dataset1[i] = poiTTens[:, :, :, i].flatten(order='F')
        
    w = [None]*nT
    for i in range(nT):
        w[i] = lmbda * (1 - lmbda)**(nT - i)
    w_sum = np.sum(w)
        
    vec_m = [None]*nT 
    vec_m0 = [None]*nT
    m=[None]*nT
    mu = np.zeros([q]) 
    for i in range(nT):
       m[i] = np.zeros([q]) 
    
    #H1 
    for i in range(nT):     
        def g(x):
            g = np.exp(x)-vec_dataset1[i] + np.dot(iS10,x-mu)
            return(g)
        vec_m[i] = root(g, m[i]).x
    mu = (np.sum([w[i]*vec_m[i] for i in range(nT)],axis=0)) / w_sum 

    y = [None]*nT
    for i in range(nT):
        y[i]=vec_m[i].reshape((d1,d2,d3),order='F')
    
    #H0
    for i in range(nT):     
        def g(x):
            g = np.exp(x)-vec_dataset1[i] + np.dot(iS10,x-Mu0.flatten(order='F'))
            return(g)
        vec_m0[i] = root(g, m[i]).x
 
    y0 = [None]*nT
    for i in range(nT):
        y0[i]=vec_m0[i].reshape((d1,d2,d3),order='F')
    
    one = 0.5*d2*d3*fast_logdet(Sigma10)+ 0.5*d1*d3*fast_logdet(Sigma20)+ 0.5*d1*d2*fast_logdet(Sigma30) 
    
    for i in range(nT): 
        loglik0_i = w[i]*(-0.5*np.dot(np.dot((vec_m[i]-Mu0).T,iS10),vec_m[i]-Mu0) - np.sum(np.exp(y0[i]))+np.sum(np.multiply(poiTTens[:,:,:,i],y0[i]))-one - np.sum(gammaln(poiTTens[:,:,:,i] + 1))-0.5*np.log(np.det(-(np.diag(np.exp(vec_m0[i]))-iS10))))
        loglik0.append(loglik0_i) 
    
    for i in range(nT): 
        loglik_i = w[i]*(-0.5*np.dot(np.dot((vec_m[i]-mu).T,iS10),vec_m[i]-mu) - np.sum(np.exp(y[i]))+np.sum(np.multiply(poiTTens[:,:,:,i],y[i]))-one - np.sum(gammaln(poiTTens[:,:,:,i] + 1))-0.5*np.log(np.det(-(np.diag(np.exp(vec_m[i]))-iS10))))
        loglik.append(loglik_i)
        
    Results = {'H1_laplace' : np.sum(loglik),
                'H0_laplace' : np.sum(loglik0)}

    return Results

'''
The code below finds ARL with TWLRT control chart
'''

def ARL0_la(k,Sigma1, Sigma2, Sigma3, Mu, Mu1, h, M, m):
    
    rl0 = np.zeros([M])
    for i in range(M):
        j = 11
        W = 0
        all_poiTTens = generate_poiTTens(Sigma1,Sigma2,Sigma3, Mu1, 1010)
        while W < h and j <= 1000:
            
            poiTTens = all_poiTTens[:, :, :, j-m:j]
            res_laplace = H0_laplace(poiTTens,m)
            
            W = res_laplace['H1_laplace'] - res_laplace['H0_laplace']
            j +=1
            
        rl0[i] = j-11
        
    arl0 = np.round(np.mean(rl0),2)
    sdrl0 = np.round(np.std(rl0),2)
    a0 = [arl0, sdrl0]

    return a0

'''
The code below finds ARL with TEWMA control chart
'''

def MLE_TND(poiTTens, nT):

    vec_dataset = [None]*nT
    for i in range(nT):
        vec_dataset[i] = poiTTens[:, :, :, i].flatten(order='F').reshape(-1, 1)

    mu = np.mean(vec_dataset,axis=0)
    muu = mu.reshape((d1,d2,d3),order='F')
            
    Sigma1_new = np.sum([Matrixing(poiTTens[:, :, :, i]-muu, 0)@(Matrixing(poiTTens[:, :, :, i]-muu,0)).T for i in range(nT)],axis=0)/(nT*d2*d3)
    Sigma2_new = np.sum([Matrixing(poiTTens[:, :, :, i]-muu, 1)@(Matrixing(poiTTens[:, :, :, i]-muu,1)).T for i in range(nT)],axis=0)/(nT*d1*d3)
    Sigma3_new = np.sum([Matrixing(poiTTens[:, :, :, i]-muu, 2)@(Matrixing(poiTTens[:, :, :, i]-muu,2)).T for i in range(nT)],axis=0)/(nT*np.trace(Sigma2_new)*np.trace(Sigma1_new))      
    
    it = 1
    check = 0
    itMax = 200
    eps = 0.0005

    while check == 0:

        Sigma1_old=Sigma1_new
        Sigma2_old=Sigma2_new
        Sigma3_old=Sigma3_new
        
        Sigma1_new = np.sum([Matrixing(poiTTens[:, :, :, i]-muu, 0)@np.linalg.inv(np.kron(Sigma3_old,Sigma2_old))@(Matrixing(poiTTens[:, :, :, i]-muu,0)).T for i in range(nT)],axis=0)/(nT*d2*d3)
        Sigma2_new = np.sum([Matrixing(poiTTens[:, :, :, i]-muu, 1)@np.linalg.inv(np.kron(Sigma3_old,Sigma1_new))@(Matrixing(poiTTens[:, :, :, i]-muu,1)).T for i in range(nT)],axis=0)/(nT*d1*d3)
        Sigma3_new = np.sum([Matrixing(poiTTens[:, :, :, i]-muu, 2)@np.linalg.inv(np.kron(Sigma2_new,Sigma1_new))@(Matrixing(poiTTens[:, :, :, i]-muu,2)).T for i in range(nT)],axis=0)/(nT*d1*d2)
   
        if  np.linalg.norm(Sigma1_new- Sigma1_old,ord=2)<=eps and np.linalg.norm(Sigma2_new- Sigma2_old,ord=2)<=eps and np.linalg.norm(Sigma3_new- Sigma3_old,ord=2)<=eps :
            check=1
        else:
            check=check

        it = it+1

        if it == itMax:
            check = 1
            print("no convergence")

    Results = {
        'Mu': muu,
        'Sigma1': Sigma1_new,
        'Sigma2': Sigma2_new,
        'Sigma3': Sigma3_new,
        'iterations': it
    }
    return Results

def ARL0_TND(Sigma1, Sigma2, Sigma3, Mu, Mu1, h, M):
    
    rl0 = np.zeros([M])
    for i in range(M):
        j = 1
        E = np.zeros([d1,d2,d3])
        TEWMA = 0
        all_poiTTens = generate_poiTTens(Sigma1,Sigma2,Sigma3, Mu1, 1010)
        while TEWMA  < h and j <= 1000:
 
            poiTTens = all_poiTTens[:,:,:,j]
            poiTTens = poiTTens-Muic
            poiTTens = mode_n_product(poiTTens, np.linalg.inv(sqrtm(Sigma1ic)), mode=1)
            poiTTens = mode_n_product(poiTTens, np.linalg.inv(sqrtm(Sigma2ic)), mode=2)
            poiTTens = mode_n_product(poiTTens, np.linalg.inv(sqrtm(Sigma3ic)), mode=3) 
            
            E = (1-lmbda)*E + lmbda*poiTTens
            TEWMA  = (np.linalg.norm(E))**2
            j +=1
            
        rl0[i] = j-1

    arl0 = np.round(np.mean(rl0),2)
    sdrl0 = np.round(np.std(rl0),2)
    a0 = [arl0, sdrl0]
    
    return a0


'''
The code below finds ARL with TROD control chart
'''

result = tl.decomposition.parafac(Pics, rank, random_state=123)
weights, factors = result

V1 = factors[0]
V2 = factors[1]
V3 = factors[2]
V4 = factors[3]
lambdas_nP = weights

lam = np.zeros([nP,rank]) 
for i in range(rank):
    lam[:,i]=lambdas_nP[i]*V4[:,i]  
basis1 = np.kron(np.kron(V3[:, 0], V2[:, 0]), V1[:, 0])
basis1 = basis1.reshape((d1,d2,d3),order='F')
basis2 = np.kron(np.kron(V3[:, 1], V2[:, 1]), V1[:, 1])
basis2 = basis2.reshape((d1,d2,d3),order='F')
basis3 = np.kron(np.kron(V3[:, 2], V2[:, 2]), V1[:, 2])
basis3 = basis3.reshape((d1,d2,d3),order='F')
Sigma = np.cov(lam, rowvar=False)
StdM_trod = np.linalg.inv(sqrtm(Sigma))

def ARL_TROD(basis1, basis2, basis3, StdM_trod, Mu, Mu1,h1, h2, N):
    
    arlv = np.zeros(N)
    countT = 0
    countQ = 0

    for i in range(N):
        count = 0
        Q = 0
        T2 = 0
        while((T2 <= h1) & (Q <= h2) & (count <= 1000)):

            Tens = np.random.normal(0, 1, size=(d1, d2, d3))
            TTens = mode_n_product(Tens, sqrtm(Sigma1), mode=1)
            TTens = mode_n_product(TTens, sqrtm(Sigma2), mode=2)
            TTens = mode_n_product(TTens, sqrtm(Sigma3), mode=3)
            TTens = TTens + Mu1

            eTTens=np.exp(TTens)
            poiTTens=np.random.poisson(lam=eTTens)
            
            V321 = khatri_rao(V3, khatri_rao(V2, V1))
            newlambdas = poiTTens.flatten(order='F')@ (np.linalg.inv(V321 @ (V321.T) ) @ V321) 

            newlambdas = np.dot(StdM_trod, newlambdas)
            T2 = np.sum((newlambdas)**2) 
            T2 = math.sqrt(T2)/10000 

            eplion_new = (poiTTens - newlambdas[0]*basis1-newlambdas[1]*basis2-newlambdas[2]*basis3).flatten(order='F')

            Q = np.sum((eplion_new)**2)
            Q = math.sqrt(Q)/100000000 
            count = count+1

            if(T2 > h1):
                countT = countT+1
            if(Q > h2):
                countQ = countQ+1

        arlv[i] = count
    
    a0 = [np.round(np.mean(arlv),2),np.round(np.std(arlv),2)]
    
    return a0

'''
The code below finds ARL with MPCA control chart
'''

Phi = 0
for i in range(0, nP):
    Phi = Phi+np.dot(Matrixing(Pics[:, :, :, i], mode=0),
                     np.transpose(Matrixing(Pics[:, :, :, i], mode=0)))
evalue1, evector1 = np.linalg.eig(Phi)  
sorted_indices = np.argsort(evalue1)  
U1 = evector1[:, sorted_indices[:-P1-1:-1]]

Phi = 0
for i in range(0, nP):
    Phi = Phi+np.dot(Matrixing(Pics[:, :, :, i], mode=1),
                     np.transpose(Matrixing(Pics[:, :, :, i], mode=1)))
evalue2, evector2 = np.linalg.eig(Phi)
sorted_indices = np.argsort(evalue2)
U2 = evector2[:, sorted_indices[:-P2-1:-1]]

Phi = 0
for i in range(0, nP):
    Phi = Phi+np.dot(Matrixing(Pics[:, :, :, i], mode=2),
                     np.transpose(Matrixing(Pics[:, :, :, i], mode=2)))
evalue3, evector3 = np.linalg.eig(Phi)
sorted_indices = np.argsort(evalue3)
U3 = evector3[:, sorted_indices[:-P3-1:-1]]

for k in range(0, K):
    UPhi1 = np.kron(U2, U3)
    UPhi2 = np.kron(U3, U1)
    UPhi3 = np.kron(U1, U2)
    Phi1 = 0
    Phi2 = 0
    Phi3 = 0
    for i in range(0, nP):
        Phi1 = Phi1+np.dot(np.dot(np.dot(Matrixing(Pics[:, :, :, i], mode=0), UPhi1),
                                  np.transpose(UPhi1)), np.transpose(Matrixing(Pics[:, :, :, i], mode=0)))
        Phi2 = Phi2+np.dot(np.dot(np.dot(Matrixing(Pics[:, :, :, i], mode=1), UPhi2),
                                  np.transpose(UPhi2)), np.transpose(Matrixing(Pics[:, :, :, i], mode=1)))
        Phi3 = Phi3+np.dot(np.dot(np.dot(Matrixing(Pics[:, :, :, i], mode=2), UPhi3),
                                  np.transpose(UPhi3)), np.transpose(Matrixing(Pics[:, :, :, i], mode=2)))

    evalue1, evector1 = np.linalg.eig(Phi1)
    sorted_indices = np.argsort(evalue1)
    U1 = evector1[:, sorted_indices[:-P1-1:-1]]
    evalue2, evector2 = np.linalg.eig(Phi2)
    sorted_indices = np.argsort(evalue2)
    U2 = evector2[:, sorted_indices[:-P2-1:-1]]
    evalue3, evector3 = np.linalg.eig(Phi3)
    sorted_indices = np.argsort(evalue3)
    U3 = evector3[:, sorted_indices[:-P3-1:-1]]

OutputPics = mode_n_product(Pics, np.transpose(U1), 1)
OutputPics = mode_n_product(OutputPics, np.transpose(U2), 2)
OutputPics = mode_n_product(OutputPics, np.transpose(U3), 3)
OutputPics.shape

Sigma = np.cov(Matrixing(OutputPics, mode=3), rowvar=False)
StdM = np.linalg.inv(sqrtm(Sigma))

def ARL_MPCA(U1,U2,U3, StdM, Mu, Mu1, h1, h2, N):    
    
    arlv = np.zeros(N)
    countT = 0
    countQ = 0
    for i in range(N):
        count = 0
        Q = 0
        T2 = 0
        while((T2 <= h1) & (Q <= h2) & (count <= 1000)):

            randomTen = np.random.normal(0, 1, size=(d1, d2, d3))
            randomTen = mode_n_product(randomTen, sqrtm(Sigma1), mode=1)
            randomTen = mode_n_product(randomTen, sqrtm(Sigma2), mode=2)
            randomTen = mode_n_product(randomTen, sqrtm(Sigma3), mode=3)
            randomTen = randomTen + Mu1
 
            randomTen=np.exp(randomTen)
            randomTen=np.random.poisson(lam=randomTen)

            newTen=mode_n_product(randomTen, np.transpose(U1), 1)
            newTen=mode_n_product(newTen, np.transpose(U2), 2)
            newTen=mode_n_product(newTen, np.transpose(U3), 3)

            newTen2=newTen.flatten(order='F')
            newTen2 = np.dot(StdM, newTen2)
            T2 = np.sum((newTen2)**2)  

            remodelTen = mode_n_product(newTen, U1, 1)
            remodelTen = mode_n_product(remodelTen, U2, 2)
            remodelTen = mode_n_product(remodelTen, U3, 3)

            Q = np.sum((randomTen - remodelTen)**2)
            count = count+1

            if(T2 > h1):
                countT = countT+1
            if(Q > h2):
                countQ = countQ+1
        arlv[i] = count

    a0 = [np.round(np.mean(arlv),2),np.round(np.std(arlv),2)]
    
    return a0

'''
The code below finds ARL with MEWMA control chart
'''

poiTTens = generate_poiTTens(Sigma1,Sigma2,Sigma3,Mu,nT)
vec_dataset = [None]*nT
for i in range(nT):
    vec_dataset[i] = poiTTens[:, :, :, i].flatten(order='F')
poiss_vec_mean = np.mean(vec_dataset, axis=0).reshape(-1,1)
poiss_cov = np.cov(Matrixing(poiTTens, 3),rowvar=False)

def ARL0_T2(Sigma1, Sigma2, Sigma3, Mu1, h, M):
    
    rl0 = np.zeros([M])
    
    for i in range(M):
        j = 1
        E = np.zeros(q).reshape(-1,1)
        T2 = 0
        all_poiTTens = generate_poiTTens(Sigma1,Sigma2,Sigma3, Mu1, 1010)

        while T2 < h and j <= 1000:
            
            E = (1-lmbda)*E + lmbda*(all_poiTTens[:,:,:,j].flatten(order='F').reshape(-1, 1) - poiss_vec_mean)
            var = (lmbda*(1-(1-lmbda)**(2*j))/(2-lmbda))*poiss_cov
            T2 =  E.T @ np.linalg.inv(var) @ E
            
            j +=1
            
        rl0[i] = j-1

    arl = np.round(np.mean(rl0),2)
    sdrl = np.round(np.std(rl0),2)
    a = [arl, sdrl]
    
    return a

'''
Out-of-Control state
'''

shift = [-5,-4,-3,-2,-1,0.2,0.4,0.6,0.8, 1,1.5] 
Mu1 = [np.copy(Mu) for i in range(len(shift))]
for i in range(len(shift)):
    Mu1[i][0,0,0] = Mu1[i][0,0,0] + shift[i]
    Mu1[i][1,1,1] = Mu1[i][1,1,1] + shift[i]
    Mu1[i][2,2,2] = Mu1[i][2,2,2] + shift[i]
    Mu1[i][3,3,3] = Mu1[i][3,3,3] + shift[i]
    Mu1[i][4,4,4] = Mu1[i][4,4,4] + shift[i]

