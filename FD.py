import numpy as np
from sklearn.model_selection import KFold # import KFold
from sklearn.linear_model import LinearRegression
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt
import math
import time
import data_generation

X, Y = data_generation.generate_data(missing_level =0.7)
LM = LinearRegression()

@jit(nopython=True)
def basis_construction(n1,n2,user,u,d,n):
    for col in range(n1,n2): #user1 data completion
        v = user[:,col]
        vOmega =v[v!=0]
        uOmega =u[v!=0,:]
        size1=vOmega.size
        uOmega= np.reshape(uOmega, (size1,d))
        vOmega= np.reshape(vOmega, (size1,1))
        w= np.linalg.inv((uOmega.T)@uOmega)@(uOmega.T)@vOmega
        u = np.ascontiguousarray(u)
        w = np.ascontiguousarray(w)
        p=u@w
        vtilda = v.copy()
        vtilda[vtilda == 0] = p[vtilda == 0,0]
        r=vtilda-p[:,0]
        mat=np.concatenate((np.eye(d,d), w), axis =1)
        norm = np.linalg.norm(r)
        zero = np.concatenate((np.zeros((1,d)),np.array([[norm]])),axis=1)
        mat=np.concatenate((mat, zero), axis =0)
        [Utilda, _, _]=np.linalg.svd(mat)
        norm = r/np.linalg.norm(r)
        u = np.column_stack((u,norm))
        u= u@Utilda
        u=u[:,0:d]
    return u

@jit(nopython=True)
def resnorm_clac(n1,n2,x,u,d,n,resnorm):
    for col in range(n1,n2): #user1 data resNorm
        v = x[:,col]
        vOmega =v[v!=0]
        uOmega =u[v!=0,:]
        size1=vOmega.size
        uOmega= np.reshape(uOmega, (size1,d))
        vOmega= np.reshape(vOmega, (size1,1))
        w= np.linalg.inv((uOmega.T)@uOmega)@(uOmega.T)@vOmega
        u = np.ascontiguousarray(u)
        w = np.ascontiguousarray(w)
        p=u@w
        vtilda = v
        vtilda[vtilda == 0] = p[vtilda == 0,0]
        r=vtilda-p[:,0]
        normr=np.linalg.norm(r)
        normp =np.linalg.norm(p)
        resnorm += normr/normp
    return resnorm

@jit(nopython=True)
def weight_matrix(n1,n2,user,u,d,n):
    W=np.zeros((d,n2-n1))
    for col in range(n1,n2):
        v = user[:,col]
        vOmega =v[v!=0]
        uOmega =u[v!=0,:]
        size1=vOmega.size
        uOmega= np.reshape(uOmega, (size1,d))
        vOmega= np.reshape(vOmega, (size1,1))
        w= np.linalg.inv((uOmega.T)@uOmega)@(uOmega.T)@vOmega
        W[:,col-n1] = w[:,0]
    return W
    
    

D = [5,10,20,30,40]
xtest = X[90:120,:].T
ytest = Y[90:120]

X = X[0:90,:]
Y = Y[0:90]
#shuffle
arr = np.arange(90)
for randomness in range(15):
    np.random.shuffle(arr)
    x = X[arr,:]
    y = Y[arr]
    user1=np.array(x[0:54])
    user2=np.array(x[54:81])
    user3 = np.array(x[81:90])    
    Y1=np.array(y[0:54])
    Y2=np.array(y[54:81])
    Y3=np.array(y[81:90])
    Matrixerror = {}
    for rank in D:
        K=[2,4,5] if rank==5 else K=np.linspace(rank//5,rank,5, dtype = int)
        for k in K:
            kf = KFold(n_splits=5) 
            train_test_splits = [
    list(kf.split(user)) for user in [user1, user2, user3]
]
               
            for index in range(0,5):
                train_indices = [split[index][0] for split in train_test_splits]
                test_indices = [split[index][1] for split in train_test_splits]

                # Concatenate training and testing data
                tray = np.concatenate([Y1[train_indices[0]], Y2[train_indices[1]], Y3[train_indices[2]]])
                tsty = np.concatenate([Y1[test_indices[0]], Y2[test_indices[1]], Y3[test_indices[2]]])
                
                user_train = np.concatenate([user1[train_indices[0]], user2[train_indices[1]], user3[train_indices[2]]], axis=0).T
                user_test = np.concatenate([user1[test_indices[0]], user2[test_indices[1]], user3[test_indices[2]]], axis=0).T

                n1 = 43
                n2 = 21
                n3 = 7
                n = 150 # 
                d=rank # number of features in time intervals
                
                H = np.random.randn(n, d)
                uh, _, vh = np.linalg.svd(H, full_matrices=False)
                u = uh @ vh
               
                ### user side
                for i in range(20): 
                    u = basis_construction(0,n1,user,u,d,n)
                    u = basis_construction(n1,n1+n2,user,u,d,n)
                    u = basis_construction(n1+n2,n1+n2+n3,user,u,d,n)
              
                ###########user1:
                b1=weight_matrix(0,n1,user,u,d,n)
                
                ###########user2:
                b2=weight_matrix(n1,n1+n2,user,u,d,n)
                    
                ###########user3:
                b3=weight_matrix(n1+n2,n1+n2+n3,user,u,d,n)
                ### server side
                B = np.concatenate((b1,b2,b3),axis =1)
                Bbar =np.reshape(np.mean(B, axis = 1), (d,1))
                Btilda = B-Bbar
                Utilda, _, _ = np.linalg.svd(Btilda, full_matrices=True)
                PCscores = (Utilda.T)@Btilda
                PCscores = (PCscores[0:k,:]).T
                
                ######################################### Test data##################

                n1 = 9
                n2 = 5
                n3 = 2
                nd = n1+n2+n3 #Total test
                ### user side
                #####user1
                b1=weight_matrix(0,n1,usertst,u,d,n)
              
                #######user2
                b2=weight_matrix(n1,n1+n2,usertst,u,d,n)
             
                #############user3
                b3=weight_matrix(n1+n2,n1+n2+n3,usertst,u,d,n)
              
                b1= b1-Bbar
                b2= b2-Bbar
                b3= b3-Bbar
    
                PCscorestst1 = (Utilda.T)@b1
                PCscorestst2 = (Utilda.T)@b2
                PCscorestst3 = (Utilda.T)@b3
                ### server side for our compariosn purpose
                PCscorestst = np.concatenate((PCscorestst1,PCscorestst2,PCscorestst3), axis=1)
                PCscorestst= (PCscorestst[0:k,:]).T
                ##################Modeling: 
                model = LM.fit(PCscores, np.log(tray[0:71]))
                predictions=model.predict(PCscorestst)
                abs_errors = np.mean(np.abs(np.exp(predictions) - tsty) / np.abs(tsty))
                Matrixerror[(k, rank)] = Matrixerror.get((k, rank), 0) + abs_errors

                
            
    ################### Final  Training Data#################################    
      
    optimalrow, d = min(Matrixerror,key=Matrixerror.get)
    n1 = 54
    n2 = 27
    n3 = 9
    
    H = np.random.randn(n, d)
    uh, sh, vh = np.linalg.svd(H, full_matrices=False)
    u = uh @ vh
    
    resnorm_prev =1000 # a big number
    diff = 1
    for _ in range(100): # 10 cycles
        resnorm =0
        u = basis_construction(0,n1,x.T,u,d,n)
        resnorm = resnorm_clac(0,n1,x.T,u,d,n,resnorm)

        u = basis_construction(n1,n1+n2,x,u,d,n)
        resnorm = resnorm_clac(n1,n1+n2,x.T,u,d,n,resnorm)

        u = basis_construction(n1+n2,n1+n2+n3,x.T,u,d,n)
        resnorm = resnorm_clac(n1+n2,n1+n2+n3,x.T,u,d,n,resnorm)

        diff =resnorm_prev -resnorm
        resnorm_prev = resnorm
        
    ###########user1:
    b1=weight_matrix(0,n1,x.T,u,d,n)
    ###########user2:
    b2=weight_matrix(n1,n1+n2,x.T,u,d,n)
    ###########user3:
    b3=weight_matrix(n1+n2,n1+n2+n3,x.T,u,d,n)
    ##### Server side
    B = np.concatenate((b1,b2,b3),axis =1)
    Bbar =np.reshape(np.mean(B, axis = 1), (d,1))
    Btilda = B-Bbar
    Utilda, _, _ = np.linalg.svd(Btilda, full_matrices=True)
    PCscores = (Utilda.T)@Btilda
    PCscores = (PCscores[0:optimalrow,:]).T

    n1 = 30
    n2 = 30
    n3 = 30
    nd=n1+n2+n3
    ###user side
    b1=weight_matrix(0,n1,xtest,u,d,n)
    ###########user2:
    b2=weight_matrix(0,n2,xtest,u,d,n)
    ###########user3:
    b3=weight_matrix(0,n3,xtest,u,d,n)

    b1= b1-Bbar
    b2= b2-Bbar
    b3= b3-Bbar

    PCscorestst1 = (Utilda.T)@b1
    PCscorestst2 = (Utilda.T)@b2
    PCscorestst3 = (Utilda.T)@b3
    #### server side
    PCscorestst = np.concatenate((PCscorestst1,PCscorestst2,PCscorestst3), axis=1)
    PCscorestst= (PCscorestst[0:optimalrow,:]).T

    ##################Modeling: 
    model = LM.fit(PCscores, np.log(y))
    prediction2 = model.predict(PCscorestst)
    for i in range(0,nd):
        prediction2[i]=abs(np.exp(prediction2[i])-ytest[i%30])/abs(ytest[i%30])       
   
end = time.time()
total_time = end-start

print(total_time)
