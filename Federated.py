import numpy as np
from sklearn.model_selection import KFold # import KFold
from sklearn.linear_model import LinearRegression
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt
import math
import time
import data_generation
np.random.seed(5022)

@jit(nopython=True)
def basis_construction(user,u,d,n):
    n_observations =user.shape[1]
    for col in range(n_observations): #user1 data completion
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
        vtilda = v
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
def weight_matrix(user,u,d,n):
    n_observations = user.shape[1]
    W=np.zeros((d,n_observations))
    for col in range(n_observations):
        v = user[:,col]
        vOmega =v[v!=0]
        uOmega =u[v!=0,:]
        size1=vOmega.size
        uOmega= np.reshape(uOmega, (size1,d))
        vOmega= np.reshape(vOmega, (size1,1))
        w= np.linalg.inv((uOmega.T)@uOmega)@(uOmega.T)@vOmega
        W[:,col] = w[:,0]

    return W

List =[10,20,50,100,150,300,500,800,1000]
for n_users in List:
    start_time = time.time()
    X, Y = data_generation.generate_data(missing_level=0.7)
    
    LM = LinearRegression()
    
    D=[5,10,20,30,40,50,60]
    xtest = X[-30:, :].T
    ytest = Y[-30:]
    fold_users = [index for index, value in enumerate(user_numbers) if value >= 5]
    timer_cv= np.zeros(len(fold_users))
    timer = np.zeros(n_users)
    X = X[:-30, :]
    Y = Y[:-30]
    arr = np.arange(m-30)
    for randomness in range(1):
        np.random.shuffle(arr)
        x = X[arr,:]
        y = Y[arr]
        users=[]
        users_y=[]
        start=0
        for i in range(n_users):
            users.append(np.array(x[start:start+user_numbers[i]]))
            users_y.append(np.array(y[start:start+user_numbers[i]]))
            start += user_numbers[i]
        Matrixerror = {}
        for rank in D:
            rank=int(rank)
            K=[2,4,5] if rank==5 else np.linspace(rank/5,rank,5, dtype=int)
            for k in K:
                k=int(k)
                kf = KFold(n_splits=5) 
                train=[]
                test=[]
                
                for i in fold_users:
                    kf_split = list(kf.split(users[i]))
                    train_indices.append([split[0] for split in kf_split])
                    test_indices.append([split[1] for split in kf_split])
                
                
                for index in range(0,5):
                    
                    All_usersx = []
                    All_usersx_test=[]
                    All_usersy = []
                    All_usersy_test = []
                    for j in range(len(fold_users)):
                        All_usersx.extend(users[fold_users[j]][train[j][index]])
                        All_usersx_test.extend(users[fold_users[j]][test[j][index]])
                    All_usersy.extend(users_y[0][train[0][index]])
                    All_usersy_test.extend(users_y[0][test[0][index]])
                
                    n = 150 # 
                    d=rank # number of features in time intervals
                    
                    H = np.random.randn(n, d)
                    uh, _, vh = np.linalg.svd(H, full_matrices=False)
                    u = uh @ vh
                   
                    for i in range(20): # 10 cycles
                        
                        s_t = time.time()
                        u = basis_construction(users[0][train[0][index]].T,u,d,n)
                        s_e = time.time()
                        timer_cv[0]+=(s_e-s_t)

                            
                            
                     
                    b = []
                    
                    s_t = time.time()
                    output =weight_matrix(users[0][train[0][index]].T,u,d,n)
                    b.extend(output.T)
                    s_e = time.time()
                    timer_cv[0]+=(s_e-s_t)
                
                    B=np.array(b).T
                    Bbar =np.reshape(np.mean(B, axis = 1), (d,1))
                    Btilda = B-Bbar
                    Utilda, _, _ = np.linalg.svd(Btilda, full_matrices=True)
                    PCscores = (Utilda.T)@Btilda
                    PCscores = (PCscores[0:k,:]).T
                    
                    ######################################### Test data##################
                    b_test = []
                   
                    output = weight_matrix(users[0][test[0][index]].T,u,d,n)
                    b_test.extend(output.T)
                        
                    b_test=np.array(b_test).T
                    b_test = b_test-Bbar
                    PCscorestst = (Utilda.T)@b_test
                    PCscorestst= (PCscorestst[0:k,:]).T
                    ##################Modeling: 
                    model = LM.fit(PCscores, np.log(All_usersy))
                    predictions=model.predict(PCscorestst)
                    abs_errors = np.abs(np.exp(predictions) - All_usersy_test) / np.abs(All_usersy_test)
                    nd = len(All_usersy_test)
                    #print('ml',k,rank,index)
                    if index == 0:
                        all_errors = abs_errors.reshape(nd, 1)
                    else:
                        while all_errors.shape[0]!=len(abs_errors)
                            if len(abs_errors)<all_errors.shape[0]:
                                abs_errors = np.append(abs_errors, 0)
                            else:
                                abs_errors= abs_errors[:-1]
                        nd = len(abs_errors)
                        all_errors = np.concatenate((all_errors, abs_errors.reshape(nd, 1)), axis=1)
                ErrorSum=all_errors.sum().sum()
                Matrixerror.update({(k,rank): ErrorSum})
                
        ################### Final  Training Data#################################    
          
        argmax = min(Matrixerror,key=Matrixerror.get)
        optimalrow= argmax[0]
        d=argmax[1] # number of features in time intervals
        
        
        H = np.random.randn(n, d)
        uh, sh, vh = np.linalg.svd(H, full_matrices=False)
        u = uh @ vh
        
        resnorm1 =1000
        diff = 1
        for _ in range(100): # 100 cycles
            
            s_t = time.time()
            u = basis_construction(users[0].T,u,d,n)
            s_e = time.time()
            timer[0]+=(s_e-s_t)
                
                
         
        b = []
        
        s_t = time.time()
        output =weight_matrix(users[0].T,u,d,n)
        b.extend(output.T)
        s_e = time.time()
        timer[j]+=(s_e-s_t)
        
        
        B=np.array(b).T
        Bbar =np.reshape(np.mean(B, axis = 1), (d,1))
        Btilda = B-Bbar
        Utilda, _, _ = np.linalg.svd(Btilda, full_matrices=True)
        PCscores = (Utilda.T)@Btilda
        PCscores = (PCscores[0:optimalrow,:]).T
        
        ######################################### Test data##################
        b_test = []
        for j in range(n_users):
            output = weight_matrix(xtest,u,d,n)
            b_test.extend(output.T)
            
        b_test=np.array(b_test).T
        b_test = b_test-Bbar
        PCscorestst = (Utilda.T)@b_test
        PCscorestst= (PCscorestst[0:optimalrow,:]).T
        
      
        ##################Modeling: 
        y_all=[]
        
        y_all.extend(users_y[0])
        model = LM.fit(PCscores, np.log(y_all))
        prediction2 = model.predict(PCscorestst)
        nd=30
        for i in range(0,nd):
            prediction2[i]=abs(np.exp(prediction2[i])-ytest[i%30])/abs(ytest[i%30])       
        
    abs_errors = pd.DataFrame(prediction2)
    end = time.time()
    total_time = (end-start_time)
    print(total_time)
    
