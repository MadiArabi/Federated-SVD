import numpy as np
from sklearn.model_selection import KFold # import KFold
from sklearn.linear_model import LinearRegression
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt
import math
import time
######### data preperation:
start = time.time()
np.random.seed(5022)
n_users =150 # number of users
user_numbers=[np.random.choice(20,1)[0]+1 for i in range(n_users)] #observatons per user
m = sum(user_numbers)+30 # number of total observations from all users+ test data
########################### Data generation


LM = LinearRegression()

@jit(nopython=True)
def basis_construction(user,u,d,n):
    d=np.int64(d)
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
        #print('one')
        p=u@w
        vtilda = v
        vtilda[vtilda == 0] = p[vtilda == 0,0]
        r=vtilda-p[:,0]
        mat=np.concatenate((np.eye(d,d), w), axis =1)
        norm = np.linalg.norm(r)
        zero = np.concatenate((np.zeros((1,d)),np.array([[norm]])),axis=1)
        #print('two')
        mat=np.concatenate((mat, zero), axis =0)
        [Utilda, _, _]=np.linalg.svd(mat)
        norm = r/np.linalg.norm(r)
        u = np.column_stack((u,norm))
        u= u@Utilda
        #print('last')
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
    d=np.int64(d)
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
    
    
X , Y =generate_data(m=m,missing_level=0.3)
D=[5,10,15,20]
xtest = X[m-30:m,:].T
ytest = Y[m-30:m]
X=X[0:m-30,:]
Y = Y[0:m-30]
#shuffle
times = np.zeros(n_users)
arr = np.arange(m-30)
for USER in range(n_users):
    print('User',USER)
    if user_numbers[USER]>1:
        all_errors=0
        for randomness in range(10):
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
            if user_numbers[USER]>=5:
                Matrixerror = {}
                for rank in D:
                    rank=int(rank)
                    if rank==5:
                        K=[2,4,5]
                    else:
                        K=np.linspace(rank/5,rank,5)
                    for k in K:
                        k=int(k)
                        kf = KFold(n_splits=5) 
                        
                        train=[]
                        test=[]
                        for train_index, test_index in kf.split(users[USER]):
                           train.append(train_index)
                           test.append(test_index)
                        
                        for index in range(0,5):
                
                            All_usersx = users[USER][train[index]]
                            All_usersy = users_y[USER][train[index]]
                            All_usersy_test = users_y[USER][test[index]]
                            
                            n = 150 # 
                            d=rank # number of features in time intervals
                            
                            H = np.random.randn(n, d)
                            uh, _, vh = np.linalg.svd(H, full_matrices=False)
                            u = uh @ vh
                           
                            
                            for i in range(20): # 10 cycles
                                u = basis_construction(np.array(All_usersx).T,u,d,n)
                                    
                            b =weight_matrix(np.array(All_usersx).T,u,d,n)
                    
                            
                            #B = np.concatenate((b1,b2,b3),axis =1)
                            B=np.array(b)
                            Bbar =np.reshape(np.mean(B, axis = 1), (d,1))
                            Btilda = B-Bbar
                            Utilda, _, _ = np.linalg.svd(Btilda, full_matrices=True)
                            PCscores = (Utilda.T)@Btilda
                            PCscores = (PCscores[0:k,:]).T
                            
                            ######################################### Test data##################
              
                            
                            b_test = weight_matrix(np.array(users[USER][test[index]]).T,u,d,n)
                   
                            b_test=np.array(b_test)
                            b_test = b_test-Bbar
                            PCscorestst = (Utilda.T)@b_test
                            PCscorestst= (PCscorestst[0:k,:]).T
                            ##################Modeling: 
                            model = LM.fit(PCscores, np.log(All_usersy))
                            predictions=model.predict(PCscorestst)
                            abs_errors = np.abs(np.exp(predictions) - All_usersy_test) / np.abs(All_usersy_test)
                            nd = len(All_usersy_test)
                            if index == 0:
                                all_errors = abs_errors.reshape(nd, 1)
                            else:
                                while all_errors.shape[0]!=len(abs_errors):
                                    if len(abs_errors)<all_errors.shape[0]:
                                        abs_errors = np.append(abs_errors, 0)
                                    else:
                                        abs_errors= abs_errors[:-1]
                                nd = len(abs_errors)
                                all_errors = np.concatenate((all_errors, abs_errors.reshape(nd, 1)), axis=1)
                
                        ErrorSum=all_errors.sum().sum()
                        Matrixerror.update({(k,rank): ErrorSum})
                    
            ################### Final  Training Data#################################    
            if user_numbers[USER]>=5:
                argmax = min(Matrixerror,key=Matrixerror.get)
                optimalrow= argmax[0]
                d=argmax[1] # number of features in time intervals
                
            elif 2<=user_numbers[USER]<5:
                optimalrow= user_numbers[USER]
                d=user_numbers[USER] # number of features in time intervals
            
            All_users = users[USER]
            All_usersy = users_y[USER]
            
            H = np.random.randn(n, d)
            uh, sh, vh = np.linalg.svd(H, full_matrices=False)
            u = uh @ vh
            
    
            diff = 1
            #print(diff)
            for _ in range(100): # 100 cycles
                u = basis_construction(np.array(All_users).T,u,d,n)
                    
            b= output =weight_matrix(np.array( All_users).T,u,d,n)
    
            B=np.array(b)
            Bbar =np.reshape(np.mean(B, axis = 1), (d,1))
            Btilda = B-Bbar
            Utilda, _, _ = np.linalg.svd(Btilda, full_matrices=True)
            PCscores = (Utilda.T)@Btilda
            PCscores = (PCscores[0:optimalrow,:]).T
            
            ######################################### Test data################
            b_test= weight_matrix(xtest,u,d,n)
                
            b_test=np.array(b_test)
            b_test = b_test-Bbar
            PCscorestst = (Utilda.T)@b_test
            PCscorestst= (PCscorestst[0:optimalrow,:]).T
            
          
            ##################Modeling: 
            model = LM.fit(PCscores, np.log(All_usersy))
            prediction2 = model.predict(PCscorestst)
            nd=30
            for i in range(0,nd):
                prediction2[i]=abs(np.exp(prediction2[i])-ytest[i%30])/abs(ytest[i%30])       
            
            
            abs_errors = pd.DataFrame(prediction2)
            if 2<=user_numbers[USER]<5:
                group =1
            elif 5<=user_numbers[USER]<15:
                group=2
            elif 15<=user_numbers[USER]<=20:
                group=3
        
            abs_errors.to_csv(path +str(group)+'.csv', encoding='utf-8', index=False, mode='a')
                )
        
    end = time.time()
    total_time = (end-start)/60
    times[USER]=total_time

    print(total_time)
    
