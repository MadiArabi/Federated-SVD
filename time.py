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
m= 120 # number of total observations
p=150
y = []
x = []
D= 2
sigma = 20
n_users =150 # number of users
user_numbers=[np.random.choice(20,1)[0]+1 for i in range(n_users)] #observatons per user
m = sum(user_numbers)+30 # number of total observations from all users+ test data
########################### Data generation
group1=0
group2 =0
group3 =0

for i in user_numbers:
    if 2<=i<5:
        group1 +=1
    elif 5<=i<15:
        group2+=1
    elif 15<=i<=20:
        group3+=1
        
        

func = lambda c,t: -c/(math.log(t))
inter = []  
for i in range(m):
    x.append([])
    c= np.random.normal(1,0.25)
    epsilon= np.random.normal(0,0.025)
    #y.append(math.exp((-c/D)+epsilon))
    y.append(math.exp((-c/D)))
    for j in range(1,p+1):
        interval = j*0.006
        if interval <y[i]:
            x[i].append(func(c,interval))
        else:
             x[i].append(0)
summ = 0             
for i in x:
    for j in i:
        if j!=0:
            summ = summ+ j
          
sigmaP = math.sqrt(summ)/sigma*500
X=np.array(x)
xx = np.linspace(0,150,150)
for i in X:
    
    plt.plot(xx,i)
for i in x:
    for j in range(len(i)):
        if i[j]:
            i[j] = i[j]+ np.random.normal(0,0.2)
X=np.array(x)
Y=np.array(y)

for i in X:
    
    plt.plot(xx,i)
# adding missing values
'''
missing1 = 0
for i in X:
    for j in i:
        if j==0:
            missing1 +=1
            
All = int((150*100 -missing1)*0.3+ missing1)
missung_want = All/(100*150)   
'''
missingindices = np.random.choice(150*m, size=int(m*150*0.3), replace=False) # missing level
X=X.ravel()
X[missingindices] = 0
X= np.reshape(X,(m,150))
'''
missing1 = 0
for i in X:
    for j in i:
        if j==0:
            missing1 +=1

missing_is = missing1/(100*150)
'''


LM = LinearRegression()

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
    
    

D=[5,10,20,30,40,50,60]
#D=[5]
xtest = X[m-30:m,:].T
ytest = Y[m-30:m]
fold_users =[index for index, value in enumerate(user_numbers) if value >= 5]
timer_cv= np.zeros(len(fold_users))
timer = np.zeros(n_users)
X=X[0:m-30,:]
Y = Y[0:m-30]
#shuffle
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
        if rank==5:
            K=[2,4,5]
        else:
            K=np.linspace(rank/5,rank,5)
        for k in K:
            k=int(k)
            kf = KFold(n_splits=5) 
            train=[]
            test=[]
            
            
            for i in fold_users:
                train.append([])
                test.append([])
                for train_index, test_index in kf.split(users[i]):
                   train[-1].append(train_index)
                   test[-1].append(test_index)
            
            for index in range(0,5):
                
                All_usersx = []
                All_usersx_test=[]
                All_usersy = []
                All_usersy_test = []
                for j in range(len(fold_users)):
                    All_usersx.extend(users[fold_users[j]][train[j][index]])
                    All_usersx_test.extend(users[fold_users[j]][test[j][index]])
                    All_usersy.extend(users_y[fold_users[j]][train[j][index]])
                    All_usersy_test.extend(users_y[fold_users[j]][test[j][index]])
                
                n = 150 # 
                d=rank # number of features in time intervals
                
                H = np.random.randn(n, d)
                uh, _, vh = np.linalg.svd(H, full_matrices=False)
                u = uh @ vh
               
                
                for i in range(20): # 10 cycles
                    for j in range(len(fold_users)):
                        s_t = time.time()
                        u = basis_construction(users[fold_users[j]][train[j][index]].T,u,d,n)
                        s_e = time.time()
                        timer_cv[j]+=(s_e-s_t)
                    #u = basis_construction(np.array(All_usersx).T,u,d,n)
                        
                        
                 
                b = []
                for j in range(len(fold_users)):
                    s_t = time.time()
                    output =weight_matrix(users[fold_users[j]][train[j][index]].T,u,d,n)
                    b.extend(output.T)
                    s_e = time.time()
                    timer_cv[j]+=(s_e-s_t)
                
                
                #B = np.concatenate((b1,b2,b3),axis =1)
                B=np.array(b).T
                Bbar =np.reshape(np.mean(B, axis = 1), (d,1))
                Btilda = B-Bbar
                Utilda, _, _ = np.linalg.svd(Btilda, full_matrices=True)
                PCscores = (Utilda.T)@Btilda
                PCscores = (PCscores[0:k,:]).T
                
                ######################################### Test data##################
                b_test = []
                for j in range(len(fold_users)):
                    output = weight_matrix(users[fold_users[j]][test[j][index]].T,u,d,n)
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
                    #print(all_errors.shape[0])
                    #print(len(abs_errors))
                    
                    while all_errors.shape[0]!=len(abs_errors):
                        
                        if len(abs_errors)<all_errors.shape[0]:
                            abs_errors = np.append(abs_errors, 0)
                        else:
                            abs_errors= abs_errors[:-1]
                    nd = len(abs_errors)
                    all_errors = np.concatenate((all_errors, abs_errors.reshape(nd, 1)), axis=1)
                print('end',k,rank,index)
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
        for j in range(n_users):
            s_t = time.time()
            u = basis_construction(users[j].T,u,d,n)
            s_e = time.time()
            timer[j]+=(s_e-s_t)
            
            
     
    b = []
    for j in range(n_users):
        s_t = time.time()
        output =weight_matrix(users[j].T,u,d,n)
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
    for i in range(n_users):
        y_all.extend(users_y[i])
    model = LM.fit(PCscores, np.log(y_all))
    prediction2 = model.predict(PCscorestst)
    nd=30
    for i in range(0,nd):
        prediction2[i]=abs(np.exp(prediction2[i])-ytest[i%30])/abs(ytest[i%30])       
    
    
abs_errors = pd.DataFrame(prediction2)
#abs_errors.to_csv(r'G:\My Drive\research1\simultaed\FD_sim13010.csv', encoding='utf-8', index=False, mode='a')
    #abs_errors.to_csv(r'G:\My Drive\research1\simultaed\Federated70159users.csv', encoding='utf-8', index=False, mode='a')
timer = pd.DataFrame(timer)
timer.to_csv(r'G:\My Drive\research1\simultaed\FD_timer150.csv', encoding='utf-8', index=False, mode='a')
timer_cv = pd.DataFrame(timer_cv)
timer_cv.to_csv(r'G:\My Drive\research1\simultaed\FD_timer_cv150.csv', encoding='utf-8', index=False, mode='a')
end = time.time()
total_time = (end-start)/60

print(total_time)
