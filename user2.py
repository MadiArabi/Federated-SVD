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
print(start)
np.random.seed(5022)
m= 120 # number of total observations
p=150
y = []
x = []
D= 2
sigma = 20


########################### Data generation


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

xx = np.linspace(0,150,150)
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
missingindices = np.random.choice(150*120, size=int(120*150*0.7), replace=False) # missing level
X=X.ravel()
X[missingindices] = 0
X= np.reshape(X,(120,150))
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
    
    

D=[5,10,20]
xtest = X[90:120,:].T
ytest = Y[90:120]

X=X[0:90,:]
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
    
    
    x= np.concatenate((user1,user2,user3), axis=0).T
    y=np.concatenate((Y1,Y2,Y3))

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
            
            train1=[]
            test1=[]
            train2=[]
            test2=[]
            train3=[]
            test3=[]
           
            for train_index, test_index in kf.split(user1):
               train1.append(train_index)
               test1.append(test_index)
            for train_index, test_index in kf.split(user2):
               train2.append(train_index)
               test2.append(test_index)
            for train_index, test_index in kf.split(user3):
               train3.append(train_index)
               test3.append(test_index)
               
            for index in range(0,5):
                tray=np.concatenate((Y1[train1[index].astype(int)],Y2[train2[index].astype(int)],Y3[train3[index].astype(int)]),axis=0)
                tsty=np.concatenate((Y1[(test1[index]).astype(int)],Y2[(test2[index]).astype(int)],Y3[(test3[index]).astype(int)]),axis=0)
                user=np.concatenate((user1[(train1[index]).astype(int),:],user2[(train2[index]).astype(int),:],user3[(train3[index]).astype(int),:]),axis=0).T
                usertst=np.concatenate((user1[(test1[index]).astype(int),:],user2[(test2[index]).astype(int),:],user3[(test3[index]).astype(int),:]),axis=0).T
                n1 = 43
                n2 = 21
                n3 = 7
                n = 150 # 
                d=rank # number of features in time intervals
                
                H = np.random.randn(n, d)
                uh, _, vh = np.linalg.svd(H, full_matrices=False)
                u = uh @ vh
               
                
                for i in range(20): # 10 cycles
                    u = basis_construction(n1,n1+n2,user,u,d,n)
                    
                 
                    
                    
                ###########user1:
                b1=weight_matrix(n1,n1+n2,user,u,d,n)
                
              
                
             
                Bbar =np.reshape(np.mean(b1, axis = 1), (d,1))
                Btilda = b1-Bbar
                Utilda, _, _ = np.linalg.svd(Btilda, full_matrices=True)
                PCscores = (Utilda.T)@Btilda
                PCscores = (PCscores[0:k,:]).T
                
                ######################################### Test data##################

                n1 = 9
                n2 = 5
                n3 = 2
                nd = 5 #Total test
                
                #####user1
                b1=weight_matrix(n1,n1+n2,usertst,u,d,n)
              
               
              
                b1= b1-Bbar
               
    
                PCscorestst = (Utilda.T)@b1
                
                PCscorestst= (PCscorestst[0:k,:]).T
                ##################Modeling: 
                model = LM.fit(PCscores, np.log(tray[43:64]))
                predictions=model.predict(PCscorestst)
                abs_errors = np.abs(np.exp(predictions) - tsty[n1:n1+nd]) / np.abs(tsty[n1:n1+nd])
    
                if index == 0:
                    all_errors = abs_errors.reshape(nd, 1)
                else:
                    all_errors = np.concatenate((all_errors, abs_errors.reshape(nd, 1)), axis=1)

            ErrorSum=all_errors.sum().sum()
            Matrixerror.update({(k,rank): ErrorSum})
            
    ################### Final  Training Data#################################    
      
    argmax = min(Matrixerror,key=Matrixerror.get)
    n1 = 54
    n2 = 27
    n3 = 9
    optimalrow= argmax[0]
    d=argmax[1] # number of features in time intervals
    
    
    H = np.random.randn(n, d)
    uh, sh, vh = np.linalg.svd(H, full_matrices=False)
    u = uh @ vh
    
    resnorm1 =1000
    diff = 1
    for _ in range(100): # 10 cycles
        resnorm =0
        u = basis_construction(n1,n1+n2,x,u,d,n)
        resnorm = resnorm_clac(n1,n1+n2,x,u,d,n,resnorm)
        diff =resnorm1 -resnorm
        resnorm1 = resnorm
        #print(resnorm1)
        
    ###########user1:
    b1=weight_matrix(n1,n1+n2,x,u,d,n)
   
    
    Bbar =np.reshape(np.mean(b1, axis = 1), (d,1))
    Btilda = b1-Bbar
    Utilda, _, _ = np.linalg.svd(Btilda, full_matrices=True)
    PCscores = (Utilda.T)@Btilda
    PCscores = (PCscores[0:optimalrow,:]).T
    
   
    
  
        ######################################### Final Test sata#############
    ###############Missing elements
    n1 = 30
    nd=30
    #d=300 # number of features in time intervals
    
    b1=weight_matrix(0,n1,xtest,u,d,n)
    b1= b1-Bbar
    

    PCscorestst = (Utilda.T)@b1
    PCscorestst= (PCscorestst[0:optimalrow,:]).T
    
    
  
    ##################Modeling: 
    model = LM.fit(PCscores, np.log(Y2))
    prediction2 = model.predict(PCscorestst)
    
    for i in range(0,nd):
        prediction2[i]=abs(np.exp(prediction2[i])-ytest[i%30])/abs(ytest[i%30])       
    
    
    abs_errors = pd.DataFrame(prediction2)
    abs_errors.to_csv(r'G:\My Drive\research1\simultaed\user270159.csv', encoding='utf-8', index=False, mode='a')
   
end = time.time()
total_time = end-start

print(total_time)