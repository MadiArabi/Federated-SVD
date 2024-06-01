import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import pandas as pd
from numba import jit
import time
from data_generation import generate_data

# Data preparation
start = time.time()
print(f"Start time: {start}")

np.random.seed(5022)
m = 120  # number of total observations
p = 150
D = [5, 10, 20, 30, 40]

X, Y = generate_data(m=m, p=p,missing_level=0.3)

LM = LinearRegression()

@jit(nopython=True)
def basis_construction(user, u, d):
    n_observations = user.shape[1]
    for col in range(n_observations):
        v = user[:, col]
        vOmega = v[v != 0]
        uOmega = u[v != 0, :]
        size1 = vOmega.size
        uOmega = np.reshape(uOmega, (size1, d))
        vOmega = np.reshape(vOmega, (size1, 1))
        w = np.linalg.inv(uOmega.T @ uOmega) @ (uOmega.T @ vOmega)
        p = u @ w
        vtilda = v.copy()
        vtilda[vtilda == 0] = p[vtilda == 0]
        r = vtilda - p[:, 0]
        mat = np.zeros((d + 1, d + 1))
        mat[:d, :d] = np.eye(d)
        mat[:d, d] = w.flatten()
        norm = np.linalg.norm(r)
        mat[d, d] = norm
        Utilda, _, _ = np.linalg.svd(mat)
        r_norm = r / norm
        u = np.column_stack((u, r_norm)).dot(Utilda)
        u = u[:, :d]
    return u

@jit(nopython=True)
def weight_matrix(user, u, d):
    n_observations = user.shape[1]
    W = np.zeros((d, n_observations))
    for col in range(n_observations):
        v = user[:, col]
        vOmega = v[v != 0]
        uOmega = u[v != 0, :]
        size1 = vOmega.size
        uOmega = np.reshape(uOmega, (size1, d))
        vOmega = np.reshape(vOmega, (size1, 1))
        w = np.linalg.inv(uOmega.T @ uOmega) @ (uOmega.T @ vOmega)
        W[:, col] = w.flatten()
    return W

# Shuffle and prepare data for cross-validation
xtest = X[90:120, :].T
ytest = Y[90:120]
X = X[0:90, :]
Y = Y[0:90]

arr = np.arange(90)
for randomness in range(15):
    np.random.shuffle(arr)
    x = X[arr, :]
    y = Y[arr]

    user1 = np.array(x[0:54])
    user2 = np.array(x[54:81])
    user3 = np.array(x[81:90])
    Y1 = np.array(y[0:54])
    Y2 = np.array(y[54:81])
    Y3 = np.array(y[81:90])

    x = np.concatenate((user1, user2, user3), axis=0).T
    y = np.concatenate((Y1, Y2, Y3))

    Matrixerror = {}

    for rank in D:
        rank = int(rank)
        K = [2, 4, 5] if rank == 5 else np.linspace(rank // 5, rank, 5, dtype=int)
        for k in K:
            k = int(k)
            kf = KFold(n_splits=5)

            train_indices, test_indices = [], []
            for train_index, test_index in kf.split(user1):
                train_indices.append(train_index)
                test_indices.append(test_index)
            for train_index, test_index in kf.split(user2):
                train_indices.append(train_index)
                test_indices.append(test_index)
            for train_index, test_index in kf.split(user3):
                train_indices.append(train_index)
                test_indices.append(test_index)

            for index in range(5):
                tray = np.concatenate((Y1[train_indices[index]], Y2[train_indices[index]], Y3[train_indices[index]]))
                tsty = np.concatenate((Y1[test_indices[index]], Y2[test_indices[index]], Y3[test_indices[index]]))
                user_train = np.concatenate((user1[train_indices[index]], user2[train_indices[index]], user3[train_indices[index]]), axis=0).T
                user_test = np.concatenate((user1[test_indices[index]], user2[test_indices[index]], user3[test_indices[index]]), axis=0).T

                n, d = 150, rank
                H = np.random.randn(n, d)
                u = np.linalg.svd(H, full_matrices=False)[0] @ np.linalg.svd(H, full_matrices=False)[2]

                for _ in range(20):
                    u = basis_construction(user_train, u, d)

                b_train = weight_matrix(user_train, u, d)
                Bbar = np.mean(b_train, axis=1, keepdims=True)
                Btilda = b_train - Bbar
                Utilda = np.linalg.svd(Btilda, full_matrices=True)[0]
                PCscores = (Utilda.T @ Btilda)[:k, :].T

                b_test = weight_matrix(user_test, u, d)
                b_test -= Bbar
                PCscorestst = (Utilda.T @ b_test)[:k, :].T

                model = LM.fit(PCscores, np.log(tray))
                predictions = model.predict(PCscorestst)
                abs_errors = np.abs(np.exp(predictions) - tsty) / np.abs(tsty)

                if index == 0:
                    all_errors = abs_errors.reshape(-1, 1)
                else:
                    all_errors = np.concatenate((all_errors, abs_errors.reshape(-1, 1)), axis=1)

            ErrorSum = all_errors.sum().sum()
            Matrixerror[(k, rank)] = ErrorSum

    argmax = min(Matrixerror, key=Matrixerror.get)
    optimalrow, d = argmax

    H = np.random.randn(n, d)
    u = np.linalg.svd(H, full_matrices=False)[0] @ np.linalg.svd(H, full_matrices=False)[2]

    for _ in range(100):
        u = basis_construction(x, u, d)

    b_final = weight_matrix(x, u, d)
    Bbar = np.mean(b_final, axis=1, keepdims=True)
    Btilda = b_final - Bbar
    Utilda = np.linalg.svd(Btilda, full_matrices=True)[0]
    PCscores = (Utilda.T @ Btilda)[:optimalrow, :].T

    b_test_final = weight_matrix(xtest, u, d)
    b_test_final -= Bbar
    PCscorestst_final = (Utilda.T @ b_test_final)[:optimalrow, :].T

    model = LM.fit(PCscores, np.log(Y))
    prediction2 = model.predict(PCscorestst_final)
    errors = np.abs(np.exp(prediction2) - ytest) / np.abs(ytest)

end = time.time()
total_time = end - start

print(f"Total time: {total_time} seconds")
