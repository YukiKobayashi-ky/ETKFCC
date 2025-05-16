from concurrent.futures import process
import numpy as np

###############################################################
# Lorenz-96
###############################################################
N = 40   # 変数の個数
F = 8.0  # パラメータ

def dX_dt(ary):
    xj_zero = ary.copy()
    xj_one = np.roll(ary, -1)
    xj_m_two = np.roll(ary, 2)
    xj_m_one = np.roll(ary, 1)
    dX_dt = (xj_one - xj_m_two)*xj_m_one - xj_zero + np.ones(N)*F
    return dX_dt

def dX_dt_matrix(ary, m):
    xj_zero = ary.copy()
    xj_one = np.roll(ary, -1, axis=0)
    xj_m_two = np.roll(ary, 2, axis=0)
    xj_m_one = np.roll(ary, 1, axis=0)
    dX_dt = (xj_one - xj_m_two)*xj_m_one - xj_zero + np.ones((N,m))*F
    return dX_dt

def step(x):
    for i in range(5):
        k1 = 0.01 * dX_dt(x)
        k2 = 0.01 * dX_dt(x + k1/2)
        k3 = 0.01 * dX_dt(x + k2/2)
        k4 = 0.01 * dX_dt(x + k3)
        x = x + (k1 + 2*k2 + 2*k3 + k4)/6
    return x

def step_matrix(X, m):
    k1 = 0.05 * dX_dt_matrix(X, m)
    k2 = 0.05 * dX_dt_matrix(X + k1/2, m)
    k3 = 0.05 * dX_dt_matrix(X + k2/2, m)
    k4 = 0.05 * dX_dt_matrix(X + k3, m)
    X = X + (k1 + 2*k2 + 2*k3 + k4)/6
    return X

###############################################################
# 実行部分
###############################################################
def make_true_data():
    x_true = np.random.normal(0.0, 1.0, N)

    for i in range(1460):
        x_true = step(x_true)
    
    data = np.empty((0,40))
    for i in range(1460*11):
        x_true = step(x_true)
        data = np.vstack([data,x_true])
    
    file = open("true_data.csv", 'w')
    np.savetxt(file, data, delimiter=',')
    file.close()

def make_start_ensemble(m):
    X_ens = np.random.normal(0.0, 1.0, N*m).reshape(N, m)

    for i in range(1460):
            X_ens = step_matrix(X_ens,m)
    
    file = open("start_ensemble.csv", 'w')
    np.savetxt(file, X_ens, delimiter=',')
    file.close()

def make_eta():
    eta = np.empty((1460*11,N))
    error_orig =  np.random.normal(0.0, 1.0, 1460*11*N)
    for i in range(N):
        eta[:,i] = error_orig[i*1460*11 : (i+1)*1460*11]

    file = open("eta.csv", 'w')
    np.savetxt(file, eta, delimiter=',')
    file.close()

### 本体 ###
if __name__ == "__main__":
    m = 40 # 走らせるアンサンブルの数

    #make_true_data()
    #make_start_ensemble(m)
    #make_eta()