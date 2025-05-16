from concurrent.futures import process
import numpy as np
import numpy.linalg as LA
import multiprocessing
import time
import warnings
import os

warnings.simplefilter('error')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

###############################################################
# Lorenz-96
###############################################################
N = 40   # 変数の個数
F = 8.0  # パラメータ
DT = 0.05 # 時間刻み幅

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
    k1 = DT * dX_dt_matrix(X, m)
    k2 = DT * dX_dt_matrix(X + k1/2, m)
    k3 = DT * dX_dt_matrix(X + k2/2, m)
    k4 = DT * dX_dt_matrix(X + k3, m)
    X = X + (k1 + 2*k2 + 2*k3 + k4)/6
    return X



#############################################################
# フィルタ
#############################################################
### 本体 ###
def ETKFCC(X_forcast, m, H, y, R, inf, add):
    # 偏差の生成
    xm_forcast  = (np.mean(X_forcast, axis = 1)).reshape(N,1)
    dX_f = ( X_forcast - np.tile(xm_forcast, (1,m)) )
    dY_d = np.dot(H,dX_f) * (1.0-add)

    # 各変数の解析
    R_inv = np.diag(np.reciprocal(R))
    UDU = ((m-1)/inf)*np.eye(m) + (np.dot(dY_d.T, np.dot(R_inv,dY_d))) # Multiplicative inflation
    D, U = LA.eigh(UDU)
    D_inv = np.reciprocal(D)
    sqrtD_inv = np.diag(np.sqrt(D_inv))
    T = np.sqrt(m-1)*(np.dot(U,np.dot(sqrtD_inv,U.T)))
    dX_a = dX_f @ T
    update = np.dot(dX_f,(np.dot((np.dot(T,T.T)/(m-1)),np.dot(dY_d.T,np.dot(R_inv,np.tile(y.reshape(H.shape[0],1) - np.dot(H,xm_forcast), (1,m))))) + T))
    X_analise = np.tile(xm_forcast, (1,m)) + update

    B_mat = (dX_f @ dX_f.T) / (m-1)
    B = np.trace(B_mat)/N
    A_mat = (dX_a @ dX_a.T) / (m-1)
    A = np.trace(A_mat)/N
    ## Innovation Statistics ##
    # 基本の差分
    d_ob = (y.reshape(40,1) - np.dot(H,xm_forcast)).reshape(40)
    d_oa = (y.reshape(40,1) - np.dot(H,np.mean(X_analise, axis = 1).reshape(N,1))).reshape(40)
    d_ab = (np.dot(H,np.mean(X_analise, axis = 1).reshape(40,1)) - np.dot(H,xm_forcast)).reshape(40)

    # 各種統計量
    ob_ob = np.dot(d_ob,d_ob)
    oa_oa = np.dot(d_oa,d_oa)
    ab_ab = np.dot(d_ab,d_ab)
    ab_ob = np.dot(d_ab,d_ob)
    oa_ob = np.dot(d_oa,d_ob)
    ab_oa = np.dot(d_ab,d_oa)
    HBH = np.trace(H @ B_mat @H.T)
    HAH = np.trace(H @ A_mat @H.T)

    return X_analise,B,A, ob_ob,oa_oa,ab_ab,ab_ob,oa_ob,ab_oa,HBH,HAH


############################################################
# 観測の定義
############################################################
def make_all(num=0):
    """
    全点観測
    """
    H = np.eye(N)
    R = np.ones(N)
    return H, R

def make_add_error(X_forcast, eta, x_true, add, k, H):
    """
    X_forcast : アンサンブル
    m : アンサンブル数
    x_true : 真値
    add : addパラメータ
    i : 時刻(乱数参照用)
    """
    # 誤差の生成
    error_f_orig = np.mean(X_forcast, axis=1) - x_true
    error_o_add = (H@(add*error_f_orig)).reshape(H.shape[0]) + (H@eta[k,:].T).reshape(H.shape[0])
    return error_o_add


###############################################################
# 実行部分　
###############################################################
def run(list):
    x_true, X_start, m, inf, add_true, add_sys, r, eta, file_name, ETKF_inf,ETKF_R = list
    # 初期値
    X_analise = X_start
    H, R = make_all()
    R = r*R
    k = 0
    #file = open(file_name+"/time_series_ETKFinf={:.2f}_ETKFR={:.1f}_add={:.1f}_inf={:.2f}.csv".format(ETKF_inf,ETKF_R,add_true,inf), 'w')
    #file.writelines("time,forecast_RMSE,analysis_RMSE,observation_RMSE,correlation\n")
    try:
        # Spin up
        for i in range(1460):
            # 予測
            X_forcast = step_matrix(X_analise, m)
            f_error = LA.norm(np.mean(X_forcast, axis=1) - x_true[k,:]) / np.sqrt(N)
            error = make_add_error(X_forcast, eta, x_true[k,:], add_true, k, H) # 相関のある誤差
            o_error = LA.norm(error) / np.sqrt(N)
            cov = np.dot(error,np.mean(X_forcast, axis=1) - x_true[k,:]) / N
            correlation = cov / (o_error*f_error)
            y = H@(x_true[k,:].T) + error

            # 更新
            X_analise,B_tmp,A_tmp, ob_ob_tmp,oa_oa_tmp,ab_ab_tmp,ab_ob_tmp,oa_ob_tmp,ab_oa_tmp,HBH_tmp,HAH_tmp \
                = ETKFCC(X_forcast, m, H, y, R, inf, add_sys)

            # 出力
            #a_error = LA.norm(np.mean(X_analise, axis = 1)-x_true[k,:]) / np.sqrt(N)
            #np.savetxt(file, [[k,f_error,a_error,o_error,correlation]], delimiter=',')

            k += 1

        # 推定結果の出力の準備
        analysis_error = np.array([])
        observer_error = np.array([])
        forcast_error = np.array([])
        B = np.array([])
        A = np.array([])
        ob_ob = np.array([])
        oa_oa = np.array([])
        ab_ab = np.array([])
        ab_ob = np.array([])
        oa_ob = np.array([])
        ab_oa = np.array([])
        HBH = np.array([])
        HAH = np.array([])

        # 実行部分
        for i in range(1460*10):
            # 予測
            X_forcast = step_matrix(X_analise, m)
            f_error = LA.norm(np.mean(X_forcast, axis=1) - x_true[k,:]) / np.sqrt(N)
            forcast_error = np.hstack([forcast_error, np.mean(X_forcast, axis=1) - x_true[k,:]])
            error = make_add_error(X_forcast, eta, x_true[k,:], add_true, k, H) # 相関のある誤差
            o_error = LA.norm(error) / np.sqrt(N)
            cov = np.dot(error,np.mean(X_forcast, axis=1) - x_true[k,:]) / N
            correlation = cov / (o_error*f_error)
            y = H@(x_true[k,:].T)+ error

            # 更新
            X_analise,B_tmp,A_tmp, ob_ob_tmp,oa_oa_tmp,ab_ab_tmp,ab_ob_tmp,oa_ob_tmp,ab_oa_tmp,HBH_tmp,HAH_tmp \
                = ETKFCC(X_forcast, m, H, y, R, inf, add_sys)

            # 出力
            a_error = LA.norm(np.mean(X_analise, axis = 1)-x_true[k,:]) / np.sqrt(N)
            #np.savetxt(file, [[k,f_error,a_error,o_error,correlation]], delimiter=',')
            x_mean = np.mean(X_analise, axis = 1)
            analysis_error = np.hstack([analysis_error,x_mean-x_true[k,:]])
            observer_error = np.hstack([observer_error,(y-H@x_true[k,:]).reshape(H.shape[0])])
            B = np.hstack([B,B_tmp])
            A = np.hstack([A,A_tmp])
            ob_ob = np.hstack([ob_ob,ob_ob_tmp])
            oa_oa = np.hstack([oa_oa,oa_oa_tmp])
            ab_ab = np.hstack([ab_ab,ab_ab_tmp])
            ab_ob = np.hstack([ab_ob,ab_ob_tmp])
            oa_ob = np.hstack([oa_ob,oa_ob_tmp])
            ab_oa = np.hstack([ab_oa,ab_oa_tmp])
            HBH = np.hstack([HBH,HBH_tmp])
            HAH = np.hstack([HAH,HAH_tmp])

            k += 1
    except:
        a_error = 0
        o_error = 0
        f_error = 0
        correlation = 0
        B = 0
        A = 0
        ob_ob = 0
        oa_oa = 0
        ab_ab = 0
        ab_ob = 0
        oa_ob = 0
        ab_oa = 0
        HBH = 0
        HAH = 0
        print("Error :: ETKF inf={:.2f} R={:.1f} : add_true={:.1f} add_sys={:.2f} Ruc={:.1f} inf={:.2f}".format(ETKF_inf,ETKF_R,add_true,add_sys,r,inf))
    else:
        a_error = LA.norm(analysis_error) / np.sqrt(analysis_error.shape[0])
        o_error = LA.norm(observer_error) / np.sqrt(observer_error.shape[0])
        f_error = LA.norm(forcast_error) / np.sqrt(forcast_error.shape[0])
        cov = np.dot(observer_error,forcast_error) / (forcast_error.shape[0])
        correlation = cov / (o_error*f_error)
        B = np.sqrt(np.mean(B))
        A = np.sqrt(np.mean(A))
        ob_ob = np.mean(ob_ob)
        oa_oa = np.mean(oa_oa)
        ab_ab = np.mean(ab_ab)
        ab_ob = np.mean(ab_ob)
        oa_ob = np.mean(oa_ob)
        ab_oa = np.mean(ab_oa)
        HBH = np.mean(HBH)
        HAH = np.mean(HAH)
        print("Clear :: ETKF inf={:.2f} R={:.1f} : add_true={:.1f} add_sys={:.2f} Ruc={:.1f} inf={:.2f}".format(ETKF_inf,ETKF_R,add_true,add_sys,r,inf))
    #file.close()
    return add_true, add_sys, inf, r, a_error, o_error, f_error, correlation, B,A, ob_ob,oa_oa,ab_ab,ab_ob,oa_ob,ab_oa,HBH,HAH, ETKF_inf,ETKF_R

###################################################################################
# 出力部分
###################################################################################
### 出力 ###
def output_all(m):
    x_true = np.loadtxt("./true_data.csv", delimiter=',')
    X_start = np.loadtxt("./start_ensemble.csv", delimiter=',')
    eta = np.loadtxt("./eta.csv", delimiter=',')

    ### Estimated ETKFCC with inflation ###
    print("All estimated ETKFCC with inflation")

    # parameter estimation
    data = np.loadtxt("./stability_estimated_parameters.csv", delimiter=',', skiprows=1)
    add_true = data[:,0]
    add_size = add_true.shape[0]
    ETKF_inf = data[:,1]
    ETKF_R = data[:,2]
    add_est1 = data[:,3]
    Ruc_est1 = data[:,4]
    add_est2 = data[:,5]
    Ruc_est2 = data[:,6]
    Ruc_est3 = data[:,7]
    add_ests = [add_est1,add_est2,add_est2]
    Ruc_ests = [Ruc_est1,Ruc_est2,Ruc_est3]
    file_name = ["./stability_abob_obob_ETKFCC_with_inf","./stability_aboa_obob_ETKFCC_with_inf","./stability_aboa_oaob_ETKFCC_with_inf"]

    set_inf = np.array([1.0 + 0.01*i for i in range(21)])
    set_inf = np.hstack((set_inf, [1.3 + 0.1*i for i in range(7)]))
    set_inf = np.hstack((set_inf, [2.0 + 1.0*i for i in range(4)]))
    inf_size = set_inf.shape[0]

    for k in range(3):
        start = time.time()
        add_est = add_ests[k]
        Ruc_est = Ruc_ests[k]
        list = []
        for i in range(add_size):
            for j in range(inf_size):
                tmp_list = [x_true, X_start, m, set_inf[j], add_true[i],add_est[i], Ruc_est[i], eta, file_name[k], ETKF_inf[i],ETKF_R[i]]
                list.append(tmp_list)

        os.makedirs(file_name[k], exist_ok=True)
        with multiprocessing.Pool(processes=31) as pool:
            result = pool.map(run,list)
        stop = time.time()
        print("{:.1f}h".format((stop-start)/3600.0))

        # CSV出力
        file = open(file_name[k]+".csv", 'w')
        file.writelines("#ens={:d} error=0\n".format(m))
        file.writelines("add_true,add_sys,multi_inf,Ruc_sys,analysis_error,observation_error,forcast_error,correlation,spread_B,spread_A,ob_ob,oa_oa,ab_ab,ab_ob,oa_ob,ab_oa,HBH,HAH,ETKF_inf,ETKF_R\n")
        np.savetxt(file, result, delimiter=',')
        file.close()

    print("End program")
    
### 本体 ###
if __name__ == "__main__":
    m = 40 # 走らせるアンサンブルの数

    output_all(m)
