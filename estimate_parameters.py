import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, BoundaryNorm, rgb2hex
import matplotlib.gridspec as gridspec
from matplotlib.legend_handler import HandlerTuple
import cmcrameri
import cmcrameri.cm as cmc
import pandas as pd


def optimal_parameters():
    df = pd.read_csv('ETKF_tuning.csv', comment='#')
    df["RMSE_ratio"] = df["analysis_error"] / df["observation_error"]
    df_calc = df.query('analysis_error > 0.0')
    index = []
    for i in range(20):
        df_tmp = df_calc.query("{:.2f} < add_true < {:.2f}".format(-1.05 + 0.1*i,-0.95 + 0.1*i))
        index.append(df_tmp['analysis_error'].idxmin())
        #index.append(df_tmp['RMSE_ratio'].idxmin())
    etkf = df.iloc[index]

    # Export
    N = 40
    add_true = etkf["add_true"].to_numpy()
    inf = etkf["multi_inf"].to_numpy()
    HBH = etkf["HBH"].to_numpy()/N
    HAH = etkf["HAH"].to_numpy()/N
    ob_ob = etkf["ob_ob"].to_numpy()/N
    ab_ob = etkf["ab_ob"].to_numpy()/N
    ab_oa = etkf["ab_oa"].to_numpy()/N
    oa_ob = etkf["oa_ob"].to_numpy()/N

    # with inflation
    HBH = inf * HBH

    add_est1 = 1.0 - ab_ob/HBH
    Ruc_est1 = ob_ob - ab_ob*ab_ob/HBH

    add_est2 = (HAH-ab_oa)/HBH
    Ruc_est2 = (ob_ob - (1.0-add_est2)*(1.0-add_est2)*HBH)
    Ruc_est3 = (oa_ob + add_est2*(1.0-add_est2)*HBH)

    file = open("./estimated_parameters.csv", 'w')
    file.writelines("add_true,add_est1,Ruc_est1,add_est2,Ruc_est2,Ruc_est3\n")
    data = np.empty((20,6))
    data[:,0] = add_true
    data[:,1] = add_est1
    data[:,2] = Ruc_est1
    data[:,3] = add_est2
    data[:,4] = Ruc_est2
    data[:,5] = Ruc_est3
    np.savetxt(file, data, delimiter=',')
    file.close()


def optimal_data():
    # optimal ETKF
    df = pd.read_csv('ETKF_tuning.csv', comment='#')
    df_calc = df.query('analysis_error > 0.0')
    index = []
    for i in range(20):
        df_tmp = df_calc.query("{:.2f} < add_true < {:.2f}".format(-1.05 + 0.1*i,-0.95 + 0.1*i))
        index.append(df_tmp['analysis_error'].idxmin())
    optimal_etkf = df.iloc[index]
    print("ETKF")
    for i in range(20):
        df_tmp = optimal_etkf.query("{:.2f} < add_true < {:.2f}".format(-1.05 + 0.1*i,-0.95 + 0.1*i))
        print("add={:.1f} : inf={:.2f}, r={:.1f}".format(float(df_tmp["add_true"]),float(df_tmp["multi_inf"]),float(df_tmp["r"])))
    print("")

    """
    # optimal ETKFCC
    df = pd.read_csv('ETKFCC_tuning.csv', comment='#')
    df_calc = df.query('analysis_error > 0.0')
    index = []
    for i in range(20):
        df_tmp = df_calc.query("{:.2f} < add_true < {:.2f}".format(-1.05 + 0.1*i,-0.95 + 0.1*i))
        index.append(df_tmp['analysis_error'].idxmin())
    optimal_etkfcc = df.iloc[index]
    print("ETKFCC")
    for i in range(20):
        df_tmp = optimal_etkfcc.query("{:.2f} < add_true < {:.2f}".format(-1.05 + 0.1*i,-0.95 + 0.1*i))
        print("add={:.1f} : inf={:.2f}, r={:.1f}".format(float(df_tmp["add_true"]),float(df_tmp["multi_inf"]),float(df_tmp["Ruc"])))
    print("")
    """

    # estimated ETKFCC
    df = pd.read_csv('estimated_ETKFCC_tuning.csv', comment='#')
    df_calc = df.query('analysis_error > 0.0')
    index = []
    for i in range(20):
        df_tmp = df_calc.query("{:.2f} < add_true < {:.2f}".format(-1.05 + 0.1*i,-0.95 + 0.1*i))
        index.append(df_tmp['analysis_error'].idxmin())
    optimal_etkfcc = df.iloc[index]
    print("estimated ETKFCC")
    for i in range(20):
        df_tmp = optimal_etkfcc.query("{:.2f} < add_true < {:.2f}".format(-1.05 + 0.1*i,-0.95 + 0.1*i))
        print("add={:.1f} : inf={:.2f}, r={:.1f}".format(float(df_tmp["add_true"]),float(df_tmp["multi_inf"]),float(df_tmp["Ruc_sys"])))
    print("")


def all_parameters():
    df = pd.read_csv('ETKF_tuning.csv', comment='#')
    df_calc = df.query('analysis_error > 0.0')
    index = []
    set_inf = np.array([1.0 + 0.01*i for i in range(10)])
    set_inf = np.hstack((set_inf, [1.1 + 0.1*i for i in range(9)]))
    set_inf = np.hstack((set_inf, [2.0 + 1.0*i for i in range(4)]))
    inf_size = set_inf.shape[0]
    for i in range(20):
        df_add = df_calc.query("{:.2f} < add_true < {:.2f}".format(-1.05 + 0.1*i,-0.95 + 0.1*i))
        for j in range(inf_size):
            df_tmp = df_add.query("{:.3f} < multi_inf < {:.3f}".format(set_inf[j]-0.001,set_inf[j]+0.001))
            index.append(df_tmp['analysis_error'].idxmin())
    etkf = df.iloc[index]

    # Export
    N = 40
    add_true = etkf["add_true"].to_numpy()
    inf = etkf["multi_inf"].to_numpy()
    R = etkf["r"].to_numpy()
    HBH = etkf["HBH"].to_numpy()/N
    HAH = etkf["HAH"].to_numpy()/N
    ob_ob = etkf["ob_ob"].to_numpy()/N
    ab_ob = etkf["ab_ob"].to_numpy()/N
    ab_oa = etkf["ab_oa"].to_numpy()/N
    oa_ob = etkf["oa_ob"].to_numpy()/N

    # with inflation
    HBH = inf * HBH

    add_est1 = 1.0 - ab_ob/HBH
    Ruc_est1 = ob_ob - ab_ob*ab_ob/HBH

    add_est2 = (HAH-ab_oa)/HBH
    Ruc_est2 = (ob_ob - (1.0-add_est2)*(1.0-add_est2)*HBH)
    Ruc_est3 = (oa_ob + add_est2*(1.0-add_est2)*HBH)

    file = open("./stability_estimated_parameters.csv", 'w')
    file.writelines("add_true,ETKF_inf,ETKF_R,add_est1,Ruc_est1,add_est2,Ruc_est2,Ruc_est3\n")
    data = np.empty((add_true.shape[0],8))
    data[:,0] = add_true
    data[:,1] = inf
    data[:,2] = R
    data[:,3] = add_est1
    data[:,4] = Ruc_est1
    data[:,5] = add_est2
    data[:,6] = Ruc_est2
    data[:,7] = Ruc_est3
    np.savetxt(file, data, delimiter=',')
    file.close()


##########################################################################
#  出力
##########################################################################
if __name__ == "__main__":
    #optimal_parameters()
    #optimal_data()
    all_parameters()