import shutil
import numpy as np
import os
import pandas as pd

def ETKF_collect():
    df = pd.read_csv('./ETKF_tuning.csv', comment='#')
    df_calc = df.query('analysis_error > 0.0')
    index = []
    set_inf = np.array([1.0 + 0.01*i for i in range(10)])
    set_inf = np.hstack((set_inf, [1.1 + 0.1*i for i in range(9)]))
    set_inf = np.hstack((set_inf, [2.0 + 1.0*i for i in range(4)]))
    inf_size = set_inf.shape[0]
    for i in range(20):
        df_tmp = df_calc.query("{:.2f} < add_true < {:.2f}".format(-1.05 + 0.1*i,-0.95 + 0.1*i))
        index.append(df_tmp['analysis_error'].idxmin())
    etkf = df.iloc[index]

    os.makedirs("./collect_ETKF", exist_ok=True)

    for i in range(20):
        add = -1.0 + 0.1*i
        df_tmp = etkf.query("{:.2f} < add_true < {:.2f}".format(-1.05 + 0.1*i,-0.95 + 0.1*i))
        inf = float(df_tmp["multi_inf"])
        r = float(df_tmp["r"])
        path = "./ETKF_tuning/time_series_add={:.1f}_inf={:.2f}_R={:.1f}.csv".format(add,inf,r)
        shutil.copy2(path, "./collect_ETKF/")


def ETKFCC_collect():
    df = pd.read_csv('./ETKFCC_tuning.csv', comment='#')
    df_calc = df.query('analysis_error > 0.0')
    index = []
    set_inf = np.array([-1.0 + 0.1*i for i in range(20)])
    inf_size = set_inf.shape[0]
    for i in range(20):
        df_tmp = df_calc.query("{:.2f} < add_true < {:.2f}".format(-1.05 + 0.1*i,-0.95 + 0.1*i))
        index.append(df_tmp['analysis_error'].idxmin())
    etkfcc = df.iloc[index]

    os.makedirs("./collect_ETKFCC", exist_ok=True)

    for i in range(20):
        add = -1.0 + 0.1*i
        df_tmp = etkfcc.query("{:.2f} < add_true < {:.2f}".format(-1.05 + 0.1*i,-0.95 + 0.1*i))
        inf = float(df_tmp["multi_inf"])
        path = "./ETKFCC_tuning/time_series_add={:.1f}_inf={:.2f}.csv".format(add,inf)
        shutil.copy2(path, "./collect_ETKFCC/")    


def collect_estimated_ETKFCC():
    for name in ["aboa_oaob","aboa_obob","abob_obob"]:
        df = pd.read_csv("./"+name+'_ETKFCC.csv', comment='#')
        df_calc = df.query('analysis_error > 0.0')
        index = []
        set_inf = np.array([1.0 + 0.01*i for i in range(21)])
        set_inf = np.hstack((set_inf, [1.3 + 0.1*i for i in range(7)]))
        set_inf = np.hstack((set_inf, [2.0 + 1.0*i for i in range(4)]))
        for i in range(20):
            df_tmp = df_calc.query("{:.2f} < add_true < {:.2f}".format(-1.05 + 0.1*i,-0.95 + 0.1*i))
            try:
                index.append(df_tmp['analysis_error'].idxmin())
            except:
                continue
        etkfcc = df.iloc[index]

        os.makedirs("./collect"+name+"/", exist_ok=True)

        for i in range(20):
            add = -1.0 + 0.1*i
            try:
                df_tmp = etkfcc.query("{:.2f} < add_true < {:.2f}".format(-1.05 + 0.1*i,-0.95 + 0.1*i))
                inf = float(df_tmp["multi_inf"])
                path = name+"_ETKFCC/time_series_add={:.1f}_inf={:.2f}.csv".format(add,inf)
                shutil.copy2(path, "./collect"+name+"/") 
                #print(path)
            except:
                continue  


### 本体 ###
if __name__ == "__main__":
    #ETKF_collect()
    #ETKFCC_collect()
    collect_estimated_ETKFCC()