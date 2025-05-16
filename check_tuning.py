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

plt.rcParams['mathtext.fontset'] = 'cm'

def get_color_code(cname,num):
  cmap = plt.cm.get_cmap(cname,num)
  code_list =[]
  for i in range(cmap.N):
    rgb = cmap(i)[:3]
    #print(rgb2hex(rgb))
    code_list.append(rgb2hex(rgb))
  return code_list

def heatmap_check(innovation1,innovation2):
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
    inf = etkf["multi_inf"].to_numpy()
    HBH = etkf["HBH"].to_numpy()/N
    HBH = inf * HBH
    HAH = etkf["HAH"].to_numpy()/N
    ob_ob = etkf["ob_ob"].to_numpy()/N
    ab_ob = etkf["ab_ob"].to_numpy()/N
    ab_oa = etkf["ab_oa"].to_numpy()/N
    oa_ob = etkf["oa_ob"].to_numpy()/N
    add_est1 = 1.0 - ab_ob/HBH
    Ruc_est1 = ob_ob - ab_ob*ab_ob/HBH
    add_est2 = (HAH-ab_oa)/HBH
    Ruc_est2 = (ob_ob - ((1.0-add_est2)**2)*HBH)
    Ruc_est3 = (oa_ob + add_est2*(1.0-add_est2)*HBH)

    #print(Ruc_est2)

    etkf["add_est_abob"] = add_est1
    etkf["Ruc_est_abob_obob"] = Ruc_est1
    etkf["add_est_aboa"] = add_est2
    etkf["Ruc_est_aboa_obob"] = Ruc_est2
    etkf["Ruc_est_aboa_oaob"] = Ruc_est3

    #print(etkf)

    analysis_RMSE = np.empty((inf_size,20))
    analysis_RMSE_ratio = np.empty((inf_size,20))
    spread_A = np.empty((inf_size,20))
    correlation = np.empty((inf_size,20))
    for i in range(20): # True a
        df_true = etkf.query('{:.2f} < add_true < {:.2f}'.format(-1.0+0.1*i-0.01,-1.0+0.1*i+0.01))
        for j in range(inf_size): # Inflation
            df_tmp = df_true.query('{:.3f} < multi_inf < {:.3f}'.format(set_inf[j]-0.001,set_inf[j]+0.001))
            if float(df_tmp["analysis_error"]) > 0.0 and float(df_tmp["analysis_error"]/df_tmp["observation_error"]) < 1.0:
                analysis_RMSE[j,i] = df_tmp["analysis_error"]
                analysis_RMSE_ratio[j,i] = df_tmp["analysis_error"]/df_tmp["observation_error"]
                spread_A[j,i] = df_tmp["spread_A"]
                correlation[j,i] = df_tmp["correlation"]
            else:
                analysis_RMSE[j,i] = np.nan
                analysis_RMSE_ratio[j,i] = np.nan
                spread_A[j,i] = np.nan
                correlation[j,i] = np.nan

    data = analysis_RMSE_ratio
    
    index = np.argsort(analysis_RMSE, axis=0)
    mask1 = np.zeros_like(data)
    mask1[index[0],np.arange(data.shape[1])] = 1
    mask1 = np.ma.masked_where(mask1 != 1, data)
    data = [data[0:11,:],data[10:20,:],data[19:24,:]]
    mask1 = [mask1[0:11,:],mask1[10:20,:],mask1[19:24,:]]
    color_code = get_color_code("cmc.vik",10)
    fig = plt.figure(figsize = (8, 11))
    gs = gridspec.GridSpec(29, 1, figure=fig)
    cmap = ListedColormap(color_code)
    cmap.set_over('black')
    bounds = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    norm = BoundaryNorm(bounds,cmap.N)
    y_lab = [["1.00","1.05","1.10"],["1.1","1.5","2.0"],["2","5"]]
    y_major_ticks = [np.array([0,5,10]),np.array([0,4,9]),np.array([0,3])]
    y_minor_ticks = [np.arange(11),np.arange(10),np.arange(4)]
    y_lim = [[0,11],[0,10],[0,4]]

    
    for i in range(3):
        if i == 0:
            ax = fig.add_subplot(gs[0:4,:])
            ax.set_title("ETKF",fontsize=25)
            ax.text(0.0,3.0,"(a)",fontsize=20,color="white")
        elif i == 1:
            ax = fig.add_subplot(gs[5:15,:])
            ax.set_ylabel("Inflation parameter",fontsize=20)
            ax.text(0.0,9.0,"(b)",fontsize=20,color="white")
        elif i == 2:
            ax = fig.add_subplot(gs[16:29,:])
            ax.set_xlabel("parameter $a$",fontsize=20)
            ax.text(0.0,10.0,"(c)",fontsize=20,color="white")
        ax.tick_params(labelbottom=False, labelleft=True, labelright=False, labeltop=False)
        c = ax.pcolor(data[2-i], cmap=cmap,norm=norm) # ヒートマップ
        ax.pcolor(mask1[2-i], hatch='//', edgecolor='white', cmap=cmap,norm=norm)
        ax.set_xticks(np.arange(0,21,1)+ 0.5,minor=True)
        ax.set_xticks(np.array([-1.0,-0.5,0.0,0.5,0.9])*10 + 10.5) # x軸目盛の位置
        ax.set_xticklabels(["-1.0","-0.5","0.0","0.5","0.9"],fontsize=20)  # x軸目盛のラベル
        ax.set_yticks(y_major_ticks[2-i] + 0.5) # y軸目盛の位置
        ax.set_yticklabels(y_lab[2-i],fontsize=20)  # y軸目盛のラベル
        ax.set_yticks(y_minor_ticks[2-i]+ 0.5,minor=True)
        #ax.set_aspect('equal', adjustable='box')
        ax.set_ylim(y_lim[2-i][0],y_lim[2-i][1])
        ax.set_xlim(0,20)
    ax.tick_params(labelbottom=True, labelleft=True, labelright=False, labeltop=False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=1.0) #カラーバーを下に表示
    c_bar = fig.colorbar(c, ax=ax, cax=cax, orientation='horizontal', extend='max') #カラーバーを回転
    c_bar.ax.set_xlabel("RMSE",fontsize=20)
    c_bar.ax.set_xticklabels(bounds,fontsize=20)
    fig.subplots_adjust(top=0.965,bottom=0.065)
    plt.savefig("test.png")
    plt.show()

###################################################################
# Parameter Estimation from ETKF
###################################################################
def parameter_estimation(innovation1,innovation2,diff):
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
    inf = etkf["multi_inf"].to_numpy()
    HBH = etkf["HBH"].to_numpy()/N
    HBH = inf * HBH
    HAH = etkf["HAH"].to_numpy()/N
    ob_ob = etkf["ob_ob"].to_numpy()/N
    ab_ob = etkf["ab_ob"].to_numpy()/N
    ab_oa = etkf["ab_oa"].to_numpy()/N
    oa_ob = etkf["oa_ob"].to_numpy()/N

    if innovation1 == "abob":
        add_est = 1.0 - ab_ob/HBH
        Ruc_est = ob_ob - ab_ob*ab_ob/HBH
    elif innovation1 == "aboa":
        add_est = (HAH-ab_oa)/HBH
        if innovation2 == "obob":
            Ruc_est = (ob_ob - ((1.0-add_est)**2)*HBH)
        elif innovation2 == "oaob":
            Ruc_est = (oa_ob + add_est*(1.0-add_est)*HBH)
    else:
        print("check Innovation1")
        return

    #print(Ruc_est2)

    etkf["add_est"] = add_est
    etkf["Ruc_est"] = Ruc_est

    #print(etkf)

    analysis_RMSE = np.empty((inf_size,20))
    add_est = np.empty((inf_size,20))
    Ruc_est = np.empty((inf_size,20))
    for i in range(20): # True a
        df_true = etkf.query('{:.2f} < add_true < {:.2f}'.format(-1.0+0.1*i-0.01,-1.0+0.1*i+0.01))
        for j in range(inf_size): # Inflation
            df_tmp = df_true.query('{:.3f} < multi_inf < {:.3f}'.format(set_inf[j]-0.001,set_inf[j]+0.001))
            if float(df_tmp["analysis_error"]) > 0.0 and float(df_tmp["analysis_error"]/df_tmp["observation_error"]) < 1.0:
                analysis_RMSE[j,i] = df_tmp["analysis_error"]
                if diff == True:
                    add_est[j,i] = df_tmp["add_est"] -(-1.0 + 0.1*i)
                    Ruc_est[j,i] = df_tmp["Ruc_est"] -1.0
                else:
                    add_est[j,i] = df_tmp["add_est"]
                    Ruc_est[j,i] = df_tmp["Ruc_est"]
            else:
                analysis_RMSE[j,i] = np.nan
                add_est[j,i] = np.nan
                Ruc_est[j,i] = np.nan

    est = [add_est, Ruc_est]
    if diff == True:
        bounds = [[-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5],[-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5]]
        title = ["add_est_diff","Ruc_est_diff"]
    else:
        bounds = [[-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]]
        title = ["add_est_row","Ruc_est_row"]

    for k in range(2):
        data = est[k]

        index = np.argsort(analysis_RMSE, axis=0)
        mask1 = np.zeros_like(data)
        mask1[index[0],np.arange(data.shape[1])] = 1
        mask1 = np.ma.masked_where(mask1 != 1, data)
        data = [data[0:11,:],data[10:20,:],data[19:24,:]]
        mask1 = [mask1[0:11,:],mask1[10:20,:],mask1[19:24,:]]
        color_code = get_color_code("cmc.vik",10)
        fig = plt.figure(figsize = (8, 11))
        gs = gridspec.GridSpec(29, 1, figure=fig)
        cmap = ListedColormap(color_code)
        #cmap.set_over('black')
        norm = BoundaryNorm(bounds[k],cmap.N)
        y_lab = [["1.00","1.05","1.10"],["1.1","1.5","2.0"],["2","5"]]
        y_major_ticks = [np.array([0,5,10]),np.array([0,4,9]),np.array([0,3])]
        y_minor_ticks = [np.arange(11),np.arange(10),np.arange(4)]
        y_lim = [[0,11],[0,10],[0,4]]

        
        for i in range(3):
            if i == 0:
                ax = fig.add_subplot(gs[0:4,:])
                ax.set_title(title[k],fontsize=25)
                ax.text(0.0,3.0,"(a)",fontsize=20,color="white")
            elif i == 1:
                ax = fig.add_subplot(gs[5:15,:])
                ax.set_ylabel("Inflation parameter",fontsize=20)
                ax.text(0.0,9.0,"(b)",fontsize=20,color="white")
            elif i == 2:
                ax = fig.add_subplot(gs[16:29,:])
                ax.set_xlabel("parameter $a$",fontsize=20)
                ax.text(0.0,10.0,"(c)",fontsize=20,color="white")
            ax.tick_params(labelbottom=False, labelleft=True, labelright=False, labeltop=False)
            c = ax.pcolor(data[2-i], cmap=cmap,norm=norm) # ヒートマップ
            ax.pcolor(mask1[2-i], hatch='//', edgecolor='white', cmap=cmap,norm=norm)
            ax.set_xticks(np.arange(0,21,1)+ 0.5,minor=True)
            ax.set_xticks(np.array([-1.0,-0.5,0.0,0.5,0.9])*10 + 10.5) # x軸目盛の位置
            ax.set_xticklabels(["-1.0","-0.5","0.0","0.5","0.9"],fontsize=20)  # x軸目盛のラベル
            ax.set_yticks(y_major_ticks[2-i] + 0.5) # y軸目盛の位置
            ax.set_yticklabels(y_lab[2-i],fontsize=20)  # y軸目盛のラベル
            ax.set_yticks(y_minor_ticks[2-i]+ 0.5,minor=True)
            #ax.set_aspect('equal', adjustable='box')
            ax.set_ylim(y_lim[2-i][0],y_lim[2-i][1])
            ax.set_xlim(0,20)
        ax.tick_params(labelbottom=True, labelleft=True, labelright=False, labeltop=False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=1.0) #カラーバーを下に表示
        c_bar = fig.colorbar(c, ax=ax, cax=cax, orientation='horizontal', extend='both') #カラーバーを回転
        c_bar.ax.set_xlabel(title[k],fontsize=20)
        c_bar.ax.set_xticklabels(bounds[k],fontsize=20)
        fig.subplots_adjust(top=0.965,bottom=0.065)
        plt.savefig("Fig_tmp_Parameter_estimation_"+title[k]+".png")
        plt.show()


###################################################################
# Estimation result from ETKF->ETKFCC
###################################################################
def estimation_result(innovation1,innovation2):
    etkf = pd.read_csv('ETKF_tuning.csv', comment='#')
    df_calc = etkf.query('analysis_error > 0.0')
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
    df = etkf.iloc[index]
    df["rmse_ratio"] = df["analysis_error"] / df["observation_error"]
    df = df.reset_index(drop=True)
    row = df.shape[0]

    etkfcc = pd.read_csv('all_'+innovation1+'_'+innovation2+'_ETKFCC_with_inf.csv', comment='#')
    a_rmse = np.empty(row)
    o_rmse = np.empty(row)
    rmse_ratio = np.empty(row)
    for i in range(row):
        df_tmp = etkfcc.iloc[i:i+inf_size]
        df_tmp = df_tmp.query('analysis_error > 0.0')
        try:
            index = df_tmp['analysis_error'].idxmin()
            df_tmp = etkfcc.iloc[index]
            a_rmse[i] = df_tmp["analysis_error"]
            o_rmse[i] = df_tmp["observation_error"]
            rmse_ratio[i] = df_tmp["analysis_error"] / df_tmp["observation_error"]
        except:
            a_rmse[i] = np.nan
            o_rmse[i] = np.nan
            rmse_ratio[i] = np.nan

    
    df["cc_analysis_error"] = a_rmse
    df["cc_observatiom_error"] = o_rmse
    df["cc_rmse_ratio"] = rmse_ratio

    #print(df)

    set_inf = np.array([1.0 + 0.01*i for i in range(10)])
    set_inf = np.hstack((set_inf, [1.1 + 0.1*i for i in range(9)]))
    set_inf = np.hstack((set_inf, [2.0 + 1.0*i for i in range(4)]))
    inf_size = set_inf.shape[0]
    analysis_RMSE = np.empty((inf_size,20))
    etkf = np.empty((inf_size,20))
    etkfcc = np.empty((inf_size,20))
    diff = np.empty((inf_size,20))
    for i in range(20): # True a
        df_true = df.query('{:.2f} < add_true < {:.2f}'.format(-1.0+0.1*i-0.01,-1.0+0.1*i+0.01))
        for j in range(inf_size): # Inflation
            df_tmp = df_true.query('{:.3f} < multi_inf < {:.3f}'.format(set_inf[j]-0.001,set_inf[j]+0.001))
            #if float(df_tmp["analysis_error"]) > 0.0 and float(df_tmp["analysis_error"]/df_tmp["observation_error"]) < 1.0:
            if float(df_tmp["analysis_error"]) > 0.0:
                analysis_RMSE[j,i] = df_tmp["analysis_error"]
                etkf[j,i] = df_tmp["rmse_ratio"]
                etkfcc[j,i] = df_tmp["cc_rmse_ratio"]
                if float(df_tmp["cc_analysis_error"]) > 0.0:
                    diff[j,i] = df_tmp["cc_rmse_ratio"] - df_tmp["rmse_ratio"]
                else:
                    diff[j,i] = np.nan
            else:
                etkf[j,i] = np.nan
                etkfcc[j,i] = np.nan
                diff[j,i] = np.nan
            

        
    result = [etkf, etkfcc,diff]
    bounds = [[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],[-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25]]
    title = ["ETKF","ETKFCC with estimation","difference"]

    for k in range(3):
        data = result[k]

        index = np.argsort(analysis_RMSE, axis=0)
        mask1 = np.zeros_like(data)
        mask1[index[0],np.arange(data.shape[1])] = 1
        mask1 = np.ma.masked_where(mask1 != 1, data)
        data = [data[0:11,:],data[10:20,:],data[19:24,:]]
        mask1 = [mask1[0:11,:],mask1[10:20,:],mask1[19:24,:]]
        color_code = get_color_code("cmc.vik",10)
        fig = plt.figure(figsize = (8, 11))
        gs = gridspec.GridSpec(29, 1, figure=fig)
        cmap = ListedColormap(color_code)
        #cmap.set_over('black')
        norm = BoundaryNorm(bounds[k],cmap.N)
        y_lab = [["1.00","1.05","1.10"],["1.1","1.5","2.0"],["2","5"]]
        y_major_ticks = [np.array([0,5,10]),np.array([0,4,9]),np.array([0,3])]
        y_minor_ticks = [np.arange(11),np.arange(10),np.arange(4)]
        y_lim = [[0,11],[0,10],[0,4]]

        
        for i in range(3):
            if i == 0:
                ax = fig.add_subplot(gs[0:4,:])
                ax.set_title(title[k],fontsize=25)
                ax.text(0.0,3.0,"(a)",fontsize=20,color="white")
            elif i == 1:
                ax = fig.add_subplot(gs[5:15,:])
                ax.set_ylabel("Inflation parameter",fontsize=20)
                ax.text(0.0,9.0,"(b)",fontsize=20,color="white")
            elif i == 2:
                ax = fig.add_subplot(gs[16:29,:])
                ax.set_xlabel("parameter $a$",fontsize=20)
                ax.text(0.0,10.0,"(c)",fontsize=20,color="white")
            ax.tick_params(labelbottom=False, labelleft=True, labelright=False, labeltop=False)
            c = ax.pcolor(data[2-i], cmap=cmap,norm=norm) # ヒートマップ
            ax.pcolor(mask1[2-i], hatch='//', edgecolor='white', cmap=cmap,norm=norm)
            ax.set_xticks(np.arange(0,21,1)+ 0.5,minor=True)
            ax.set_xticks(np.array([-1.0,-0.5,0.0,0.5,0.9])*10 + 10.5) # x軸目盛の位置
            ax.set_xticklabels(["-1.0","-0.5","0.0","0.5","0.9"],fontsize=20)  # x軸目盛のラベル
            ax.set_yticks(y_major_ticks[2-i] + 0.5) # y軸目盛の位置
            ax.set_yticklabels(y_lab[2-i],fontsize=20)  # y軸目盛のラベル
            ax.set_yticks(y_minor_ticks[2-i]+ 0.5,minor=True)
            #ax.set_aspect('equal', adjustable='box')
            ax.set_ylim(y_lim[2-i][0],y_lim[2-i][1])
            ax.set_xlim(0,20)
        ax.tick_params(labelbottom=True, labelleft=True, labelright=False, labeltop=False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=1.0) #カラーバーを下に表示
        c_bar = fig.colorbar(c, ax=ax, cax=cax, orientation='horizontal', extend='both') #カラーバーを回転
        c_bar.ax.set_xlabel(title[k],fontsize=20)
        c_bar.ax.set_xticklabels(bounds[k],fontsize=20)
        fig.subplots_adjust(top=0.965,bottom=0.065)
        plt.savefig("Fig_tmp_RMSE_"+title[k]+".png")
        plt.show()





### 本体 ###
if __name__ == "__main__":
    innovation1 = "abob"
    #innovation1 = "aboa"

    innovation2 = "obob"
    #innovation2 = "oaob"

    #heatmap_check(innovation1,innovation2)

    diff1 = True
    #parameter_estimation(innovation1,innovation2,diff1)

    diff2 = False
    #estimation_result(innovation1,innovation2)
