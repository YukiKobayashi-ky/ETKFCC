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
import multiprocessing

plt.rcParams['mathtext.fontset'] = 'cm'

# その他
def optimal_data(inf,innovation1,innovation2):
    # optimal ETKF
    df = pd.read_csv('ETKF_tuning.csv', comment='#')
    df_calc = df.query('analysis_error > 0.0')
    index = []
    for i in range(20):
        df_tmp = df_calc.query("{:.2f} < add_true < {:.2f}".format(-1.05 + 0.1*i,-0.95 + 0.1*i))
        index.append(df_tmp['analysis_error'].idxmin())
    optimal_etkf = df.iloc[index]
    #if inf == True:
    #    optimal_etkf["HBH"] = optimal_etkf["HBH"] * optimal_etkf["multi_inf"]


    # optimal ETKFCC
    df = pd.read_csv('ETKFCC_tuning.csv', comment='#')
    df_calc = df.query('analysis_error > 0.0')
    index = []
    for i in range(20):
        df_tmp = df_calc.query("{:.2f} < add_true < {:.2f}".format(-1.05 + 0.1*i,-0.95 + 0.1*i))
        index.append(df_tmp['analysis_error'].idxmin())
    optimal_etkfcc = df.iloc[index]

    # Estimated ETKFCC from optimal ETKF
    #if inf == True:
    #    cat = 'with'
    #else:
    #    cat = 'no'
    #df = pd.read_csv(innovation1+'_'+innovation2+'_ETKFCC_'+cat+'_inf.csv', comment='#')
    df = pd.read_csv(innovation1+'_'+innovation2+'_ETKFCC.csv', comment='#')
    df_calc = df.query('analysis_error > 0.0')
    index = []
    for i in range(20):
        df_tmp = df_calc.query("{:.2f} < add_true < {:.2f}".format(-1.05 + 0.1*i,-0.95 + 0.1*i))
        #print(df_tmp.head())
        try:
            index.append(df_tmp['analysis_error'].idxmin())
        except:
            continue
    estimated_etkfcc = df.iloc[index]
    

    return optimal_etkf, optimal_etkfcc, estimated_etkfcc

def print_optimal():
    optimal_etkf, optimal_etkfcc, estimated_etkfcc = optimal_data(inf=True,innovation1="abob",innovation2="obob")
    dfs = [optimal_etkf,optimal_etkfcc,estimated_etkfcc]
    optimal_etkf, optimal_etkfcc, estimated_etkfcc = optimal_data(inf=True,innovation1="aboa",innovation2="obob")
    dfs.append(estimated_etkfcc)
    optimal_etkf, optimal_etkfcc, estimated_etkfcc = optimal_data(inf=True,innovation1="aboa",innovation2="oaob")
    dfs.append(estimated_etkfcc)
    optimal_etkf, optimal_etkfcc, estimated_etkfcc = optimal_data(inf=False,innovation1="abob",innovation2="obob")
    dfs.append(estimated_etkfcc)
    optimal_etkf, optimal_etkfcc, estimated_etkfcc = optimal_data(inf=False,innovation1="aboa",innovation2="obob")
    dfs.append(estimated_etkfcc)
    optimal_etkf, optimal_etkfcc, estimated_etkfcc = optimal_data(inf=False,innovation1="aboa",innovation2="oaob")
    dfs.append(estimated_etkfcc)
    names = ["optimal_etkf","optimal_etkfcc","abob_obob inf","aboa_obob inf","aboa_oaob inf","abob_obob no","aboa_obob no","aboa_oaob no"]
    Rs = ["r","Ruc","Ruc_sys","Ruc_sys","Ruc_sys","Ruc_sys","Ruc_sys","Ruc_sys"]

    for k in range(8):
        print(names[k])
        df = dfs[k]
        for i in range(20):
            df_tmp = df.query("{:.2f} < add_true < {:.2f}".format(-1.05 + 0.1*i,-0.95 + 0.1*i))
            try:
                print("add={:.1f} : inf={:.2f}, r={:.1f}".format(float(df_tmp["add_true"]),float(df_tmp["multi_inf"]),float(df_tmp[Rs[k]])))
            except:
                print("add={:.1f} : error".format(-1 + 0.1*i))
        print("")


def get_color_code(cname,num):
  cmap = plt.get_cmap(cname,num)
  code_list =[]
  for i in range(cmap.N):
    rgb = cmap(i)[:3]
    #print(rgb2hex(rgb))
    code_list.append(rgb2hex(rgb))
  return code_list

##################################################################################
# 本番用
##################################################################################
def fig1_ETKF_data(inf,innovation1,innovation2):
    ### ETKF ###
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

    ### data ###
    etkf_analysis_RMSE = np.empty((inf_size,20))
    for i in range(20): # True a
        df_true = etkf.query('{:.2f} < add_true < {:.2f}'.format(-1.0+0.1*i-0.01,-1.0+0.1*i+0.01))
        for j in range(inf_size): # Inflation
            #print("i={} j={}".format(i,j))
            df_tmp = df_true.query('{:.3f} < multi_inf < {:.3f}'.format(set_inf[j]-0.001,set_inf[j]+0.001))
            try:
                if float(df_tmp["analysis_error"]) > 0.0 and float(df_tmp["analysis_error"]) < float(df_tmp["observation_error"]):
                    etkf_analysis_RMSE[j,i] = float(df_tmp["analysis_error"])
                else:
                    etkf_analysis_RMSE[j,i] = np.nan
            except:
                etkf_analysis_RMSE[j,i] = np.nan
    
    index1 = np.argsort(etkf_analysis_RMSE, axis=0)
    pre_mask1 = np.zeros_like(etkf_analysis_RMSE)
    pre_mask1[index1[0],np.arange(etkf_analysis_RMSE.shape[1])] = 1
    pre_mask1 = [pre_mask1[19:,:],pre_mask1[10:21,:],pre_mask1[:11,:]]

    pre_mask_set = [pre_mask1]

    data = [etkf_analysis_RMSE]
    title = [""]
    label = ["RMSE"]
    colors = [11,11,11]
    
    bounds = [[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]]
    extend = ["max"]

    fig = plt.figure(figsize = (10, 13))
    gs = gridspec.GridSpec(120, 1, figure=fig)
    location = [[gs[0:20,0],gs[15:65,0],gs[60:105,0],gs[100:115,0]]]

    for i in range(1):
        maps = data[i]
        pre_mask = pre_mask_set[i]
        map = [maps[19:,:],maps[10:21,:],maps[:11,:]]
        for j in range(3):
            ax = fig.add_subplot(location[i][j])

            color_code = get_color_code("cmc.vik",colors[i])
            cmap = ListedColormap(color_code)
            cmap.set_bad(color='white')
            norm = BoundaryNorm(bounds[i],cmap.N)

            c = ax.pcolor(map[j], cmap=cmap,norm=norm) # ヒートマップ
            mask = np.ma.masked_where(pre_mask[j] != 1, map[j])
            ax.pcolor(mask, hatch='//', edgecolor='white', cmap=cmap,norm=norm)

            if j == 0:
                ax.set_title(title[i],fontsize=40)
                ax.set_xticks(np.array([-1.0,-0.5,0.0,0.5,0.9])*10 + 10.5) # x軸目盛の位置
                ax.set_xticklabels(["","","","",""],fontsize=20)  # x軸目盛のラベル
                ax.set_xticks(np.arange(0,21,1)+ 0.5,minor=True)
                ax.set_yticks(np.array([0,3]) + 0.5) # y軸目盛の位置
                ax.set_yticklabels([" 2"," 5"],fontsize=20)  # y軸目盛のラベル
                ax.set_yticks(np.arange(0,4,1)+ 0.5,minor=True)
                #ax.set_ylabel("Inflation parameter",fontsize=20)
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlim(0,20)
                ax.set_ylim(0,4)
            elif j == 1:
                ax.set_xticks(np.array([-1.0,-0.5,0.0,0.5,0.9])*10 + 10.5) # x軸目盛の位置
                ax.set_xticklabels(["","","","",""],fontsize=20)  # x軸目盛のラベル
                ax.set_xticks(np.arange(0,21,1)+ 0.5,minor=True)
                ax.set_yticks(np.array([0,4,9]) + 0.5) # y軸目盛の位置
                ax.set_yticklabels([" 1.1"," 1.5"," 2.0"],fontsize=20)  # y軸目盛のラベル
                ax.set_yticks(np.arange(0,11,1)+ 0.5,minor=True)
                ax.set_ylabel("Inflation parameter $\\rho$",fontsize=30)
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlim(0,20)
                ax.set_ylim(0,10)
            elif j == 2:
                ax.set_xticks(np.array([-1.0,-0.5,0.0,0.5,0.9])*10 + 10.5) # x軸目盛の位置
                ax.set_xticklabels(["-1","-0.5","0","0.5","0.9"],fontsize=20)  # x軸目盛のラベル
                ax.set_xticks(np.arange(0,21,1)+ 0.5,minor=True)
                ax.set_xlabel("True $a$",fontsize=30)
                ax.set_yticks(np.array([0,5,10]) + 0.5) # y軸目盛の位置
                ax.set_yticklabels(["1.00","1.05","1.10"],fontsize=20)  # y軸目盛のラベル
                ax.set_yticks(np.arange(0,11,1)+ 0.5,minor=True)
                #ax.set_ylabel("Inflation parameter",fontsize=20)
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlim(0,20)
                ax.set_ylim(0,11)

        ax = fig.add_subplot(location[i][3])
        ax.set_axis_off()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="100%", pad=1.0) #カラーバーを下に表示
        c_bar = fig.colorbar(c, ax=ax, cax=cax, orientation='horizontal', extend=extend[i]) #カラーバーを回転
        c_bar.ax.set_xlabel(label[i],fontsize=30)
        c_bar.ax.set_xticklabels(bounds[i],fontsize=20)    
        ax.set_xlim(0,20)
        ax.set_ylim(0,11)

    fig.subplots_adjust(bottom=0.05,top=0.99)
    plt.savefig("Fig1_ETKF.png")
    plt.show()

def fig2_parameter_estimations(inf,innovation1,innovation2):
    optimal_etkf, optimal_etkfcc, estimated_etkfcc = optimal_data(inf,innovation1,innovation2)
    #optimal_etkf = optimal_data(inf,innovation1,innovation2)
    df = optimal_etkf
    fig = plt.figure(figsize = (10, 24))
    
    # parameter a estimate
    a_true = df['add_true']
    ab_ob = df['ab_ob']
    ab_oa = df['ab_oa']
    inf = df['multi_inf']
    HBH = df['HBH']
    HBH = inf*HBH
    HAH = df['HAH']
    a_est_ab_ob = 1.0 - ab_ob/HBH
    a_est_ab_oa = (HAH-ab_oa)/HBH

    ax1 = fig.add_subplot(3,1,1)
    ax1.set_title("(a) Parameter $a$ estimation", fontsize=20)
    #ax1.text(-1.0,0.88,"(a)",fontsize=20,color="black")
    ax1.set_xlabel("True parameter $a^{true}$", fontsize=20)
    x_major_tick = [-1,-0.5,0,0.5,1]
    x_minor_tick = np.arange(-1.0,1.0,0.1)
    ax1.set_xticks(x_major_tick,minor=False)
    ax1.set_xticklabels(["{}".format(x_major_tick[i]) for i in range(len(x_major_tick))],fontsize=20)
    ax1.set_xticks(x_minor_tick,minor=True)
    ax1.set_xlim(-1.0,1.0)
    ax1.set_ylabel("Estimated parameter $a^{est}$", fontsize=20)
    y_major_tick = [-1,-0.5,0,0.5,1]
    y_minor_tick = np.arange(-1.0,1.0,0.1)
    ax1.set_yticks(y_major_tick,minor=False)
    ax1.set_yticklabels(["{}".format(y_major_tick[i]) for i in range(len(y_major_tick))],fontsize=20)
    ax1.set_yticks(y_minor_tick,minor=True)
    ax1.set_ylim(-1.0,1.0)

    ax1.vlines(0.0, ymin=-5,ymax=2, colors='black', linewidth=0.8)
    ax1.hlines(0.0, xmin=-2,xmax=1, colors='black', linewidth=0.8)
    ax1.plot(np.arange(-1.0,1.1,0.1), np.arange(-1.0,1.1,0.1), ":", color='black', zorder=0)
    ax1.plot(a_true, a_est_ab_ob, "-o", color='black', label='Eq.(29) : $<d^{a-b}(d^{o-b})^\\mathrm{T}>$')
    ax1.plot(a_true, a_est_ab_oa, ":x", color='black', label='Eq.(30) : $<d^{a-b}(d^{o-a})^\\mathrm{T}>$')

    ax1.legend(loc='upper left',fontsize=18)


    # Ruc estimation
    N = 40
    ob_ob = df["ob_ob"]/N
    oa_ob = df["oa_ob"]/N
    inf = df['multi_inf']
    HBH = df['HBH']/N
    HBH = inf*HBH
    HAH = df['HAH']/N

    a_est = [a_est_ab_ob, a_est_ab_oa]
    Ruc = []
    for i in range(2):
        Ruc.append(ob_ob - (1.0-a_est[i])*(1.0-a_est[i])*HBH)
        Ruc.append(oa_ob + a_est[i]*(1.0-a_est[i])*HBH)
    color = ["red","blue","red","blue"]
    line = ["-o","-o",":x",":x"]
    #color = ["orange","orange","c","c"]
    #line = ["-o","--x","-o","--x"]

    ax2 = fig.add_subplot(3,1,2)        
    ax2.set_title("(b) Parameter $r^{uc}$ estimation", fontsize=20)
    #ax2.text(-1.0,0.88,"(a)",fontsize=20,color="black")
    ax2.set_xlabel("True parameter $a^{true}$", fontsize=20)
    x_major_tick = [-1,-0.5,0,0.5,1]
    x_minor_tick = np.arange(-1.0,1.0,0.1)
    ax2.set_xticks(x_major_tick,minor=False)
    ax2.set_xticklabels(["{}".format(x_major_tick[i]) for i in range(len(x_major_tick))],fontsize=20)
    ax2.set_xticks(x_minor_tick,minor=True)
    ax2.set_xlim(-1,1)
    ax2.set_ylabel("Estimated parameter $r^{uc\_est}$", fontsize=20)
    y_major_tick = [0.6,0.8,1,1.2,1.4]
    y_minor_tick = np.arange(0.6,1.4,0.1)
    ax2.set_yticks(y_major_tick,minor=False)
    ax2.set_yticklabels(["{}".format(y_major_tick[i]) for i in range(len(y_major_tick))],fontsize=20)
    ax2.set_yticks(y_minor_tick,minor=True)
    ax2.set_ylim(0.6,1.4)

    ax2.vlines(0.0, ymin=-5,ymax=2, colors='black', linewidth=0.8)
    ax2.hlines(1.0, xmin=-1,xmax=1, colors='black', linewidth=1.5)
    
    for i in [1,0,2,3]:
        ax2.plot(a_true, Ruc[i], line[i], color=color[i])
    
    line_abob_obob, = ax2.plot([-2], [-2], "-o", color="red")
    line_aboa_obob, = ax2.plot([-2], [-2], "--x", color="red")
    line_aboa_oaob, = ax2.plot([-2], [-2], "--x", color="blue")

    ax2.legend([line_abob_obob,line_aboa_obob,line_aboa_oaob], \
                ["Eq.(34) : $<d^{a-b}(d^{o-b})^\\mathrm{T}> \\times <d^{o-b}(d^{o-b})^\\mathrm{T}>$", \
                 "Eq.(35) : $<d^{a-b}(d^{o-a})^\\mathrm{T}> \\times <d^{o-b}(d^{o-b})^\\mathrm{T}>$", \
                 "Eq.(36) : $<d^{a-b}(d^{o-a})^\\mathrm{T}> \\times <d^{o-a}(d^{o-b})^\\mathrm{T}>$"],\
                 loc='upper left',fontsize=18,handler_map={tuple: HandlerTuple(ndivide=None)},handlelength=4,ncol=1)


    # Spread vs RMSE
    ax = fig.add_subplot(3,1,3)
    ax.set_title("(c) Quality of ensemble spreads",fontsize=20)
    ax.set_xlabel("True parameter $a^{true}$", fontsize=20)
    x_major_tick = (-1,-0.5,0,0.5,1)
    ax.set_xticks(x_major_tick,minor=False)
    ax.set_xticklabels(["{}".format(x_major_tick[k]) for k in range(len(x_major_tick))],fontsize=20)
    ax.set_xticks(np.arange(-1,1,0.1),minor=True)
    ax.set_xlim(-1, 1)
    ax.set_ylabel("$\\delta$MSE", fontsize=20)
    y_major_tick = (-0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5)
    ax.set_yticks(y_major_tick, minor=False)
    ax.set_yticklabels(["{}".format(y_major_tick[k]) for k in range(len(y_major_tick))],fontsize=20)
    ax.set_yticks(np.arange(-0.5,3.5,0.1),minor=True)
    ax.set_ylim(-0.5,1.0)

    ax.vlines(0.0, ymin=-3.5,ymax=3.5, colors='black', linewidth=0.8)
    ax.hlines(0.0, xmin=-2,xmax=3, colors='black', linewidth=0.8)

    delta_f = optimal_etkf["multi_inf"]*(np.array(optimal_etkf["spread_B"])**2) - np.array(optimal_etkf["forecast_error"])**2
    delta_a = np.array(optimal_etkf["spread_A"])**2 - np.array(optimal_etkf["analysis_error"])**2
    ax.plot(optimal_etkf["add_true"], delta_f, "-o", color="blue", label="$\\delta{\\widehat{\\sigma^f}}^2$")
    ax.plot(optimal_etkf["add_true"], delta_a, "-o", color="red", label="$\\delta{\\widehat{\\sigma^a}}^2$")
    ax.plot(optimal_etkf["add_true"], optimal_etkf["add_true"]*delta_f, "-o", color="green", label="$a^{true}\\delta{\\widehat{\\sigma^f}}^2$")

    ax.legend(loc='upper left',fontsize=18)


    fig.subplots_adjust(bottom=0.04,top=0.98)
    plt.savefig("Fig2_parameter_estimations.png")
    plt.show()

def fig3_estimated_ETKFCC_tuning(inf,innovation1,innovation2):
    df = pd.read_csv(innovation1+'_'+innovation2+'_ETKFCC.csv', comment='#')

    analysis_RMSE = np.empty((11,20))
    analysis_RMSE_ratio = np.empty((11,20))
    spread_A = np.empty((11,20))
    correlation = np.empty((11,20))

    set_add_true =  [-1.0 + 0.1*i for i in range(20)]
    set_inf = np.array([1.0 + 0.01*i for i in range(11)])
    for i in range(20): # True a
        df_true = df.query('{:.2f} < add_true < {:.2f}'.format(set_add_true[i]-0.01,set_add_true[i]+0.01))
        for j in range(11): # Inflation
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
    index = np.argsort(analysis_RMSE, axis=0)
    pre_mask = np.zeros_like(analysis_RMSE)
    pre_mask[index[0],np.arange(analysis_RMSE.shape[1])] = 1

    data = [analysis_RMSE,analysis_RMSE_ratio,spread_A,correlation]
    title = ["(a) Analysis RMSE","(b) Analysis RMSE ratio","(c) Spread","(d) Correlation"]
    label = ["Analysis RMSE","Analysis RMSE ratio","Spread","Correlation coefficient"]
    colors = [11,11,11,10]
    bounds = [[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] \
              ,[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]]
    extend = ["max","max","max","both"]

    
    fig = plt.figure(figsize = (20, 16))
    for i in range(4):
        map = data[i]
        ax = fig.add_subplot(2,2,i+1)
        #ax.text(-1.0,0.88,"(a)",fontsize=20,color="black")

        color_code = get_color_code("cmc.vik",colors[i])
        cmap = ListedColormap(color_code)
        cmap.set_bad(color='white')
        if i == 1:
            cmap.set_over('white')
        norm = BoundaryNorm(bounds[i],cmap.N)

        c = ax.pcolor(map, cmap=cmap,norm=norm) # ヒートマップ
        mask = np.ma.masked_where(pre_mask != 1, map)
        ax.pcolor(mask, hatch='//', edgecolor='white', cmap=cmap,norm=norm)

        ax.set_title(title[i],fontsize=25)
        ax.set_xticks(np.array([-1.0,-0.5,0.0,0.5,0.9])*10 + 10.5) # x軸目盛の位置
        ax.set_xticklabels(["-1","-0.5","0","0.5","0.9"],fontsize=20)  # x軸目盛のラベル
        ax.set_xticks(np.arange(0,21,1)+ 0.5,minor=True)
        ax.set_xlabel("True parameter $a^{true}$",fontsize=20)
        ax.set_yticks(np.array([0,5,10]) + 0.5) # y軸目盛の位置
        ax.set_yticklabels(["1.00","1.05","1.10"],fontsize=20)  # y軸目盛のラベル
        ax.set_yticks(np.arange(0,11,1)+ 0.5,minor=True)
        ax.set_ylabel("Inflation parameter $\\rho$",fontsize=20)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0,20)
        ax.set_ylim(0,11)

        #ax = fig.add_subplot(location[i][2])
        #ax.set_axis_off()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=1.0) #カラーバーを下に表示
        c_bar = fig.colorbar(c, ax=ax, cax=cax, orientation='horizontal', extend=extend[i]) #カラーバーを回転
        c_bar.ax.set_xlabel(label[i],fontsize=20)
        c_bar.ax.set_xticklabels(bounds[i],fontsize=20)    
        ax.set_xlim(0,20)
        ax.set_ylim(0,11)

    fig.subplots_adjust(bottom=0.07,top=0.99,left=0.08,right=0.98)
    plt.savefig("Fig3_estimated_ETKFCC_tuning.png")
    plt.show()

def fig4_optimal_data(inf,innovation1,innovation2):
    optimal_etkf, optimal_etkfcc, estimated_etkfcc = optimal_data(inf,innovation1,innovation2)

    dfs = [optimal_etkf, optimal_etkfcc, estimated_etkfcc]
    color = ["gray","black","r"]
    line = ["-^","-v","-o"]
    name = ["Optimal ETKF","Optimal ETKFCC","Estimated ETKFCC"]
    a_true = []
    analysis_RMSE_ratio = []
    correlation = []

    for i in range(3):
        df = dfs[i]
        a_true.append(df['add_true'])
        analysis_RMSE_ratio.append(df['analysis_error']/df['observation_error'])
        correlation.append(df['correlation'])
    
    item = [a_true, analysis_RMSE_ratio, correlation]
    label = ["True parameter $a^{true}$", "RMSE ratio", "Correlation coefficient"]
    limit = [(-1,1), (0,1), (-1,1)]
    major_tick = [(-1,-0.5,0,0.5,1), (0,0.2,0.4,0.6,0.8,1), (-1,-0.5,0,0.5,1)]
    minor_tick = [np.arange(-1,1,0.1), np.arange(0,1,0.1), np.arange(-1,1,0.1)]

    XY = [(0,1), (2,1)]
    title = ["(a)","(b)"]

    fig = plt.figure(figsize = (10, 15))
    for i in range(2):
        ax = fig.add_subplot(2,1,i+1)
        ax.set_title(title[i],fontsize=25, loc="left")
        ax.set_xlabel(label[XY[i][0]], fontsize=20)
        ax.set_xticks(major_tick[XY[i][0]],minor=False)
        ax.set_xticklabels(["{}".format(major_tick[XY[i][0]][k]) for k in range(len(major_tick[XY[i][0]]))],fontsize=20)
        ax.set_xticks(minor_tick[XY[i][0]],minor=True)
        ax.set_xlim(limit[XY[i][0]][0],limit[XY[i][0]][1])
        ax.set_ylabel(label[XY[i][1]], fontsize=20)
        ax.set_yticks(major_tick[XY[i][1]],minor=False)
        ax.set_yticklabels(["{}".format(major_tick[XY[i][1]][k]) for k in range(len(major_tick[XY[i][1]]))],fontsize=20)
        ax.set_yticks(minor_tick[XY[i][1]],minor=True)
        ax.set_ylim(limit[XY[i][1]][0],limit[XY[i][1]][1])

        ax.vlines(0.0, ymin=-2,ymax=3, colors='black', linewidth=0.8)
        ax.hlines(0.0, xmin=-2,xmax=3, colors='black', linewidth=0.8)

        for j in range(3):
            ax.plot(item[XY[i][0]][j], item[XY[i][1]][j], line[j], color=color[j], label=name[j], zorder=j)
        ax.legend(loc='upper left',fontsize=15)

        if i == 0:
            ax.scatter(item[XY[i][0]][2], item[XY[i][1]][2], s=40, c=effective_est(inf,innovation1,innovation2), edgecolors='red', zorder=3)

    fig.subplots_adjust(bottom=0.05,top=0.95)
    plt.savefig("Fig4_optimal_data.png")
    plt.show()


def fig5_stability_of_parameter_estimation(innovation1,innovation2):
    ### ETKF ###
    df = pd.read_csv('ETKF_tuning.csv', comment='#')
    df_calc = df.query('analysis_error > 0.0')
    index = []
    set_inf = np.array([1.0 + 0.01*i for i in range(10)])
    set_inf = np.hstack((set_inf, [1.1 + 0.1*i for i in range(9)]))
    set_inf = np.hstack((set_inf, [2.0 + 1.0*i for i in range(4)]))
    #print(set_inf)
    inf_size = set_inf.shape[0]
    for i in range(20):
        df_add = df_calc.query("{:.2f} < add_true < {:.2f}".format(-1.05 + 0.1*i,-0.95 + 0.1*i))
        for j in range(inf_size):
            df_tmp = df_add.query("{:.3f} < multi_inf < {:.3f}".format(set_inf[j]-0.001,set_inf[j]+0.001))
            index.append(df_tmp['analysis_error'].idxmin())
    etkf = df.iloc[index]

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

    etkf["add_est"] = add_est
    etkf["Ruc_est"] = Ruc_est

    ### data ###
    etkf_analysis_RMSE = np.empty((inf_size,20))
    etkf_analysis_RMSE_ratio = np.empty((inf_size,20))
    add_diff = np.empty((inf_size,20))
    Ruc_diff = np.empty((inf_size,20))
    for i in range(20): # True a
        df_true = etkf.query('{:.2f} < add_true < {:.2f}'.format(-1.0+0.1*i-0.01,-1.0+0.1*i+0.01))
        for j in range(inf_size): # Inflation
            #print("i={} j={}".format(i,j))
            df_tmp = df_true.query('{:.3f} < multi_inf < {:.3f}'.format(set_inf[j]-0.001,set_inf[j]+0.001))
            try:
                if float(df_tmp["analysis_error"]) > 0.0 and float(df_tmp["analysis_error"]) < float(df_tmp["observation_error"]):
                    etkf_analysis_RMSE[j,i] = df_tmp["analysis_error"]
                    etkf_analysis_RMSE_ratio[j,i] = df_tmp["analysis_error"]/df_tmp["observation_error"]
                    add_diff[j,i] = df_tmp["add_est"] -(-1.0 + 0.1*i)
                    Ruc_diff[j,i] = df_tmp["Ruc_est"] -1.0
                else:
                    etkf_analysis_RMSE[j,i] = np.nan
                    etkf_analysis_RMSE_ratio[j,i] = np.nan
                    add_diff[j,i] = np.nan
                    Ruc_diff[j,i] = np.nan
            except:
                etkf_analysis_RMSE[j,i] = np.nan
                etkf_analysis_RMSE_ratio[j,i] = np.nan
                add_diff[j,i] = np.nan
                Ruc_diff[j,i] = np.nan
    
    index = np.argsort(etkf_analysis_RMSE, axis=0)
    pre_mask = np.zeros_like(etkf_analysis_RMSE)
    pre_mask[index[0],np.arange(etkf_analysis_RMSE.shape[1])] = 1
    pre_mask = [pre_mask[19:,:],pre_mask[10:21,:],pre_mask[:11,:]]

    data = [add_diff,Ruc_diff]
    title = ["(a) $a^{est}$ in ETKF","(b) $r^{uc\_est}$ in ETKF"]
    label = ["$a^{est} - a^{true}$","$r^{uc\_est} - r^{uc\_true}$"]
    colors = [10,10,10,10]
    
    bounds = [[-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5],[-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25]]
    extend = ["both","both"]

    fig = plt.figure(figsize = (20, 15))
    gs = gridspec.GridSpec(120, 2, figure=fig)
    location = [[gs[0:16,0],gs[17:62,0],gs[60:110,0],gs[105:120,0]], [gs[0:16,1],gs[17:62,1],gs[60:110,1],gs[105:120,1]]]

    for i in range(2):
        maps = data[i]
        map = [maps[19:,:],maps[10:21,:],maps[:11,:]]
        for j in range(3):
            ax = fig.add_subplot(location[i][j])
            ax.set_facecolor("white")
            #ax.text(-1.0,0.88,"(a)",fontsize=20,color="black")

            color_code = get_color_code("cmc.vik",colors[i])
            cmap = ListedColormap(color_code)
            #cmap.set_over('black')
            norm = BoundaryNorm(bounds[i],cmap.N)

            c = ax.pcolor(map[j], cmap=cmap,norm=norm) # ヒートマップ
            mask = np.ma.masked_where(pre_mask[j] != 1, map[j])
            ax.pcolor(mask, hatch='//', edgecolor='white', cmap=cmap,norm=norm)

            if j == 0:
                ax.set_title(title[i],fontsize=40)
                ax.set_xticks(np.array([-1.0,-0.5,0.0,0.5,0.9])*10 + 10.5) # x軸目盛の位置
                ax.set_xticklabels(["","","","",""],fontsize=20)  # x軸目盛のラベル
                ax.set_xticks(np.arange(0,21,1)+ 0.5,minor=True)
                ax.set_yticks(np.array([0,3]) + 0.5) # y軸目盛の位置
                ax.set_yticklabels([" 2"," 5"],fontsize=20)  # y軸目盛のラベル
                ax.set_yticks(np.arange(0,4,1)+ 0.5,minor=True)
                #ax.set_ylabel("Inflation parameter",fontsize=20)
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlim(0,20)
                ax.set_ylim(0,4)
            elif j == 1:
                ax.set_xticks(np.array([-1.0,-0.5,0.0,0.5,0.9])*10 + 10.5) # x軸目盛の位置
                ax.set_xticklabels(["","","","",""],fontsize=20)  # x軸目盛のラベル
                ax.set_xticks(np.arange(0,21,1)+ 0.5,minor=True)
                ax.set_yticks(np.array([0,4,9]) + 0.5) # y軸目盛の位置
                ax.set_yticklabels([" 1.1"," 1.5"," 2.0"],fontsize=20)  # y軸目盛のラベル
                ax.set_yticks(np.arange(0,11,1)+ 0.5,minor=True)
                ax.set_ylabel("Inflation parameter $\\rho$ in ETKF",fontsize=40)
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlim(0,20)
                ax.set_ylim(0,10)
            elif j == 2:
                ax.set_xticks(np.array([-1.0,-0.5,0.0,0.5,0.9])*10 + 10.5) # x軸目盛の位置
                ax.set_xticklabels(["-1","-0.5","0","0.5","0.9"],fontsize=20)  # x軸目盛のラベル
                ax.set_xticks(np.arange(0,21,1)+ 0.5,minor=True)
                ax.set_xlabel("True parameter $a^{true}$",fontsize=40)
                ax.set_yticks(np.array([0,5,10]) + 0.5) # y軸目盛の位置
                ax.set_yticklabels(["1.00","1.05","1.10"],fontsize=20)  # y軸目盛のラベル
                ax.set_yticks(np.arange(0,11,1)+ 0.5,minor=True)
                #ax.set_ylabel("Inflation parameter",fontsize=20)
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlim(0,20)
                ax.set_ylim(0,11)

        ax = fig.add_subplot(location[i][3])
        ax.set_axis_off()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="100%", pad=1.0) #カラーバーを下に表示
        c_bar = fig.colorbar(c, ax=ax, cax=cax, orientation='horizontal', extend=extend[i]) #カラーバーを回転
        c_bar.ax.set_xlabel(label[i],fontsize=40)
        c_bar.ax.set_xticklabels(bounds[i],fontsize=20)    
        ax.set_xlim(0,20)
        ax.set_ylim(0,11)

    fig.subplots_adjust(bottom=0.07,top=0.95,left=0.08,right=0.98)
    plt.savefig("Fig5_stability_of_parameter_estimation.png")
    plt.show()

def fig6_stability_of_accuracy(innovation1,innovation2):
    ### ETKF ###
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

    etkf["add_est"] = add_est
    etkf["Ruc_est"] = Ruc_est


    ### estimated ETKFCC ###
    if innovation1 == "abob":
        df = pd.read_csv('stability_abob_obob_ETKFCC_with_inf.csv', comment='#')
    elif innovation1 == "aboa":
        if innovation2 == "obob":
            df = pd.read_csv('stability_aboa_obob_ETKFCC_with_inf.csv', comment='#')
        elif innovation2 == "oaob":
            df = pd.read_csv('stability_aboa_oaob_ETKFCC_with_inf.csv', comment='#')

    df_calc = df.query('analysis_error > 0.0')
    index = []
    set_inf = np.array([1.0 + 0.01*i for i in range(10)])
    set_inf = np.hstack((set_inf, [1.1 + 0.1*i for i in range(9)]))
    set_inf = np.hstack((set_inf, [2.0 + 1.0*i for i in range(4)]))
    inf_size = set_inf.shape[0]
    for i in range(20):
        df_add = df_calc.query("{:.2f} < add_true < {:.2f}".format(-1.05 + 0.1*i,-0.95 + 0.1*i))
        for j in range(inf_size):
            df_tmp = df_add.query("{:.3f} < ETKF_inf < {:.3f}".format(set_inf[j]-0.001,set_inf[j]+0.001))
            try:
                index.append(df_tmp['analysis_error'].idxmin())
            except:
                #print("add={:.1f}, inf={:.2f} : error".format(-1 + 0.1*i, set_inf[j]))
                pass
    estimated_etkfcc = df.iloc[index]


    ### data ###
    etkf_analysis_RMSE_ratio = np.empty((inf_size,20))
    etkfcc_analysis_RMSE_ratio = np.empty((inf_size,20))
    ratio_diff = np.empty((inf_size,20))
    for i in range(20): # True a
        df_true = etkf.query('{:.2f} < add_true < {:.2f}'.format(-1.0+0.1*i-0.01,-1.0+0.1*i+0.01))
        df_true_cc = estimated_etkfcc.query('{:.2f} < add_true < {:.2f}'.format(-1.0+0.1*i-0.01,-1.0+0.1*i+0.01))
        for j in range(inf_size): # Inflation
            #print("i={} j={}".format(i,j))
            df_tmp = df_true.query('{:.3f} < multi_inf < {:.3f}'.format(set_inf[j]-0.001,set_inf[j]+0.001))
            df_tmp_cc = df_true_cc.query('{:.3f} < ETKF_inf < {:.3f}'.format(set_inf[j]-0.001,set_inf[j]+0.001))
            try:
                if float(df_tmp["analysis_error"]) > 0.0 and  float(df_tmp["analysis_error"]) < float(df_tmp["observation_error"]):
                    etkf_analysis_RMSE_ratio[j,i] = df_tmp["analysis_error"]/df_tmp["observation_error"]
                else:
                    etkf_analysis_RMSE_ratio[j,i] = np.nan
            except:
                etkf_analysis_RMSE_ratio[j,i] = np.nan

            try:                        
                if float(df_tmp["analysis_error"]) > 0.0 and float(df_tmp_cc["analysis_error"]) > 0.0 and float(df_tmp["analysis_error"]) < float(df_tmp["observation_error"]) and float(df_tmp_cc["analysis_error"]) < float(df_tmp_cc["observation_error"]):
                    etkfcc_analysis_RMSE_ratio[j,i] = df_tmp_cc["analysis_error"]/df_tmp_cc["observation_error"]   
                    ratio_diff[j,i] = etkfcc_analysis_RMSE_ratio[j,i] - etkf_analysis_RMSE_ratio[j,i]
                else:
                    etkfcc_analysis_RMSE_ratio[j,i] = np.nan
                    ratio_diff[j,i] = np.nan
            except:
                etkfcc_analysis_RMSE_ratio[j,i] = np.nan
                ratio_diff[j,i] = np.nan
    
    index1 = np.argsort(etkf_analysis_RMSE_ratio, axis=0)
    pre_mask1 = np.zeros_like(etkf_analysis_RMSE_ratio)
    pre_mask1[index1[0],np.arange(etkf_analysis_RMSE_ratio.shape[1])] = 1
    pre_mask1 = [pre_mask1[19:,:],pre_mask1[10:21,:],pre_mask1[:11,:]]
    index2 = np.argsort(etkfcc_analysis_RMSE_ratio, axis=0)
    pre_mask2 = np.zeros_like(etkfcc_analysis_RMSE_ratio)
    pre_mask2[index2[0],np.arange(etkfcc_analysis_RMSE_ratio.shape[1])] = 1
    pre_mask2 = [pre_mask2[19:,:],pre_mask2[10:21,:],pre_mask2[:11,:]]
    pre_mask3 = np.zeros_like(ratio_diff)
    pre_mask3 = [pre_mask3[19:,:],pre_mask3[10:21,:],pre_mask3[:11,:]]

    pre_mask_set = [pre_mask1,pre_mask2,pre_mask3]

    data = [etkf_analysis_RMSE_ratio,etkfcc_analysis_RMSE_ratio,ratio_diff]
    title = ["(a) ETKF","(b) ETKFCC using $a^{est}$ and $r^{uc\_est}$\n  from  ETKF","(c) Difference"]
    #label = ["RMSE ratio","RMSE ratio","$\\delta$RMSE ratio"]
    label = ["RMSE ratio","RMSE ratio","ETKF - ETKFCC"]
    colors = [11,11,11]
    
    bounds = [[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],[-0.1,-0.08,-0.06,-0.04,-0.02,0,0.02,0.04,0.06,0.08,0.1]]
    extend = ["max","max","both"]

    fig = plt.figure(figsize = (30, 15))
    gs = gridspec.GridSpec(120, 3, figure=fig)
    location = [[gs[0:16,0],gs[17:62,0],gs[60:110,0],gs[105:120,0]], [gs[0:16,1],gs[17:62,1],gs[60:110,1],gs[105:120,1]], [gs[0:16,2],gs[17:62,2],gs[60:110,2],gs[105:120,2]]]

    for i in range(3):
        maps = data[i]
        pre_mask = pre_mask_set[i]
        map = [maps[19:,:],maps[10:21,:],maps[:11,:]]
        for j in range(3):
            ax = fig.add_subplot(location[i][j])
            
            #ax.text(-1.0,0.88,"(a)",fontsize=20,color="black")

            color_code = get_color_code("cmc.vik",colors[i])
            cmap = ListedColormap(color_code)
            cmap.set_bad("white")
            #cmap.set_over('black')
            norm = BoundaryNorm(bounds[i],cmap.N)

            cmap.set_bad(color='white')
            if i != 2:
                cmap.set_over('white')

            c = ax.pcolor(map[j], cmap=cmap,norm=norm) # ヒートマップ
            mask = np.ma.masked_where(pre_mask[j] != 1, map[j])
            ax.pcolor(mask, hatch='//', edgecolor='white', cmap=cmap,norm=norm)

            if j == 0:
                ax.set_title(title[i],fontsize=40)
                ax.set_xticks(np.array([-1.0,-0.5,0.0,0.5,0.9])*10 + 10.5) # x軸目盛の位置
                ax.set_xticklabels(["","","","",""],fontsize=20)  # x軸目盛のラベル
                ax.set_xticks(np.arange(0,21,1)+ 0.5,minor=True)
                ax.set_yticks(np.array([0,3]) + 0.5) # y軸目盛の位置
                ax.set_yticklabels([" 2"," 5"],fontsize=20)  # y軸目盛のラベル
                ax.set_yticks(np.arange(0,4,1)+ 0.5,minor=True)
                #ax.set_ylabel("Inflation parameter",fontsize=20)
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlim(0,20)
                ax.set_ylim(0,4)
            elif j == 1:
                ax.set_xticks(np.array([-1.0,-0.5,0.0,0.5,0.9])*10 + 10.5) # x軸目盛の位置
                ax.set_xticklabels(["","","","",""],fontsize=20)  # x軸目盛のラベル
                ax.set_xticks(np.arange(0,21,1)+ 0.5,minor=True)
                ax.set_yticks(np.array([0,4,9]) + 0.5) # y軸目盛の位置
                ax.set_yticklabels([" 1.1"," 1.5"," 2.0"],fontsize=20)  # y軸目盛のラベル
                ax.set_yticks(np.arange(0,11,1)+ 0.5,minor=True)
                ax.set_ylabel("Inflation parameter $\\rho$ in ETKF",fontsize=40)
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlim(0,20)
                ax.set_ylim(0,10)
            elif j == 2:
                ax.set_xticks(np.array([-1.0,-0.5,0.0,0.5,0.9])*10 + 10.5) # x軸目盛の位置
                ax.set_xticklabels(["-1","-0.5","0","0.5","0.9"],fontsize=20)  # x軸目盛のラベル
                ax.set_xticks(np.arange(0,21,1)+ 0.5,minor=True)
                ax.set_xlabel("True parameter $a^{true}$",fontsize=40)
                ax.set_yticks(np.array([0,5,10]) + 0.5) # y軸目盛の位置
                ax.set_yticklabels(["1.00","1.05","1.10"],fontsize=20)  # y軸目盛のラベル
                ax.set_yticks(np.arange(0,11,1)+ 0.5,minor=True)
                #ax.set_ylabel("Inflation parameter",fontsize=20)
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlim(0,20)
                ax.set_ylim(0,11)

        ax = fig.add_subplot(location[i][3])
        ax.set_axis_off()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="100%", pad=1.0) #カラーバーを下に表示
        c_bar = fig.colorbar(c, ax=ax, cax=cax, orientation='horizontal', extend=extend[i]) #カラーバーを回転
        c_bar.ax.set_xlabel(label[i],fontsize=40)
        c_bar.ax.set_xticklabels(bounds[i],fontsize=20)    
        ax.set_xlim(0,20)
        ax.set_ylim(0,11)

    fig.subplots_adjust(bottom=0.07,top=0.90,left=0.05,right=0.98)
    plt.savefig("Fig6_stability_of_accuracy.png")
    plt.show()

### p test ###
def bootstrap_RMSE_ratio_unit(df):
    df_sample = df.sample(n=df.shape[0], replace=True)
    RMSE_ratio = df_sample.mean().to_numpy()
    #print(RMSE_ratio)
    return RMSE_ratio

def bootstrap_RMSE_ratio(inf,innovation1,innovation2):
    print("######  sampling RMSE-ratio with bootstrap  ######")
    sample_num = int(1e+05)
    #sample_num = 8

    optimal_etkf, optimal_etkfcc, estimated_etkfcc = optimal_data(inf,innovation1,innovation2)
    data = np.empty([sample_num,0])
    line = ""

    for i in range(20):
        add = -1.0+0.1*i
        print("add={:.1f}".format(add))
        optimal_etkf_parameters = optimal_etkf.query("{:.2f} < add_true < {:.2f}".format(-1.05 + 0.1*i,-0.95 + 0.1*i))
        optimal_etkf_time_series = pd.read_csv('./collect_ETKF/time_series_add={:.1f}_inf={:.2f}_R={:.1f}.csv'.format( \
                                                float(optimal_etkf_parameters["add_true"]),float(optimal_etkf_parameters["multi_inf"]),float(optimal_etkf_parameters["r"])) \
                                                , comment='#' ).query("time > 1460.0")
        optimal_etkf_time_series["opt_ETKF_RMSE_ratio"] = optimal_etkf_time_series["analysis_RMSE"] / optimal_etkf_time_series["observation_RMSE"]
        optimal_etkfcc_parameters = optimal_etkfcc.query("{:.2f} < add_true < {:.2f}".format(-1.05 + 0.1*i,-0.95 + 0.1*i))
        optimal_etkfcc_time_series = pd.read_csv('./collect_ETKFCC/time_series_add={:.1f}_inf={:.2f}.csv'.format(\
                                                float(optimal_etkfcc_parameters["add_true"]),float(optimal_etkfcc_parameters["multi_inf"])) \
                                                , comment='#' ).query("time > 1460.0")
        optimal_etkfcc_time_series["opt_ETKFCC_RMSE_ratio"] = optimal_etkfcc_time_series["analysis_RMSE"] / optimal_etkfcc_time_series["observation_RMSE"]
        estimated_etkfcc_parameters = estimated_etkfcc.query("{:.2f} < add_true < {:.2f}".format(-1.05 + 0.1*i,-0.95 + 0.1*i))
        estimated_etkfcc_time_series = pd.read_csv('./collect{}_{}/time_series_add={:.1f}_inf={:.2f}.csv'.format(innovation1,innovation2, \
                                                float(estimated_etkfcc_parameters["add_true"]),float(estimated_etkfcc_parameters["multi_inf"])) \
                                                , comment='#' ).query("time > 1460.0")
        estimated_etkfcc_time_series["est_ETKFCC_RMSE_ratio"] = estimated_etkfcc_time_series["analysis_RMSE"] / estimated_etkfcc_time_series["observation_RMSE"]
        
        time_series = pd.concat([optimal_etkf_time_series["opt_ETKF_RMSE_ratio"], optimal_etkfcc_time_series["opt_ETKFCC_RMSE_ratio"], estimated_etkfcc_time_series["est_ETKFCC_RMSE_ratio"]],axis=1)

        #print(time_series.head())
        #print(time_series.mean())
        with multiprocessing.Pool(processes=8) as pool:
            result = pool.map(bootstrap_RMSE_ratio_unit,[time_series for i in range(sample_num)])
        data = np.hstack([data,result])
        line = line + "opt_ETKF_add={:.1f},opt_ETKFCC_add={:.1f},est_ETKFCC_add={:.1f},".format(add,add,add)
    
    line = line.strip(",") + "\n"

    file = open("RMSE_ratio_sample.csv", 'w')
    file.writelines("#sample_size={}\n".format(sample_num))
    file.writelines(line)
    np.savetxt(file, data, delimiter=',')
    file.close()
        
def p_test_RMSE_ratio():
    print("########### RMSE ratio #################")
    df = pd.read_csv('RMSE_ratio_sample.csv', comment='#')
    sample_num = df.shape[0]

    print("optimal ETKF V.S. optimal ETKFCC")
    
    for i in range(20):
        add = -1.0+0.1*i
        diff = df["opt_ETKF_add={:.1f}".format(add)].to_numpy() - df["opt_ETKFCC_add={:.1f}".format(add)].to_numpy()
        positive = np.count_nonzero(diff > 0)
        negative = np.count_nonzero(diff <= 0)
        print("add={:.1f} : p-value={}   (positive={}, check={})".format(add, positive/sample_num, positive, sample_num-(positive+negative)))
    print("")

    print("optimal ETKF V.S. estimated ETKFCC")
    for i in range(20):
        add = -1.0+0.1*i
        diff = df["opt_ETKF_add={:.1f}".format(add)].to_numpy() - df["est_ETKFCC_add={:.1f}".format(add)].to_numpy()
        positive = np.count_nonzero(diff > 0)
        negative = np.count_nonzero(diff <= 0)
        print("add={:.1f} : p-value={}   (positive={}, check={})".format(add, positive/sample_num, positive, sample_num-(positive+negative)))
    print("")

def effective_opt(inf,innovation1,innovation2):
    df = pd.read_csv('RMSE_ratio_sample.csv', comment='#')
    sample_num = df.shape[0]

    optimal_etkf, optimal_etkfcc, estimated_etkfcc = optimal_data(inf,innovation1,innovation2)
    opt_etkf_RMSE_ratio = optimal_etkf['analysis_error'].to_numpy()/optimal_etkf['observation_error'].to_numpy()
    opt_etkfcc_RMSE_ratio = optimal_etkfcc['analysis_error'].to_numpy()/optimal_etkfcc['observation_error'].to_numpy()

    face_colors = []
    for i in range(20):
        add = -1.0+0.1*i
        # probability
        diff = df["opt_ETKF_add={:.1f}".format(add)].to_numpy() - df["opt_ETKFCC_add={:.1f}".format(add)].to_numpy()
        positive = np.count_nonzero(diff > 0)
        probability = positive/sample_num
        # accuracy
        accuracy = 1.0 - opt_etkfcc_RMSE_ratio[i]/opt_etkf_RMSE_ratio[i]

        if probability > 0.99 and accuracy > 0.05:
            face_colors.append("white")
        else:
            face_colors.append("red")

    #print(face_colors)
    return face_colors

def effective_est(inf,innovation1,innovation2):
    df = pd.read_csv('RMSE_ratio_sample.csv', comment='#')
    sample_num = df.shape[0]

    optimal_etkf, optimal_etkfcc, estimated_etkfcc = optimal_data(inf,innovation1,innovation2)
    opt_etkf_RMSE_ratio = optimal_etkf['analysis_error'].to_numpy()/optimal_etkf['observation_error'].to_numpy()
    est_etkfcc_RMSE_ratio = estimated_etkfcc['analysis_error'].to_numpy()/estimated_etkfcc['observation_error'].to_numpy()

    face_colors = []
    for i in range(20):
        add = -1.0+0.1*i
        # probability
        diff = df["opt_ETKF_add={:.1f}".format(add)].to_numpy() - df["est_ETKFCC_add={:.1f}".format(add)].to_numpy()
        positive = np.count_nonzero(diff > 0)
        probability = positive/sample_num
        # accuracy
        accuracy = 1.0 - est_etkfcc_RMSE_ratio[i]/opt_etkf_RMSE_ratio[i]

        if probability > 0.999 and accuracy > 0.05:
            face_colors.append("white")
        else:
            face_colors.append("red")

    #print(face_colors)
    return face_colors




if __name__ == "__main__":
    inf = True
    #inf = False

    innovation1 = "abob"
    #innovation1 = "aboa"

    innovation2 = "obob"
    #innovation2 = "oaob"

    #print_optimal()

    fig1_ETKF_data(inf,innovation1,innovation2)
    fig2_parameter_estimations(inf,innovation1,innovation2)
    fig3_estimated_ETKFCC_tuning(inf,innovation1,innovation2)
    fig4_optimal_data(inf,innovation1,innovation2)

    fig5_stability_of_parameter_estimation(innovation1,innovation2)
    fig6_stability_of_accuracy(innovation1,innovation2)

    #bootstrap_RMSE_ratio(inf,innovation1,innovation2)
    #p_test_RMSE_ratio()
    #effective(inf,innovation1,innovation2)
    pass
