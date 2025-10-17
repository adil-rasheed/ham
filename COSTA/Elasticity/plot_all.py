import numpy as np
from matplotlib import pyplot as plt, cm
plt.rc('font', size=13) #increase font text size from 10
import json
from solvers import COLORS


def reset():
    global wins
    global losses
    global winbys
    global losebys
    global addstd
    wins = {
            'FEM' : [0,0,0,0],
            'DNN' : [0,0,0,0],
            'CoSTA_DNN' : [0,0,0,0],
            }
    losses = {
            'FEM' : [0,0,0,0],
            'DNN' : [0,0,0,0],
            'CoSTA_DNN' : [0,0,0,0],
            }
    winbys = {
            'FEM' : [],
            'DNN' : [],
            'CoSTA_DNN' : [],
            }
    losebys = {
            'FEM' : [],
            'DNN' : [],
            'CoSTA_DNN' : [],
            }
    addstd=False

def plot_bars(result_folders, solnames, figname=None):
    
    global wins
    global losses
    global winbys
    global losebys
    global addstd

    # Import the data
    l2_devs_i = []
    l2_devs_x = []
    for solname in solnames:
        with open(result_folders[0]+f'sol{solname}_l2_devs.json') as f:
            l2_devs_i.append(json.load(f))
        with open(result_folders[1]+f'sol{solname}_l2_devs.json') as f:
            l2_devs_x.append(json.load(f))

    # Calculate plotted values
    meandata = np.zeros((4,3))
    stddata = np.zeros((4,3))
    for i, alpha in enumerate([-0.5,0.7,1.5,2.5]):
        l2_devs = l2_devs_i if i%3!=0 else l2_devs_x
        lstsc = {
                'FEM' : [],
                'DNN' : [],
                'CoSTA_DNN' : [],
                }
        for j, name in enumerate(['FEM', 'DNN', 'CoSTA_DNN']):
            means = []
            stds = []
            for k in range(len(solnames)):
                curr_devs = np.array(l2_devs[k][f'{alpha}'][name])
                means.append(np.mean(curr_devs, axis=0))
                lstsc[name].append(means[-1][-1])
                if not name in ['FEM', 'FEM_2']:
                    stds.append(np.std(curr_devs, axis=0, ddof=1))
                    if addstd and not np.isnan(stds[-1][-1]):
                        lstsc[name][-1] += stds[-1][-1]
            meanmean = np.mean(means)
            meanstd = np.mean(stds)
            meandata[i,j] = meanmean
            stddata[i,j] = meanstd

        # count wins/losses:
        for k in range(len(solnames)):
            if lstsc['FEM'][k] < lstsc['DNN'][k] and lstsc['FEM'][k] < lstsc['CoSTA_DNN'][k]:
                wins['FEM'][0] += 1
                winbys['FEM'].append(min(lstsc['DNN'][k]/lstsc['FEM'][k], lstsc['CoSTA_DNN'][k]/lstsc['FEM'][k]))
                if lstsc['FEM'][k]*10**0.5 < lstsc['DNN'][k] and lstsc['FEM'][k]*10**0.5 < lstsc['CoSTA_DNN'][k]:
                    wins['FEM'][1] += 1
                    if lstsc['FEM'][k]*10**1 < lstsc['DNN'][k] and lstsc['FEM'][k]*10**1 < lstsc['CoSTA_DNN'][k]:
                        wins['FEM'][2] += 1
                        if lstsc['FEM'][k]*10**2 < lstsc['DNN'][k] and lstsc['FEM'][k]*10**2 < lstsc['CoSTA_DNN'][k]:
                            wins['FEM'][3] += 1

            elif lstsc['DNN'][k] < lstsc['FEM'][k] and lstsc['DNN'][k] < lstsc['CoSTA_DNN'][k]:
                wins['DNN'][0] += 1
                winbys['DNN'].append(min(lstsc['FEM'][k]/lstsc['DNN'][k], lstsc['CoSTA_DNN'][k]/lstsc['DNN'][k]))
                if lstsc['DNN'][k]*10**0.5 < lstsc['FEM'][k] and lstsc['DNN'][k]*10**0.5 < lstsc['CoSTA_DNN'][k]:
                    wins['DNN'][1] += 1
                    if lstsc['DNN'][k]*10**1 < lstsc['FEM'][k] and lstsc['DNN'][k]*10**1 < lstsc['CoSTA_DNN'][k]:
                        wins['DNN'][2] += 1
                        if lstsc['DNN'][k]*10**2 < lstsc['FEM'][k] and lstsc['DNN'][k]*10**2 < lstsc['CoSTA_DNN'][k]:
                            wins['DNN'][3] += 1

            elif lstsc['CoSTA_DNN'][k] < lstsc['DNN'][k] and lstsc['CoSTA_DNN'][k] < lstsc['FEM'][k]:
                wins['CoSTA_DNN'][0] += 1
                winbys['CoSTA_DNN'].append(min(lstsc['DNN'][k]/lstsc['CoSTA_DNN'][k], lstsc['FEM'][k]/lstsc['CoSTA_DNN'][k]))
                if lstsc['CoSTA_DNN'][k]*10**0.5 < lstsc['DNN'][k] and lstsc['CoSTA_DNN'][k]*10**0.5 < lstsc['FEM'][k]:
                    wins['CoSTA_DNN'][1] += 1
                    if lstsc['CoSTA_DNN'][k]*10**1 < lstsc['DNN'][k] and lstsc['CoSTA_DNN'][k]*10**1 < lstsc['FEM'][k]:
                        wins['CoSTA_DNN'][2] += 1
                        if lstsc['CoSTA_DNN'][k]*10**2 < lstsc['DNN'][k] and lstsc['CoSTA_DNN'][k]*10**2 < lstsc['FEM'][k]:
                            wins['CoSTA_DNN'][3] += 1
            else:
                print('Error', lstsc['CoSTA_DNN'][k], lstsc['DNN'][k], lstsc['FEM'][k])

            if lstsc['FEM'][k] > lstsc['DNN'][k] and lstsc['FEM'][k] > lstsc['CoSTA_DNN'][k]:
                losses['FEM'][0] += 1
                losebys['FEM'].append(min(lstsc['FEM'][k]/lstsc['DNN'][k], lstsc['FEM'][k]/lstsc['CoSTA_DNN'][k]))
                if lstsc['FEM'][k]/10**0.5 > lstsc['DNN'][k] and lstsc['FEM'][k]/10**0.5 > lstsc['CoSTA_DNN'][k]:
                    losses['FEM'][1] += 1
                    if lstsc['FEM'][k]/10**1 > lstsc['DNN'][k] and lstsc['FEM'][k]/10**1 > lstsc['CoSTA_DNN'][k]:
                        losses['FEM'][2] += 1
                        if lstsc['FEM'][k]/10**2 > lstsc['DNN'][k] and lstsc['FEM'][k]/10**2 > lstsc['CoSTA_DNN'][k]:
                            losses['FEM'][3] += 1

            if lstsc['DNN'][k] > lstsc['FEM'][k] and lstsc['DNN'][k] > lstsc['CoSTA_DNN'][k]:
                losses['DNN'][0] += 1
                losebys['DNN'].append(min(lstsc['DNN'][k]/lstsc['FEM'][k], lstsc['DNN'][k]/lstsc['CoSTA_DNN'][k]))
                if lstsc['DNN'][k]/10**0.5 > lstsc['FEM'][k] and lstsc['DNN'][k]/10**0.5 > lstsc['CoSTA_DNN'][k]:
                    losses['DNN'][1] += 1
                    if lstsc['DNN'][k]/10**1 > lstsc['FEM'][k] and lstsc['DNN'][k]/10**1 > lstsc['CoSTA_DNN'][k]:
                        losses['DNN'][2] += 1
                        if lstsc['DNN'][k]/10**2 > lstsc['FEM'][k] and lstsc['DNN'][k]/10**2 > lstsc['CoSTA_DNN'][k]:
                            losses['DNN'][3] += 1

            if lstsc['CoSTA_DNN'][k] > lstsc['DNN'][k] and lstsc['CoSTA_DNN'][k] > lstsc['FEM'][k]:
                losses['CoSTA_DNN'][0] += 1
                losebys['CoSTA_DNN'].append(min(lstsc['CoSTA_DNN'][k]/lstsc['DNN'][k], lstsc['CoSTA_DNN'][k]/lstsc['FEM'][k]))
                if lstsc['CoSTA_DNN'][k]/10**0.5 > lstsc['DNN'][k] and lstsc['CoSTA_DNN'][k]/10**0.5 > lstsc['FEM'][k]:
                    losses['CoSTA_DNN'][1] += 1
                    if lstsc['CoSTA_DNN'][k]/10**1 > lstsc['DNN'][k] and lstsc['CoSTA_DNN'][k]/10**1 > lstsc['FEM'][k]:
                        losses['CoSTA_DNN'][2] += 1
                        if lstsc['CoSTA_DNN'][k]/10**2 > lstsc['DNN'][k] and lstsc['CoSTA_DNN'][k]/10**2 > lstsc['FEM'][k]:
                            losses['CoSTA_DNN'][3] += 1

    return # to not plot bar plots of integrated error, but step function of wins/losses

    # Prepare plotting
    plt.figure(figsize=(6,3.4))
    wd = 0.4 # bar widtho
    x_pos = np.arange(1, 8, 2)

    # Plot
    plt.bar(x_pos, meandata[:,0], color=COLORS['FEM'], width=wd, yerr=stddata[:,0], ecolor='g', capsize=5)
    plt.bar(x_pos+wd, meandata[:,1], color=COLORS['DNN'], width=wd, yerr=stddata[:,1], ecolor='g', capsize=5)
    plt.bar(x_pos+wd*2, meandata[:,2], color=COLORS['CoSTA_DNN'], width=wd, yerr=stddata[:,2], ecolor='g', capsize=5)
    plt.xticks(x_pos+wd, [
        r'$\alpha=-0.5$',
        r'$\alpha=0.7$',
        r'$\alpha=1.5$',
        r'$\alpha=2.5$',
        ])
    plt.ylabel('mean RRMSE')
    plt.tight_layout()

    log = False
    if log:
        plt.yscale('log')
        if figname != None:
            plt.savefig(figname+'_log.pdf')
    else:
        plt.ylim(bottom=0)
        if figname != None:
            plt.savefig(figname+'.pdf')
    plt.close()
    #plt.show()


def plotall():
 
    # lowdimred
    solnames =['DR_0','DR_1','DR_4','DR_5'] 
    result_folders = [f'../master/saved_results/bp_heat/lowdr/full_test/interpol/', f'../master/saved_results/bp_heat/lowdr/full_test/extrapol/']
    plot_bars(result_folders=result_folders,solnames=solnames, figname='../master/all/lowdimred')
    
    # dimred
    solnames =['DR_0','DR_1','DR_4','DR_5'] 
    result_folders = [f'../master/saved_results/bp_heat/full_test/interpol/', f'../master/saved_results/bp_heat/full_test/extrapol/']
    plot_bars(result_folders=result_folders,solnames=solnames, figname='../master/all/dimred')
    
    # vark
    solnames =['var_k3', 'var_k4','var_k5', 'var_k6']
    result_folders = [f'../master/saved_results/1d_heat/known_f/full_test/interpol/', f'../master/saved_results/1d_heat/known_f/full_test/extrapol/']
    plot_bars(result_folders=result_folders,solnames=solnames, figname='../master/all/vark')
    
    # xvark
    solnames =['xvar_k0', 'xvar_k1','xvar_k2', 'xvar_k3']
    result_folders = [f'../master/saved_results/1d_heat/xvark/full_test/interpol/', f'../master/saved_results/1d_heat/xvark/full_test/extrapol/']
    plot_bars(result_folders=result_folders,solnames=solnames, figname='../master/all/xvark')
    
    # exact
    solnames =['ELsol0', 'ELsol1','ELsol2']
    source = 'exact_source'
    result_folders = [f'../master/saved_results/2d_elastic/{source}/el_test/interpol/',f'../master/saved_results/2d_elastic/{source}/el_test/extrapol/']
    plot_bars(result_folders=result_folders,solnames=solnames, figname='../master/all/exact')
    
    # zero
    source = 'zero_source'
    result_folders = [f'../master/saved_results/2d_elastic/{source}/el_test/interpol/',f'../master/saved_results/2d_elastic/{source}/el_test/extrapol/']
    plot_bars(result_folders=result_folders,solnames=solnames, figname='../master/all/zero')
    
    # elastic dimred
    source = 'reduced_source'
    result_folders = [f'../master/saved_results/2d_elastic/{source}/el_test/interpol/',f'../master/saved_results/2d_elastic/{source}/el_test/extrapol/']
    plot_bars(result_folders=result_folders,solnames=solnames, figname='../master/all/eldr')
    
    # non-lin
    source = 'non_linear'
    result_folders = [f'../master/saved_results/2d_elastic/{source}/quick_test/interpol/',f'../master/saved_results/2d_elastic/{source}/quick_test/extrapol/']
    plot_bars(result_folders=result_folders,solnames=solnames, figname='../master/all/nl')

reset()
plotall()
print(wins)
print(losses)
#print(winbys)

for arg in winbys:
    winbys[arg].append(1)
    y = -np.arange(len(winbys[arg]))-1 + len(winbys[arg])
    x = np.sort(np.array(winbys[arg]))
    plt.step(x,y, where='post', color=COLORS[arg], label={"CoSTA_DNN":"CoSTA", "DNN":"DDM", "FEM":"PBM"}[arg])

reset()
addstd = True
plotall()

for arg in winbys:
    winbys[arg].append(1)
    y = -np.arange(len(winbys[arg]))-1 + len(winbys[arg])
    x = np.sort(np.array(winbys[arg]))
    plt.step(x,y, ':', where='post', color=COLORS[arg])#, label=f'{'CoSTA' if arg=='CoSTA_DNN' else arg}')

plt.xscale('log')
plt.ylim(bottom=0)
plt.xlim(left=1)
plt.xlabel('Significance threshold')
plt.ylabel('Number of wins')
plt.legend()
plt.savefig('../master/all/wins.pdf')
plt.show()



reset()
plotall()

for arg in winbys:
    losebys[arg].append(1)
    y = -np.arange(len(losebys[arg]))-1 + len(losebys[arg])
    #print(y)
    x = np.sort(np.array(losebys[arg]))
    #print(x)
    plt.step(x,y, where='post', color=COLORS[arg], label={"CoSTA_DNN":"CoSTA", "DNN":"DDM", "FEM":"PBM"}[arg])
    if arg=='DNN': # workaround since inf is not plotted
        plt.plot([x[-3], 1e5],[2,2], color=COLORS[arg])
        plt.xlim(right=1357.3)

reset()
addstd = True
plotall()

for arg in winbys:
    losebys[arg].append(1)
    y = -np.arange(len(losebys[arg]))-1 + len(losebys[arg])
    x = np.sort(np.array(losebys[arg]))
    plt.step(x,y, ':', where='post', color=COLORS[arg])#, label=f'{'CoSTA' if arg=='CoSTA_DNN' else arg}')

plt.xscale('log')
plt.ylim(bottom=0)
plt.xlim(left=1)
plt.xlabel('Significance threshold')
plt.ylabel('Number of losses')
plt.legend()
plt.savefig('../master/all/losses.pdf')
plt.show()

