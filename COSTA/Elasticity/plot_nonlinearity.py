import numpy as np
import quadrature
import FEM
import solvers
import functions
from matplotlib import pyplot as plt, cm
import sys
import sympy as sp
import utils
import parameters

mode = 'bugfix'
if len(sys.argv)>1:
    mode = {'0':'bugfix',
            '1':'quick_test',
            '2':'full_test',
            '3':'el_test'}[sys.argv[1]]
else:
    print('No mode specified')
    print('add arg for mode:')
    print('0: bufix (default)')
    print('1: quick_test')
    print('2: full_test')
    print('syntax e.g.: python testing.py 2')

source = 'non_linear'
static = False
p=1
modelnames=[]
NNkwargs={}
Ne, time_steps, DNNkwargs, pgDNNkwargs, LSTMkwargs, pgLSTMkwargs, NoM, time_delta = parameters.set_args(mode = mode, dim=2)
T = 1

# elasticity
xa,xb,ya,yb = 0,1,0,1
for i in []:#0,1,2]:
    print(f'sol_index: {i}\n')

    # interpol
    f,u,w = functions.manufacture_elasticity_solution(d1=2, d2=2, static=static, non_linear=source=='non_linear', **functions.ELsols[i])
    sol = functions.Solution(T=T, f_raw=f, u_raw=u, zero_source=source=='zero_source', name=f'ELsol{i}',w_raw=w)
    
    model = solvers.Solvers(equation='elasticity', static=static, modelnames=modelnames, p=p,sol=sol, Ne=Ne, time_steps=time_steps,xa=xa, xb=xb, ya=ya,yb=yb,dim=2, skip_create_data=True, NNkwargs=NNkwargs)
    model.alpha_train = model.alpha_test_interpol
    model.alpha_val = []
    model.create_data()
    E_min_i, E_max_i, e_min_i, e_max_i = functions.get_elastic_nonlinearity()
    print(E_min_i, E_max_i, e_min_i, e_max_i)
    
    # extrapol
    f,u,w = functions.manufacture_elasticity_solution(d1=2, d2=2, static=static, non_linear=source=='non_linear', **functions.ELsols[i])
    sol = functions.Solution(T=T, f_raw=f, u_raw=u, zero_source=source=='zero_source', name=f'ELsol{i}',w_raw=w)
    
    model = solvers.Solvers(equation='elasticity', static=static, modelnames=modelnames, p=p,sol=sol, Ne=Ne, time_steps=time_steps,xa=xa, xb=xb, ya=ya,yb=yb,dim=2, skip_create_data=True, NNkwargs=NNkwargs)
    model.alpha_train = model.alpha_test_extrapol
    model.alpha_val = []
    model.create_data()
    E_min_x, E_max_x, e_min_x, e_max_x = functions.get_elastic_nonlinearity()
    print(E_min_x, E_max_x, e_min_x, e_max_x)


    X = np.linspace(-1, max(e_max_x, e_max_i)+1, 40)
    A, c = 10,20
    Y = np.array(A/(2*(c+X)**0.5))
    icol = (0,0.6,0.2)
    xcol = (0.7,0,0)
    extra_tag=''
    figname = f'../master/2d_elastic_figures/{source}/{mode}/{"static_"if static else ""}interpol/E_sol{i}{extra_tag}'
    plt.figure(figsize = (4,4))
    plt.plot(X,Y, 'k',label=r'E=E($\epsilon$)')
    plt.plot(X,X*0+1,'k--', label='E=1')
    plt.text(e_min_x, E_max_x,verticalalignment='center', horizontalalignment='center', s='[', color=xcol, fontweight='bold')#,fontstretch='1',fontsize='x-small')
    plt.text(e_max_x, E_min_x,verticalalignment='center', horizontalalignment='center', s=']', color=xcol, fontweight='bold')#,fontstretch='500',fontsize='x-small')
    plt.text(e_min_i, E_max_i,verticalalignment='center', horizontalalignment='center', s='[', color=icol, fontweight='bold')
    plt.text(e_max_i, E_min_i,verticalalignment='center', horizontalalignment='center', s=']', color=icol, fontweight='bold')
    #plt.ylim(bottom=0)
    #plt.legend()
    plt.grid()
    plt.xlabel(r'$||\epsilon||$')
    #plt.ylabel('E')
    plt.tight_layout()
    plt.savefig(figname+'.pdf')
    #plt.show()

# heat eqn
for sol_index in [3,4,5,6]:
    print(f'sol_index: {sol_index}\n')
    f,u = functions.var_k[sol_index]
    solname = f'var_k{sol_index}'
    sol = functions.Solution(T=T, f_raw=f, u_raw=u, zero_source=not source, name=solname, time_delta=time_delta)
    model = solvers.Solvers(modelnames=modelnames, p=p,sol=sol, Ne=Ne, time_steps=time_steps, skip_create_data=True, NNkwargs=NNkwargs)

    # Interpolation
    model.alpha_train = model.alpha_test_interpol
    model.alpha_val = []
    model.create_data()
    k_min_i, k_max_i, u_min_i, u_max_i = functions.get_heat_nonlinearity()
    print(k_min_i, k_max_i, u_min_i, u_max_i)

    # Extrapolation
    model.alpha_train = model.alpha_test_extrapol
    model.alpha_val = []
    model.create_data()
    k_min_x, k_max_x, u_min_x, u_max_x = functions.get_heat_nonlinearity()
    print(k_min_x, k_max_x, u_min_x, u_max_x)



    extra_tag = '' # for different names when testing specific stuff
    figname = f'../master/1d_heat_figures/{"known_f" if source else "unknown_f"}/{mode}/interpol/k_sol{sol_index}{extra_tag}'
    X = np.linspace(min(u_min_x, u_min_i)-1, max(u_max_x, u_max_i)+1, 40)
    A, c = 10,20
    icol = (0,0.6,0.2)
    xcol = (0.7,0,0)
    plt.figure(figsize = (4,4))
    plt.plot(X,X*0+1,'k--', label='E=1')
    if sol_index == 3:
        Y1 = np.array(X*0+0.5)
        Y2 = np.array(X*0+1.5)
        plt.fill_between(X,Y1,Y2, color='k', alpha = 0.4)
        plt.text(u_min_x, 1,verticalalignment='center', horizontalalignment='center', s='[', color=xcol, fontweight='bold')
        plt.text(u_max_x, 1,verticalalignment='center', horizontalalignment='center', s=']', color=xcol, fontweight='bold')
        plt.text(u_min_i, 1,verticalalignment='center', horizontalalignment='center', s='[', color=icol, fontweight='bold')
        plt.text(u_max_i, 1,verticalalignment='center', horizontalalignment='center', s=']', color=icol, fontweight='bold')
    if sol_index == 4:
        Y1 = np.array(-X*0.5/10+1)
        Y2 = np.array(X*2.5/10+1)
        plt.fill_between(X,Y1,Y2, color='k', alpha = 0.4)
        Y1 = np.array(X*0.7/10+1)
        Y2 = np.array(X*1.5/10+1)
        plt.fill_between(X,Y1,Y2, color='k', alpha = 0.4)
        plt.text(u_min_x, 1,verticalalignment='center', horizontalalignment='center', s='[', color=xcol, fontweight='bold')
        plt.text(u_max_x, 1,verticalalignment='center', horizontalalignment='center', s=']', color=xcol, fontweight='bold')
        plt.text(u_min_i, 1,verticalalignment='center', horizontalalignment='center', s='[', color=icol, fontweight='bold')
        plt.text(u_max_i, 1,verticalalignment='center', horizontalalignment='center', s=']', color=icol, fontweight='bold')
    if sol_index == 5:
        Y = np.exp(X/10)
        plt.plot(X,Y, 'k',label=r'E=E($\epsilon$)')
        plt.text(u_min_x, k_min_x,verticalalignment='center', horizontalalignment='center', s='[', color=xcol, fontweight='bold')
        plt.text(u_max_x, k_max_x,verticalalignment='center', horizontalalignment='center', s=']', color=xcol, fontweight='bold')
        plt.text(u_min_i, k_min_i,verticalalignment='center', horizontalalignment='center', s='[', color=icol, fontweight='bold')
        plt.text(u_max_i, k_max_i,verticalalignment='center', horizontalalignment='center', s=']', color=icol, fontweight='bold')
    if sol_index == 6:
        Y = np.exp(-X/10)
        plt.plot(X,Y, 'k',label=r'E=E($\epsilon$)')
        plt.text(u_min_x, k_max_x,verticalalignment='center', horizontalalignment='center', s='[', color=xcol, fontweight='bold')
        plt.text(u_max_x, k_min_x,verticalalignment='center', horizontalalignment='center', s=']', color=xcol, fontweight='bold')
        plt.text(u_min_i, k_max_i,verticalalignment='center', horizontalalignment='center', s='[', color=icol, fontweight='bold')
        plt.text(u_max_i, k_min_i,verticalalignment='center', horizontalalignment='center', s=']', color=icol, fontweight='bold')

    #plt.ylim(bottom=0)
    #plt.legend()
    plt.grid()
    plt.xlabel(r'$u$')
    plt.ylabel('k')
    plt.tight_layout()
    plt.savefig(figname+'.pdf')
    plt.show()
