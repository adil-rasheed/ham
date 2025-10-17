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

Ne, time_steps, DNNkwargs, pgDNNkwargs, LSTMkwargs, pgLSTMkwargs, NoM, time_delta = parameters.set_args(mode = mode, dim=2)
#time_steps = 100

source = 'reduced_source'
print('Using', source)

if len(sys.argv)>3:
    static = bool(sys.argv[3])
else:
    static = False
print(f'Using {"STATIC" if static else "TRANSIENT"} version of elastic equation')

p=1

modelnames = {
        'DNN' : NoM,
        #'pgDNN' : NoM,
        #'LSTM' : NoM,
        #'pgLSTM' : NoM,
        'CoSTA_DNN' : NoM,
        #'CoSTA_pgDNN' : NoM,
        #'CoSTA_LSTM' : NoM,
        #'CoSTA_pgLSTM' : NoM,
        }
NNkwargs = {
        'DNN':DNNkwargs,
        'CoSTA_DNN':DNNkwargs,
        'pgDNN' : pgDNNkwargs,
        'CoSTA_pgDNN':pgDNNkwargs,
        'LSTM':LSTMkwargs,
        'pgLSTM':pgLSTMkwargs,
        'CoSTA_LSTM':LSTMkwargs,
        'CoSTA_pgLSTM':pgLSTMkwargs,
        }

class Fem_method:
    # class for making the 2d fem model act as any other model in methods, for solvers workaround with 2 fem models
    def __init__(self, disc, sol_1, sol_2, pts_1d):
        self.disc = disc
        self.sol_1 = sol_1
        self.sol_2 = sol_2
        self.pts_1d = pts_1d
        pts_2d_line = disc.pts[:len(pts_1d)]
        assert (pts_2d_line[:,0] == pts_1d).all()
        self.name = 'FEM_2'
    def __call__(self, f, u_ex, alpha, callback=None, w_ex=None):
        self.sol_2.set_alpha(self.sol_1.alpha)
        self.fem_model = self.disc.make_model(self.sol_2.f, self.sol_2.u, w_ex=self.sol_2.w)
        def new_callback(t,u):
            u_line = ??#u[:len(self.pts_1d)]
            return callback(t,u_line)
        self.fem_model.solve(self.disc.time_steps, T = self.disc.T, callback = new_callback)
        return self.fem_model.u_fem[:len(self.pts_1d)]

xa,xb,ya,yb = 0,1,0,1
for i in [0,1,2]:
    print(f'sol_index: {i}\n')
    T = 1
    if source == 'reduced_source':
        f,u,w = functions.manufacture_elasticity_solution(d1=3, d2=2, static=static, **functions.ELsols3d[i])
    sol = functions.Solution(T=T, f_raw=f, u_raw=u, zero_source=source=='zero_source', name=f'ELsol{i}',w_raw=w)
    
    model = solvers.Solvers(equation='elasticity', static=static, modelnames=modelnames, p=p,sol=sol, Ne=Ne, time_steps=time_steps,xa=xa, xb=xb, ya=ya,yb=yb,dim=2, NNkwargs=NNkwargs)
    extra_tag = '' # for different names when testing specific stuff
    figname = f'../master/2d_elastic_figures/{source}/{mode}/{"static_"if static else ""}interpol/loss_sol{i}{extra_tag}'
    model_folder = f'../master/saved_models/2d_elastic/{"static_"if static else ""}{source}/{mode}{extra_tag}/'#_explosions/'
    model.plot=False
    model.train(figname=figname, model_folder = model_folder)
    #model.load_weights(model_folder)
    
    legend=False

    # Interpolation
    result_folder = f'../master/saved_results/2d_elastic/{"static_"if static else ""}{source}/{mode}{extra_tag}/interpol/'
    utils.makefolder(result_folder)
    _ = model.test(interpol = True, result_folder=result_folder)
    figname = f'../master/2d_elastic_figures/{source}/{mode}/{"static_"if static else ""}interpol/sol{i}{extra_tag}'
    model.plot_results(result_folder=result_folder, interpol = True, figname=figname, statplot = 5, legend=legend)
    #figname = f'../master/2d_elastic_figures/{source}/{mode}/{"static_"if static else ""}interpol/sol{i}_nonstat{extra_tag}'
    #model.plot_results(result_folder=result_folder, interpol = True, figname=figname, statplot = False)

    # Extrapolation
    result_folder = f'../master/saved_results/2d_elastic/{"static_"if static else ""}{source}/{mode}{extra_tag}/extrapol/'
    utils.makefolder(result_folder)
    _ = model.test(interpol = False, result_folder=result_folder)
    figname = f'../master/2d_elastic_figures/{source}/{mode}/{"static_"if static else ""}extrapol/sol{i}{extra_tag}'
    model.plot_results(result_folder=result_folder, interpol = False, figname=figname, statplot = 5, legend=legend)
    #figname = f'../master/2d_elastic_figures/{source}/{mode}/{"static_"if static else ""}extrapol/sol{i}_nonstat{extra_tag}'
    #model.plot_results(result_folder=result_folder, interpol = False, figname=figname, statplot = False)
