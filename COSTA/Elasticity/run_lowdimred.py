import numpy as np
import quadrature
import FEM
import solvers
import functions
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt, cm
import sys
import sympy as sp
import utils
import parameters

# for tracking memory usage
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


from pympler import asizeof
def memory_check(object):
    total_val = 0
    print(f'\nExposing object')
    for key, val in sorted(((name, asizeof.asizeof(val)) for name, val in object.__dict__.items()), key=lambda x: -x[1]):
        if val > 200:
            print(sizeof_fmt(val), key)
        total_val += val
    print('total val:', sizeof_fmt(total_val))

def memory_check_global():
    total_val = 0
    globals_stored = set(globals()) - {'sys'}
    print("\nGlobal Variables:")
    for key, val in sorted(((key, asizeof.asizeof(eval(key))) for key in globals_stored), key=lambda x: -x[1]):
        if val > 200:
            print(sizeof_fmt(val), key, type(eval(key)))
        total_val += val
    print('total val:', sizeof_fmt(total_val))


mode = 'bugfix'
if len(sys.argv)>1:
    mode = {'0':'bugfix',
            '1':'quick_test',
            '2':'full_test'}[sys.argv[1]]
else:
    print('No mode specified')
    print('add arg for mode:')
    print('0: bufix (default)')
    print('1: quick_test')
    print('2: full_test')
    print('syntax e.g.: python testing.py 2')

Ne, time_steps, DNNkwargs, pgDNNkwargs, LSTMkwargs, pgLSTMkwargs, NoM, time_delta = parameters.set_args(mode = mode, dim=2)

if len(sys.argv)>2:
    p = int(sys.argv[2])
else:
    p=1
print(f'Using p={p}')

modelnames = {
        'DNN' : NoM,
        #'pgDNN' : NoM,
        #'LSTM' : NoM,
        'CoSTA_DNN' : NoM,
        #'CoSTA_pgDNN' : NoM,
        #'CoSTA_LSTM' : NoM,
        }
NNkwargs = {
        'DNN':DNNkwargs,
        'CoSTA_DNN':DNNkwargs,
        'pgDNN' : pgDNNkwargs,
        'CoSTA_pgDNN':pgDNNkwargs,
        'LSTM':LSTMkwargs,
        'CoSTA_LSTM':LSTMkwargs,
        }

class Fem_method:
    # class for making the 2d fem model act as any other model in methods, for solvers workaround with 2 fem models
    def __init__(self, disc, sol_1, sol_2, pts_1d):
        self.disc = disc
        self.sol_1 = sol_1
        self.sol_2 = sol_2
        self.pts_1d = pts_1d
        self.c = len(pts_1d)
        c=self.c
        pts_2d_line = disc.pts[c*(c-1)//2:c*(c+1)//2]
        print(pts_2d_line, pts_1d)
        assert (pts_2d_line[:,0] == pts_1d).all()
        self.name = 'FEM_2'
    def __call__(self, f, u_ex, alpha, callback=None, w_ex=None):
        self.sol_2.set_alpha(self.sol_1.alpha)
        self.fem_model = self.disc.make_model(self.sol_2.f, self.sol_2.u, w_ex=self.sol_2.w)
        c=self.c
        def new_callback(t,u):
            u_line = u[c*(c-1)//2:c*(c+1)//2]
            return callback(t,u_line)
        self.fem_model.solve(self.disc.time_steps, T = self.disc.T, callback = new_callback)
        return self.fem_model.u_fem[c*(c-1)//2:c*(c+1)//2]

if Ne%2 == 1:
    Ne = 20
    #NoM = 5 #TODO change this back!!!
    assert Ne>3
    print(f'\nWARNING; Ne changed to {Ne}!!\n')
assert Ne%2 == 0 # else u_line definintion in fem is wrong

xa,xb,ya,yb = 0,1,-0.5,0.5
#memory_check_global()
for i in [0,1,4,5]:
    modelnames = {
        'DNN' : NoM,
        #'pgDNN' : NoM,
        #'LSTM' : NoM,
        'CoSTA_DNN' : NoM,
        #'CoSTA_pgDNN' : NoM,
        #'CoSTA_LSTM' : NoM,
        }
    print(f'sol_index: {i}\n')
    f_1,u_1,f_2,u_2 = functions.lowdimred[i]
    sol_1 = functions.Solution(T=1, f_raw=f_1, u_raw=u_1, zero_source=False, name=f'DR_{i}', time_delta=time_delta)
    sol_2 = functions.Solution(T=1, f_raw=f_2, u_raw=u_2, zero_source=False, name=f'DR_{i}', time_delta=time_delta)

    model = solvers.Solvers(modelnames=modelnames, p=p,sol=sol_1, Ne=Ne, time_steps=time_steps,xa=xa, xb=xb, ya=ya,yb=yb,dim=1, skip_create_data=True, NNkwargs=NNkwargs)
    extra_tag = '' # for different names when testing specific stuff
    figfolder = f'../master/bp_heat_figures/lowdr/{mode}/extrapol/'
    utils.makefolder(figfolder)
    figfolder = f'../master/bp_heat_figures/lowdr/{mode}/interpol/'
    utils.makefolder(figfolder)
    figname = f'{figfolder}loss_sol{i}{extra_tag}'
    model_folder = f'../master/saved_models/bp_heat/lowdr/{mode}{extra_tag}/interpol/'
    model.plot=False
    #model.train(figname=figname, model_folder = model_folder)
    #model.load_weights(model_folder)

    #ignore_models = ['pgDNN', 'CoSTA_pgDNN']
    ignore_models = []
    legend=False

    # Add 2d FEM model to models
    disc_2d = FEM.Disc(
                equation=model.disc.equation,
                T=sol_2.T, 
                time_steps=model.disc.time_steps, 
                Ne=Ne,
                p=model.disc.p,
                dim=2,
                xa=xa,
                xb=xb,
                ya=ya,
                yb=yb,
                static=model.disc.static,
                )
    fem_method_2d = Fem_method(disc=disc_2d, sol_1=sol_1, sol_2=sol_2, pts_1d=model.disc.pts)
    model.models.append(fem_method_2d)
    model.modelnames[fem_method_2d.name] = 1

    # Interpolation
    result_folder = f'../master/saved_results/bp_heat/lowdr/{mode}{extra_tag}/interpol/'
    utils.makefolder(result_folder)
    #_ = model.test(interpol = True, result_folder=result_folder)
    figname = f'../master/bp_heat_figures/lowdr/{mode}/interpol/sol{i}{extra_tag}'
    model.plot_results(result_folder=result_folder, interpol = True, figname=figname, statplot = 5, ignore_models = ignore_models, legend=legend)
    #figname = f'../master/bp_heat_figures/{mode}/interpol/sol{x}_nonstat{extra_tag}'
    #model.plot_results(result_folder=result_folder, interpol = True, figname=figname, statplot = False)

    # Extrapolation
    result_folder = f'../master/saved_results/bp_heat/lowdr/{mode}{extra_tag}/extrapol/'
    utils.makefolder(result_folder)
    #_ = model.test(interpol = False, result_folder=result_folder)
    figname = f'../master/bp_heat_figures/lowdr/{mode}/extrapol/sol{i}{extra_tag}'
    model.plot_results(result_folder=result_folder, interpol = False, figname=figname, statplot = 5, ignore_models = ignore_models, legend=legend)
    #figname = f'../master/bp_heat_figures/{mode}/extrapol/sol{x}_nonstat{extra_tag}'
    #model.plot_results(result_folder=result_folder, interpol = False, figname=figname, statplot = False)

    del model

    #memory_check_global()
