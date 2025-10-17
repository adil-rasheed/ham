import numpy as np
import quadrature
import FEM
import solvers
import functions
from matplotlib import pyplot as plt
import sys
import json

mode = 'full_test'
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




def set_args(mode=mode):
    print(f'\nTesting mode "{mode}"...')
    global time_steps, NoM
    if mode == 'bugfix':
        time_steps = 20
        NoM = 2
    elif mode == 'quick_test':
        time_steps = 500
        NoM=3
    elif mode == 'full_test':
        time_steps = 5000
        NoM=2#4

set_args()

if len(sys.argv)>2:
    if sys.argv[2]=='f':
        source = True 
    elif sys.argv[2]=='0':
        source = False
    else:
        print('failed reading source term')
        source = False
else:
    source = False
print('Using exact source' if source else 'Using unknown source (i.e. guessing zero)')

if len(sys.argv)>3:
    p = int(sys.argv[3])
else:
    p=1
print(f'Using p={p}')

modelnames = {
        'DNN' : NoM,
        #'pgDNN' : NoM,
        #'LSTM' : NoM,
        'CoSTA_DNN' : NoM,
        'CoSTA_pgDNN' : NoM,
        #'CoSTA_LSTM' : NoM,
        }

for sol_index in [3,4,1,2]:
    print(f'sol_index: {sol_index}\n')
    #f,u = functions.SBMFACT[sol_index]
    #sol = functions.Solution(T=5, f_raw=f, u_raw=u, zero_source=not source, name=f'{sol_index}', time_delta=time_delta)
    #model = solvers.Solvers(modelnames=modelnames, p=p,sol=sol, Ne=Ne, time_steps=time_steps)
    extra_tag = '' # for different names when testing specific stuff
    figname = f'../preproject/1d_heat_figures/{mode}/{"known_f" if source else "unknown_f"}/interpol/loss_sol{sol_index}_p{p}{extra_tag}.pdf'
    model_folder = f'../preproject/saved_models/{mode}/'
    j=0
    for name in modelnames:
        for i in range(modelnames[name]):
            with open(model_folder+f'{j}_{name}_{time_steps}_{sol_index}.json') as f:
                hist = json.load(f)
                    #'train':model.train_hist, 
                    #'val':model.val_hist
            if i==0:
                epochs_vector = np.arange(1, len(hist['train'])+1)
                plt.plot(epochs_vector, hist['train'], '--', color=solvers.COLORS[name], label=name)
                plt.plot(epochs_vector, hist['val'], color=solvers.COLORS[name])#, label='val_losses')
            else:
                epochs_vector = np.arange(1, len(hist['train'])+1)
                plt.plot(epochs_vector, hist['train'], '--', color=solvers.COLORS[name])
                plt.plot(epochs_vector, hist['val'], color=solvers.COLORS[name])
            j+=1
    plt.yscale('log')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend(title=f'sol={sol_index}')
    plt.grid()
    plt.tight_layout()

    plt.savefig(figname)
    plt.show()
