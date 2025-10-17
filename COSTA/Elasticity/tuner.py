import solvers
import functions
import parameters
import FEM
import methods

import datetime
import numpy as np

# Configs for tuning:
debug_mode = False
#modelname = 'LSTM'
modelname = 'CoSTA_LSTM'
#modelname = 'CoSTA_pgDNN'
#modelname = 'CoSTA_pgLSTM'
T = 5
rate = 0.25 # Rate to change parameters
p=1


if debug_mode:
    Ne, time_steps, DNNkwargs, pgDNNkwargs, LSTMkwargs, pgLSTMkwargs, NoM, time_delta = parameters.set_args(mode = 'bugfix', dim=1)
    if modelname=='LSTM':
        nnkwargs = LSTMkwargs
        #nnkwargs['dropout_level'] = 0.2
        #nnkwargs['noise_level'= 1e-3
    elif modelname=='CoSTA_LSTM':
        nnkwargs = LSTMkwargs
        #nnkwargs['dropout_level'] = 0.2
        #nnkwargs['noise_level'= 1e-3
    elif modelname=='CoSTA_pgDNN':
        nnkwargs = pgDNNkwargs
        nnkwargs['l1_penalty']=0.01
        nnkwargs['alpha']=0.01
    elif modelname=='CoSTA_pgLSTM':
        nnkwargs = pgLSTMkwargs
    else:
        print('implement this first')
        quit()
    trainkwargs = {'lr':5e-3, 'patience':[10,10], 'epochs':[100,100], 'min_epochs':[50,50]}
    NoM = 2
    time_delta = 5
else:
    Ne, time_steps, DNNkwargs, pgDNNkwargs, LSTMkwargs, pgLSTMkwargs, NoM, time_delta = parameters.set_args(mode = 'quick_test', dim=1)
    Ne = 20
    time_steps = 500
    if modelname=='LSTM' or modelname=='CoSTA_LSTM':
        nnkwargs = LSTMkwargs
        #nnkwargs['dropout_level'] = 0.2
        #nnkwargs['noise_level'= 1e-3
    elif modelname=='CoSTA_pgDNN':
        nnkwargs = pgDNNkwargs
        nnkwargs['l1_penalty']=0.00004
        nnkwargs['alpha']=0.1
    elif modelname=='CoSTA_pgLSTM':
        nnkwargs = pgLSTMkwargs
    else:
        print('implement this first')
        quit()
    trainkwargs = {'lr':8e-5, 'patience':[20,20], 'epochs':[2000,2000], 'min_epochs':[50,50]}
    NoM = 2
    time_delta = 0.5
# These trainkwargs may be in nnkwargs from parameters.py, we dont want that
nnkwargs.pop('patience', 0)
nnkwargs.pop('epochs', 0)
nnkwargs.pop('min_epochs', 0)
nnkwargs.pop('lr', 0)

Np = Ne+1
pts = np.linspace(0,1,Np)

def output(text):
    print(text)
    with open(f'{modelname}tuner_history{"_db" if debug_mode else ""}.txt', 'a') as f:
        f.write(text)
output(f'\n\n\n{datetime.datetime.now()}\nUsing training parameters: time_steps={time_steps}, {trainkwargs}\n\n')

disc = FEM.Disc(
        equation='heat',
        T=T,
        time_steps=time_steps,
        Ne=Ne,
        p=p,
        dim=1,
        xa=0,
        xb=1,
        static=False,
        )



def get_models(trainkwargs, nnkwargs):
    models=[]
    kwargs = {**trainkwargs, **nnkwargs}
    if modelname=='CoSTA_pgDNN':
        alpha=kwargs.pop('alpha')
        kwargs['activation'] = lambda x: methods.lrelu(x=x, alpha=alpha)
    for i in range(NoM):
        #if modelname == 'DNN':
        #    models.append(solvers.DNN_solver(T=T, pts=pts, time_steps=time_steps, Np=Np, **DNNkwargs))
        if modelname == 'LSTM':
            models.append(methods.LSTM_solver(disc, **kwargs))
        #if modelname == 'CoSTA_DNN':
        #    models.append(solvers.CoSTA_DNN_solver(T=T, Np=Np, pts=pts, time_steps=time_steps, **DNNkwargs))
        elif modelname == 'CoSTA_LSTM':
            models.append(methods.CoSTA_LSTM_solver(disc, **kwargs))
        elif modelname == 'CoSTA_pgDNN':
            models.append(methods.CoSTA_pgDNN_solver(disc, **kwargs))
        elif modelname == 'CoSTA_pgLSTM':
            models.append(methods.CoSTA_pgLSTM_solver(disc, **kwargs))
        else:
            print('implement this first')
            quit()
    return models



# prepare solvers:
f0,u0 = functions.SBMFACT_TUNING[0]
f1,u1 = functions.SBMFACT_TUNING[1]
sol0 = functions.Solution(T=T, f_raw=f0, u_raw=u0, zero_source=True, name=f'T0', time_delta=time_delta)
sol1 = functions.Solution(T=T, f_raw=f1, u_raw=u1, zero_source=True, name=f'T1', time_delta=time_delta)

models0 = get_models(trainkwargs, nnkwargs)
models1 = get_models(trainkwargs, nnkwargs)
s0 = solvers.Solvers(models=models0, sol=sol0, Ne=Ne, time_steps=time_steps)
s1 = solvers.Solvers(models=models1, sol=sol1, Ne=Ne, time_steps=time_steps)
s0.plot = False
s1.plot = False


def get_score(nnkwargs, i, tag=''):
    models0 = get_models(trainkwargs, nnkwargs)
    models1 = get_models(trainkwargs, nnkwargs)
    s0.models = models0
    s1.models = models1
    figname = f'../preproject/1d_heat_figures/tunetraining/{modelname}_{i}_{tag}_sol0.pdf'
    s0.train(figname)
    figname = f'../preproject/1d_heat_figures/tunetraining/{modelname}_{i}_{tag}_sol1.pdf'
    s1.train(figname)
    s0_res = s0.test()[1]
    s1_res = s1.test()[1]
    score = np.mean( [
        *s0_res[0.7][modelname],
        *s1_res[0.7][modelname],
        *s0_res[1.5][modelname],
        *s1_res[1.5][modelname],
        ])
    return score


parameters = [key for key in nnkwargs]
#parameters = ['dropout_level', 'noise_level']
#parameters = ['l1_penalty', 'alpha']
startscore = None

# tune:
for tuning_iteration in range(500):
    text = f'Starting new tuning iteration. Configs now: {nnkwargs}\n'
    output(text)
    if tuning_iteration == 0 and startscore != None:
        score = startscore
    else:
        score = get_score(nnkwargs, tuning_iteration, '')
    text = f'Score to beat: {score}\n'
    output(text)

    np.random.shuffle(parameters) # start with a random parameter
    total_changes = 0
    for parameter in parameters:
        signs = [-1,1]
        if nnkwargs[parameter] == 1: # minimum 1
            signs = [1]
        np.random.shuffle(signs) # Start changing in a random direction
        for sign in signs:
            for changes in range(500):
                if parameter == 'dropout_level':
                    change = sign * 0.05
                    if change+nnkwargs[parameter] < 0:
                        break
                elif parameter in ['noise_level', 'l1_penalty', 'alpha']:
                    change = nnkwargs[parameter] if sign>0 else -nnkwargs[parameter]/2
                else:
                    change = sign * max(1, int(rate*nnkwargs[parameter]))
                    if change+nnkwargs[parameter] <= 0:
                        break

                text = f'Changing {parameter} from {nnkwargs[parameter]} by adding {change}\n'
                output(text)

                new_nnkwargs = nnkwargs.copy()
                new_nnkwargs[parameter] += change
                new_score = get_score(new_nnkwargs, tuning_iteration, f'{parameter}{change}')

                text = f'new score {new_score}\n'
                output(text)
                
                if new_score < score:
                    nnkwargs = new_nnkwargs
                    score=new_score
                else:
                    break
            if changes > 0:
                break
        total_changes += changes
    if total_changes == 0:
        output('\nNO CHANGE!\n')
        #break

output(f'tuner finished(!), after {tuning_iteration} iterations. Final score: {score}, found with params\n{nnkwargs}\n\n')
