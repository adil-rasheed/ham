
def set_args(mode, dim=1):
    print(f'\nTesting with mode "{mode}"...')
    if mode == 'bugfix':
        Ne = 5 if dim==1 else 4
        time_steps = 20
        #DNNkwargs = {'n_layers_1':1, 'n_layers_2':2,'depth':20, 'bn_depth':4,'lr':5e-3,'patience':[10,20], 'epochs':[100,100], 'min_epochs':[50,50]}
        DNNkwargs = {'n_layers_1':1, 'n_layers_2':2,'depth':20, 'bn_depth':20,'lr':5e-3,'patience':[10,20], 'epochs':[100,100], 'min_epochs':[50,50]}
        pgDNNkwargs = {'n_layers_1':1,'n_layers_2':2,'depth':20,'bn_depth':4,'lr':5e-3,'patience':[10,20], 'epochs':[100,100], 'min_epochs':[50,50]}#, 'l1_penalty':0.01}
        LSTMkwargs = {'lstm_layers':2, 'lstm_depth':20, 'dense_layers':1, 'dense_depth':20, 'lr':5e-3, 'patience':[10,10], 'epochs':[100,100], 'min_epochs':[50,50]}
        pgLSTMkwargs = {'lstm_layers':2, 'lstm_depth':8, 'dense_layers':1, 'bn_depth':10, 'dense_depth':80, 'lr':5e-3, 'patience':[10,10], 'input_period':10}
        NoM = 2
        time_delta = 5
    elif mode == 'quick_test':
        Ne = 20 if dim==1 else 10
        time_steps = 500
        #DNNkwargs = {'n_layers_1':1, 'n_layers_2':2,'depth':80, 'bn_depth':8,'lr':8e-5,'patience':[20,20]}
        DNNkwargs = {'n_layers_1':1, 'n_layers_2':2,'depth':80, 'bn_depth':80,'lr':8e-5,'patience':[20,20]}
        pgDNNkwargs = {'n_layers_1':1,'n_layers_2':2,'depth':80,'bn_depth':8,'lr':8e-5,'patience':[20,20]}
        #LSTMkwargs = {'lstm_layers':4, 'lstm_depth':80, 'dense_layers':2, 'dense_depth':80, 'lr':8e-5, 'patience':[20,20]}
        LSTMkwargs = {'lstm_layers':3, 'lstm_depth':40, 'dense_layers':2, 'dense_depth':80, 'lr':8e-5, 'patience':[20,20], 'input_period':10}
        #pgLSTMkwargs = {'lstm_layers':4, 'lstm_depth':16, 'dense_layers':2, 'dense_depth':80, 'lr':8e-5, 'patience':[20,20], 'input_period':10}
        pgLSTMkwargs = {'lstm_layers':2, 'lstm_depth':16, 'bn_depth':10, 'dense_layers':2, 'dense_depth':80, 'lr':8e-5, 'patience':[20,20], 'input_period':10}
        NoM=5
        time_delta = 0.3 # max 30 steps back
    elif mode == 'full_test':
        Ne = 20 if dim==1 else 15
        time_steps = 5000 if dim==1 else 2500
        #DNNkwargs = {'n_layers_1':1, 'n_layers_2':2,'depth':80, 'bn_depth':8, 'lr':1e-5, 'patience':[20,20]}
        DNNkwargs = {'n_layers_1':1, 'n_layers_2':2,'depth':80, 'bn_depth':80, 'lr':1e-5, 'patience':[20,20]}
        pgDNNkwargs = {'n_layers_1':1, 'n_layers_2':2,'depth':80,'bn_depth':8,'lr':1e-5,'patience':[20,20]}
        LSTMkwargs = {'lstm_layers':4, 'lstm_depth':80, 'dense_layers':2, 'dense_depth':80, 'lr':1e-5, 'patience':[20,20]}
        pgLSTMkwargs = {'lstm_layers':4, 'lstm_depth':16, 'dense_layers':2, 'dense_depth':80, 'lr':1e-5, 'patience':[20,20], 'input_period':10}
        NoM=10
        time_delta = 0
    elif mode == 'el_test':
        Ne = 20 if dim==1 else 15
        time_steps = 5000 if dim==1 else 1000
        #DNNkwargs = {'n_layers_1':1, 'n_layers_2':2,'depth':80, 'bn_depth':8, 'lr':1e-5, 'patience':[20,20]}
        DNNkwargs = {'n_layers_1':1, 'n_layers_2':2,'depth':80, 'bn_depth':80, 'lr':1e-5, 'patience':[20,20]}
        pgDNNkwargs = {'n_layers_1':1, 'n_layers_2':2,'depth':80,'bn_depth':8,'lr':1e-5,'patience':[20,20]}
        LSTMkwargs = {'lstm_layers':4, 'lstm_depth':80, 'dense_layers':2, 'dense_depth':80, 'lr':5e-5, 'patience':[20,20]}
        pgLSTMkwargs = {'lstm_layers':4, 'lstm_depth':16, 'dense_layers':2, 'dense_depth':80, 'lr':4e-5, 'patience':[20,20], 'input_period':10}
        NoM=10
        time_delta = 0

    return Ne, time_steps, DNNkwargs, pgDNNkwargs, LSTMkwargs, pgLSTMkwargs, NoM, time_delta
