"""
models.py

Written by Sindre Stenen Blakseth, 2021.

Machine learning models for data-driven corrections of 1D heat conduction problems.
"""

########################################################################################################################
# Package imports.

import torch

########################################################################################################################
# File imports.

import config

########################################################################################################################
# Modules.

# Dense layer with activation function and (possibly) dropout.
class DenseLayerWithAct(torch.nn.Module):
    def __init__(self, cfg, input_size, output_size, dropout_probs):
        super(DenseLayerWithAct, self).__init__()
        self.layer = torch.nn.Linear(input_size, output_size)
        self.activation = None
        self.dropout = torch.nn.Dropout(dropout_probs)
        if cfg.act_type == 'lrelu':
            self.activation = torch.nn.LeakyReLU(cfg.act_param)
        else:
            raise Exception("Invalid loss function selection.")

    def forward(self, x):
        return self.dropout(self.activation(self.layer(x)))

# Convolution layer with activation function and (possibly) dropout.
class ConvLayerWithAct(torch.nn.Module):
    def __init__(self, cfg, num_in_ch, num_out_ch, kernel_size):
        super(ConvLayerWithAct, self).__init__()
        padding = kernel_size // 2
        self.layer = torch.nn.Conv1d(
            in_channels  = num_in_ch,
            out_channels = num_out_ch,
            kernel_size  = kernel_size,
            stride       = 1,
            padding      = padding
        )
        self.activation = None
        if cfg.act_type == 'lrelu':
            self.activation = torch.nn.LeakyReLU(cfg.act_param)
        else:
            raise Exception("Invalid loss function selection.")

    def forward(self, x):
        return self.activation(self.layer(x))

# Network of dense layers.
class DenseModule(torch.nn.Module):
    def __init__(self, cfg, num_layers, input_size, output_size, hidden_size = 0, dropout_prob = 0):
        assert num_layers >= 2
        assert num_layers == 2 or hidden_size > 0
        assert input_size > 0 and output_size > 0
        super(DenseModule, self).__init__()

        # Defining input layer.
        first_layer = torch.nn.Linear(input_size, hidden_size)
        first_activation = None
        if cfg.act_type == 'lrelu':
            first_activation = torch.nn.LeakyReLU(cfg.act_param)

        # Defining hidden layers.
        hidden_block = [
            DenseLayerWithAct(cfg, hidden_size, hidden_size, dropout_prob) for hidden_layer in range(num_layers - 2)
        ]

        # Defining output layer.
        last_layer = torch.nn.Linear(hidden_size, output_size)

        # Defining full architecture.
        self.net = torch.nn.Sequential(
            first_layer,
            first_activation,
            *hidden_block,
            last_layer
        ).double()

    def forward(self, x):
        return self.net(x)

# Network of 1D convolution layers, possibly with dense layers at the end.
class ConvolutionModule(torch.nn.Module):
    def __init__(self, cfg, num_conv_layers, output_size, kernel_size, num_filters, num_fc_layers):
        super(ConvolutionModule, self).__init__()
        assert num_conv_layers >  0
        assert num_fc_layers   >  0
        assert output_size     >  0
        assert kernel_size     >= 3
        assert num_filters     >  0

        # Defining input layer.
        padding = kernel_size // 2 - 1
        first_layer = torch.nn.Conv1d(1, num_filters, kernel_size, 1, padding)
        first_activation = None
        if cfg.act_type == 'lrelu':
            first_activation = torch.nn.LeakyReLU(cfg.act_param)

        # Defining hidden conv layers.
        hidden_conv_block = [
            ConvLayerWithAct(cfg, num_filters, num_filters, kernel_size) for hidden_layer in range(num_conv_layers - 1)
        ]

        self.transition_size = output_size * num_filters

        # Defining first dense layer.
        if num_fc_layers > 1:
            first_fc = DenseLayerWithAct(cfg, self.transition_size, output_size, 0.0)
        else:
            first_fc = torch.nn.Linear(self.transition_size, output_size)

        # Defining other dense layers.
        dense_block = [
            DenseLayerWithAct(cfg, output_size, output_size, 0.0) for fc_layer in range(num_fc_layers - 2)
        ]
        if num_fc_layers > 1:
            dense_block.append(torch.nn.Linear(output_size, output_size))

        # Defining full architecture.
        self.conv_net = torch.nn.Sequential(
            first_layer,
            *hidden_conv_block
        ).double()
        self.dense_net = torch.nn.Sequential(
            first_fc,
            *dense_block
        ).double()

    def forward(self, x):
        x1 = self.conv_net(torch.unsqueeze(x, 1))
        x2 = x1.view(-1, self.transition_size)
        return self.dense_net(x2)


# Ensemble of dense networks.
class EnsembleDenseModule(torch.nn.Module):
    def __init__(self, cfg, num_layers, output_size, network_width, dropout_prob):
        super(EnsembleDenseModule, self).__init__()

        # Define one locally-correcting network per output node.
        self.nets = [DenseModule(cfg, num_layers, 3, 1, network_width, dropout_prob) for i in range(output_size)]

        #print("\n\nNUMBER OF NETS:", len(self.nets))

    def forward(self, x):
        # TODO: Is it possible to make this more efficient?
        output = torch.zeros((x.shape[0], x.shape[1] - 2), requires_grad=True).double()
        for i in range(0, len(self.nets)):
            stencil = x[:,i:i+3].clone()
            output[:,i] = torch.squeeze(self.nets[i](stencil), 1)
        return output

class EnsembleWrapper:
    def __init__(self, cfg, module_name, input_size, output_size, model_specific_params):
        if module_name == "DenseModule":
            num_networks = model_specific_params[0]
            depth = model_specific_params[1]
            width = model_specific_params[2]
            self.nets = [
                Model(cfg, "DenseModule", input_size, output_size, [depth, width]
                )
                for i in range(num_networks)
            ]
        elif module_name == "CNNModule":
            num_networks = model_specific_params[0]
            num_conv_layers = model_specific_params[1]
            kernel_size = model_specific_params[2]
            num_filters = model_specific_params[3]
            num_fc_layers = model_specific_params[4]
            self.nets = [
                Model(cfg, "CNNModule", input_size, output_size,
                      [num_conv_layers, kernel_size, num_filters, num_fc_layers]
                )
                for i in range(num_networks)
            ]
        else:
            raise Exception("Incorrect module selection.")

########################################################################################################################
# Full model, consisting of network, loss function, optimizer and information storage facilitation.

class Model:
    def __init__(self, cfg, module_name, input_size, output_size, model_specific_params):
        # Defining network architecture.
        if module_name == 'DenseModule':
            num_layers = model_specific_params[0]
            hidden_size = model_specific_params[1]
            assert num_layers >= 2
            assert num_layers == 2 or hidden_size > 0
            self.net = DenseModule(cfg, num_layers, input_size, output_size, hidden_size, cfg.dropout_prob)
        elif module_name == 'EnsembleDenseModule':
            num_layers = model_specific_params[0]
            network_width = model_specific_params[1]
            self.net = EnsembleDenseModule(cfg, num_layers, output_size, network_width, cfg.dropout_prob)
        elif module_name == "CNNModule":
            num_conv_layers = model_specific_params[0]
            kernel_size = model_specific_params[1]
            num_filters = model_specific_params[2]
            num_fc_layers = model_specific_params[3]
            self.net = ConvolutionModule(cfg, num_conv_layers, output_size, kernel_size, num_filters, num_fc_layers)
        else:
            raise Exception("Invalid model selection.")

        # Defining loss function.
        if cfg.loss_func == 'MSE':
            self.loss = torch.nn.MSELoss(reduction='mean')
        else:
            raise Exception("Invalid loss function selection.")

        # Defining learning parameters.
        if module_name == 'DenseModule' or module_name == "CNNModule":
            params = self.net.parameters()
            num_params = sum(p.numel() for p in self.net.parameters())
        elif module_name == 'EnsembleDenseModule':
            params = []
            num_params = 0
            for i in range(len(self.net.nets)):
                params += list(self.net.nets[i].parameters())
                num_params += sum(p.numel() for p in self.net.nets[i].parameters())
            for param in params:
                param.requires_grad = True
        else:
            raise Exception("Invalid model selection.")
        self.num_params = num_params
        self.learning_rate = cfg.learning_rate
        if cfg.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        else:
            raise Exception("Invalid optimizer selection.")

        # Lists for storing training losses and corresponding training iteration numbers.
        self.train_losses     = []
        self.train_iterations = []
        self.val_losses       = []
        self.val_iterations   = []

########################################################################################################################
# Creating a new model.

def create_new_model(cfg, model_specific_params):
    if cfg.model_name == 'GlobalDense':
        return Model(cfg, 'DenseModule', cfg.N_coarse + 2, cfg.N_coarse, model_specific_params)
    elif cfg.model_name == 'GlobalCNN':
        return Model(cfg, 'CNNModule', cfg.N_coarse + 2, cfg.N_coarse, model_specific_params)
    elif cfg.model_name == 'LocalDense':
        return Model(cfg, 'EnsembleDenseModule', cfg.N_coarse + 2, cfg.N_coarse, model_specific_params)
    elif cfg.model_name == 'EnsembleLocalDense':
        return EnsembleWrapper(cfg, 'DenseModule', 3, 1, model_specific_params)
    elif cfg.model_name == 'EnsembleGlobalCNN':
        return EnsembleWrapper(cfg, 'CNNModule', cfg.N_coarse + 2, 1, model_specific_params)
    else:
        raise Exception("Invalid model selection.")

########################################################################################################################