import torch
import torch.nn as nn
from solvation_predictor.inp import TrainArgs

from solvation_predictor.models.FFN import FFN
from solvation_predictor.models.MPN import MPN
from logging import Logger


class Model(nn.Module):
    """
    A class object that is a model which contains a message passing network following by feed-forward layers.
    """
    def __init__(self, inp: TrainArgs, logger: Logger = None):
        super(Model, self).__init__()
        logger = logger.debug if logger is not None else print
        self.postprocess = inp.postprocess
        self.feature_size = inp.num_features
        self.property = inp.property
        self.inp = inp
        self.max_num_mols = inp.max_num_mols

        logger(
            f"Make {1} MPN models (no shared weight) with depth {inp.depth}, "
            f"hidden size {inp.mpn_hidden}, dropout {inp.mpn_dropout}, "
            f"activation function {inp.mpn_activation} and bias {inp.mpn_bias}"
        )
        self.mpn_1 = MPN(
            depth=inp.depth,
            hidden_size=inp.mpn_hidden,
            dropout=inp.mpn_dropout,
            activation=inp.mpn_activation,
            bias=inp.mpn_bias,
            cuda=inp.cuda,
            atomMessage=False,
            property=self.property,
            aggregation=inp.aggregation,
        )

        self.mpn_2 = MPN(
            depth=inp.depth,
            hidden_size=inp.mpn_hidden,
            dropout=inp.mpn_dropout,
            activation=inp.mpn_activation,
            bias=inp.mpn_bias,
            cuda=inp.cuda,
            atomMessage=False,
            property=self.property,
            aggregation=inp.aggregation,
        )

        logger(
            f"Make FFN model with number of layers {inp.ffn_num_layers}, hidden size {inp.ffn_hidden}, "
            f"dropout {inp.ffn_dropout}, activation function {inp.ffn_activation} and bias {inp.ffn_bias}"
        )
        ffn_input_width = {True: 2, False: 1}
        self.ffn = FFN(
            (inp.mpn_hidden + inp.f_mol_size) * ffn_input_width[inp.solute] + inp.num_features + 204,
            inp.num_targets,
            ffn_hidden_size=inp.ffn_hidden,
            num_layers=inp.ffn_num_layers,
            dropout=inp.ffn_dropout,
            activation=inp.ffn_activation,
            bias=inp.ffn_bias,
        )

    def forward(self, data, tensors):
        """
        Runs the Model Class on input.

        :param data: Parameter containing the data on which the model needs to be run.
        :param molefracs: ??
        :return: The output of the Class Model, containing a list of property predictions.
        """
        datapoints = data.get_data()
        molefracs = torch.empty([0], device=self.inp.device)

        forward_loop_dict = {True: self.solute_forward, False: self.mixture_forward}
        input = forward_loop_dict[self.inp.solute](datapoints, tensors, molefracs)

        # if self.feature_size > 0:
        #     features = data.get_scaled_features()
        #     features = torch.FloatTensor(features)
        #     if self.cudap or next(self.parameters()).is_cuda:
        #         features = features.cuda()
        #     input = torch.cat([input, features], dim=1)
        #
        # if self.postprocess:
        #     for i in range(0, len(datapoints)):
        #         datapoints[i].updated_mol_vecs = mol_encodings[i]
        #
        #     for i in range(0, len(datapoints)):
        #         datapoints[i].updated_atom_vecs = atoms_vecs[i]

        output = self.ffn(input)
        for i in range(0, len(datapoints)):
            datapoints[i].scaled_predictions = output[i]
        return output

    def mixture_forward(self, datapoints, tensors, molefracs):
        raise NotImplementedError

    def solute_forward(self, datapoints, tensors, molefracs):
        mol_encodings, atoms_vecs = self.mpn_1(tensors[0])
        mol_encodings = mol_encodings[None, :]

        for i in range(1, len(tensors)):
            mol_encoding, atoms_vecs = self.mpn_2(tensors[i])
            mol_encoding = mol_encoding[None, :]
            mol_encodings = torch.vstack([mol_encodings, mol_encoding])

        for d in datapoints:
            molfrac = d.get_molfrac_tensor()
            molfrac = molfrac[None, :]
            molefracs = torch.cat([molefracs, molfrac], dim=0)
        update_vecs = torch.empty([0], device=self.inp.device)
        for i in range(1, self.max_num_mols):
            molefrac_col = molefracs[:, i - 1]
            molefrac_col = molefrac_col.unsqueeze(1)
            binary_col = torch.gt(molefrac_col, 0.0).float()
            mol_encodings[i] = mol_encodings[i] * binary_col
            update_vec = torch.cat([mol_encodings[i], molefrac_col], dim=1)
            update_vecs = torch.cat([update_vecs, update_vec], dim=1)
        vec = update_vecs
        input = torch.cat([mol_encodings[0], vec], dim=1)
        return input