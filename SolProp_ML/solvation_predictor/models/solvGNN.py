import torch
import torch.nn as nn

from solvation_predictor.inp import TrainArgs

from solvation_predictor.models.FFN import FFN
from solvation_predictor.models.MPN import MPN
from solvation_predictor.models.MPNNconv import MPNNconv
from logging import Logger
import numpy as np
import dgl



class Model(nn.Module):
    def __init__(self, inp: TrainArgs, logger: Logger = None):
        super(Model, self).__init__()
        logger = logger.debug if logger is not None else print
        self.postprocess = inp.postprocess
        self.feature_size = inp.num_features
        self.property = inp.property
        self.max_num_mols = inp.max_num_mols
        self.inp = inp

        logger(
            f"Make {1} MPN models (no shared weight) with depth {inp.depth}, "
            f"hidden size {inp.mpn_hidden}, dropout {inp.mpn_dropout}, "
            f"activation function {inp.mpn_activation} and bias {inp.mpn_bias}"
        )

        self.mpn = MPN(
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

        self.global_conv = MPNNconv(node_in_feats=132,
                                    edge_in_feats=1,
                                    node_out_feats=132,
                                    edge_hidden_feats=32,
                                    num_step_message_passing=1)

        self.ffn = FFN(
            2*((inp.mpn_hidden + inp.f_mol_size) * 1 + inp.num_features + 1),
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
        :param molefracs: A PyTorch tensor of shape :code:`(num_molecules, max_num_mols)` containing the molefracs
        :return: The output of the Class Model, containing a list of property predictions.
        """
        datapoints = data.get_data()
        molefracs = torch.empty([0], device=self.inp.device)

        graphs = []
        for d in datapoints:
            molfrac = d.get_molfrac_tensor()
            # if molfrac.size(dim=0) != self.max_num_mols:
            molfrac = torch.cat([molfrac, torch.zeros(self.max_num_mols - molfrac.size(dim=0),
                                                           device=self.inp.device)])
            molfrac = molfrac[None, :]
            molefracs = torch.cat([molefracs, molfrac], dim=0)

        mol_encodings, _ = self.mpn(tensors[0])
        molefracs_append = (molefracs[:, 0]).unsqueeze(1)
        mol_encodings = torch.cat([mol_encodings, molefracs_append], dim=-1)
        
        for i in range(1, len(tensors)):
            mol_encoding, atoms_vecs = self.mpn(tensors[i])
            molefracs_append = (molefracs[:, i]).unsqueeze(1)
            mol_encoding = torch.cat([mol_encoding, molefracs_append], dim=-1)
            mol_encodings = torch.dstack([mol_encodings, mol_encoding])

        i = 0
        for d in datapoints:
            g = d.get_dgl_graph()
            g.ndata['h'] = mol_encodings[i, :, 0:len(d.smiles)].transpose(0, 1)
            g.edata['e'] = torch.ones(g.num_edges(), 1, device=self.inp.device)
            i += 1
            graphs.append(g)

        batch = dgl.batch(graphs)
        molecular_features = self.global_conv(batch, batch.ndata['h'], batch.edata['e'])
        num_batched_data = batch.batch_size
        features_solute = molecular_features[0]
        features_solvent_mix = torch.sum(molecular_features[1:datapoints[0].num_mols], dim=0)
        features_cat = torch.cat([features_solute, features_solvent_mix], dim=0)
        feature_index = datapoints[0].num_mols
        for i in range(1, num_batched_data):
            datapoint_length = len(datapoints[i].smiles)
            features_solute = molecular_features[feature_index]

            features_solvent_mix = torch.sum(molecular_features[feature_index+1:datapoint_length], dim=0)
            updated_features = torch.cat([features_solute, features_solvent_mix], dim=0)
            features_cat = torch.vstack([features_cat, updated_features])
            feature_index += datapoint_length

        output = self.ffn(features_cat)
        for i in range(0, len(datapoints)):
            datapoints[i].scaled_predictions = output[i]

        return output










