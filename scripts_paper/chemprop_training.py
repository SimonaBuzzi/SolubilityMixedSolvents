from abc import abstractmethod
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence, TypeAlias

from lightning import pytorch as pl
import numpy as np
import pandas as pd
from rdkit.Chem import Mol
import rdkit.Chem as Chem
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from chemprop.data.collate import BatchMolGraph, collate_batch
from chemprop.data.datapoints import MoleculeDatapoint
from chemprop.data.datasets import Datum, MoleculeDataset, MulticomponentDataset, ReactionDataset
from chemprop.data.molgraph import MolGraph
from chemprop.data.splitting import make_split_indices, split_data_by_indices
from chemprop.featurizers import Featurizer, SimpleMoleculeMolGraphFeaturizer
from chemprop.models import MulticomponentMPNN, multi
from chemprop.nn.agg import Aggregation, MeanAggregation
from chemprop.nn.hparams import HasHParams
from chemprop.nn.message_passing import BondMessagePassing, MessagePassing
from chemprop.nn.metrics import ChempropMetric, MSE, RMSE, MAE, R2Score
from chemprop.nn.predictors import Predictor, RegressionFFN
from chemprop.nn.transforms import ScaleTransform, UnscaleTransform
from chemprop.nn.utils import Activation, get_activation_function
from lightning.pytorch.core.mixins import HyperparametersMixin
from chemprop.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM



# See also chemprop.data.molgraph.MolGraph
class ComponentMolGraph(NamedTuple):
    V: np.ndarray
    E: np.ndarray
    edge_index: np.ndarray
    rev_edge_index: np.ndarray
    w_fp: float = 1.0
    """the weight of the component's fingerprint when combining components in the mixture"""


# See also chemprop.data.datasets.Datum
class ComponentDatum(NamedTuple):
    mg: ComponentMolGraph
    V_d: np.ndarray | None
    x_d: np.ndarray | None
    y: np.ndarray | None
    weight: float
    lt_mask: np.ndarray | None
    gt_mask: np.ndarray | None


# See also chemprop.data.collate.BatchMolGraph
@dataclass(repr=False, eq=False, slots=True)
class BatchComponentMolGraph(BatchMolGraph):
    mgs: InitVar[Sequence[ComponentMolGraph]]
    w_fps: Tensor = field(init=False)
    __is_empty: bool = field(init=False)

    def __post_init__(self, mgs):    
        # super(BatchComponentMolGraph, self).__post_init__(mgs) 
        self._BatchMolGraph__size = len(mgs)
        self.__is_empty = True   
        Vs = []
        Es = []
        edge_indexes = []
        rev_edge_indexes = []
        batch_indexes = []
        w_fps = []

        num_nodes = 0
        num_edges = 0
        for i, mg in enumerate(mgs):
            if mg is None:
                continue
            Vs.append(mg.V)
            Es.append(mg.E)
            edge_indexes.append(mg.edge_index + num_nodes)
            rev_edge_indexes.append(mg.rev_edge_index + num_edges)
            batch_indexes.append([i] * len(mg.V))
            w_fps.append(mg.w_fp)

            num_nodes += mg.V.shape[0]
            num_edges += mg.edge_index.shape[1]

        self.V = torch.from_numpy(np.concatenate(Vs)).float() if len(Vs) > 0 else None
        self.E = torch.from_numpy(np.concatenate(Es)).float() if len(Es) > 0 else None
        self.edge_index = torch.from_numpy(np.hstack(edge_indexes)).long() if len(edge_indexes) > 0 else None
        self.rev_edge_index = torch.from_numpy(np.concatenate(rev_edge_indexes)).long() if len(rev_edge_indexes) > 0 else None
        self.batch = torch.tensor(np.concatenate(batch_indexes)).long() if len(batch_indexes) > 0 else None
        self.w_fps = torch.from_numpy(np.array(w_fps)).float() if len(w_fps) > 0 else None

        if len(Vs) > 0:
            self.__is_empty = False

    def to(self, device: str | torch.device):
        if not self.is_empty():
            super(BatchComponentMolGraph, self).to(device)
            self.w_fps = self.w_fps.to(device)

    def is_empty(self) -> bool:
        """whether any :class:`MolGraph`\s are stored in this batch"""
        return self.__is_empty


# See also chemprop.data.collate.TrainingBatch
class BatchComponentDatum(NamedTuple):
    bmg: BatchComponentMolGraph
    V_d: Tensor | None
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


# See also chemprop.data.collate.MulticomponentTrainingBatch
class MixtureBatch(NamedTuple):
    bmgs: list[BatchMolGraph | BatchComponentMolGraph]
    V_ds: list[Tensor | None]
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


# See also chemprop.data.collate.collate_batch
def collate_component(batch: Iterable[Datum]) -> BatchComponentDatum:
    mgs, V_ds, x_ds, ys, weights, lt_masks, gt_masks = zip(*batch)

    return BatchComponentDatum(
        BatchComponentMolGraph(mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        None if ys[0] is None else torch.from_numpy(np.array(ys)).float(),
        torch.tensor(weights, dtype=torch.float).unsqueeze(1),
        None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)),
        None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)),
    )


# See also chemprop.data.collate.collate_multicomponent
def collate_mixture(batches: Iterable[Iterable[ComponentDatum | Datum]]) -> MixtureBatch:
    tbs = [
        collate_batch(batch) if isinstance(batch[0], Datum) else collate_component(batch)
        for batch in zip(*batches)
    ]

    return MixtureBatch(
        [tb.bmg for tb in tbs],
        [tb.V_d for tb in tbs],
        tbs[0].X_d,
        tbs[0].Y,
        tbs[0].w,
        tbs[0].lt_mask,
        tbs[0].gt_mask,
    )


@dataclass
class ComponentDatapoint(MoleculeDatapoint):
    w_fp: np.ndarray | None = None
    """the weight of the molecule's learned fingerprint when averaging in the mixture"""


@dataclass
class ComponentDataset(MoleculeDataset, Dataset[ComponentMolGraph]):
    data: list[ComponentDatapoint]

    @property
    def w_fps(self) -> np.ndarray:
        return np.array([d.w_fp for d in self.data])

    def __getitem__(self, idx: int) -> ComponentDatum:
        d = self.data[idx]
        if d.mol:
            mg = self.mg_cache[idx] 
            mg = ComponentMolGraph(w_fp=d.w_fp, *mg)
        else:
            mg = None

        return ComponentDatum(
            mg, self.V_ds[idx], self.X_d[idx], self.Y[idx], d.weight, d.lt_mask, d.gt_mask
        )

@dataclass(repr=False, eq=False)
class MixtureDataset(MulticomponentDataset):
    datasets: list[MoleculeDataset | ReactionDataset | ComponentDataset]

    def __getitem__(self, idx: int) -> list[ComponentDatum | Datum]:
        return [dset[idx] for dset in self.datasets]



class MulticomponentMessagePassing(nn.Module, HasHParams):
    """A `MulticomponentMessagePassing` performs message-passing on each individual input in a
    multicomponent input then concatenates the representation of each input to construct a
    global representation

    Parameters
    ----------
    blocks : Sequence[MessagePassing]
        the invidual message-passing blocks for each input
    n_components : int
        the number of components in each input
    shared : bool, default=False
        whether one block will be shared among all components in an input. If not, a separate
        block will be learned for each component.
    """

    def __init__(
        self, 
        blocks: Sequence[MessagePassing], 
        groups: Sequence[Sequence[int]], 
        shared: bool = False,
        ):
        super().__init__()
        self.hparams = {
            "cls": self.__class__,
            "blocks": [block.hparams for block in blocks],
            "groups": groups,
            "shared": shared,
        }

        if len(blocks) == 0:
            raise ValueError("arg 'blocks' was empty!")
        if groups is None:
            raise ValueError("arg 'groups' was empty!")

        if shared:
            if len(blocks) > 1:
                logger.warning(
                    "More than 1 block was supplied but 'shared' was True! Using only the 0th block..."
                )
            if len(groups) != sum(len(g) if isinstance(g, list) else 1 for g in groups):
                logger.warning(
                    "Different groups were supplied but 'shared' was True! Using only the 0th block for all groups..."
                )
        else:
            if len(blocks) != len(groups):
                raise ValueError(
                    "arg 'len(groups)' must be equal to `len(blocks)` if 'shared' is False!"
                    f"got: {len(groups)} and {len(blocks)}, respectively."
                )

        self.groups = groups
        self.shared = shared
        self.blocks = nn.ModuleList()
        if shared:
            self.blocks.extend([blocks[0]] * len(groups))
        else:
            b_idx = 0
            for g_idx, g in enumerate(groups):
                self.blocks.extend([blocks[g_idx]] * len(g))

    def __len__(self) -> int:
        return len(self.blocks)

    @property
    def output_dim(self) -> int:
        d_o = sum(block.output_dim for block in self.blocks)

        return d_o

    def forward(self, bmgs: Iterable[BatchMolGraph], V_ds: Iterable[Tensor | None]) -> list[Tensor]:
        """Encode the multicomponent inputs

        Parameters
        ----------
        bmgs : Iterable[BatchMolGraph]
        V_ds : Iterable[Tensor | None]

        Returns
        -------
        list[Tensor]
            a list of tensors of shape `V x d_i` containing the respective encodings of the `i`\th
            component, where `d_i` is the output dimension of the `i`\th encoder
        """
        if V_ds is None:
            return [block(bmg) if not (bmg.V is None) else None for block, bmg in zip(self.blocks, bmgs)]
        else:
            return [block(bmg, V_d) if not (bmg.V is None) else None for block, bmg, V_d in zip(self.blocks, bmgs, V_ds)]
        

class MixtureMessagePassing(nn.Module, HyperparametersMixin):
    r"""A :class:`MixtureMessagePassing` encodes a batch of mixtures by passing messages along
    molecules constructing a fully connected graph to model intermolecular interactions.

    It implements the following operation:

    .. math::

        h_v^{(0)} &= \tau \left( \mathbf{W}_i(x_v) \right) \\
        m_v^{(t)} &= \sum_{u \in \mathcal{w \in V \setminu v} h_w^{(t-1)} \\
        h_v^{(T)} &= \tau\left(h_v^{(0)} + \mathbf{W}_h m_v^{(t-1)}\right) \\

    where :math:`\tau` is the activation function; :math:`\mathbf{W}_i`, :math:`\mathbf{W}_h` are learned weight matrices; :math:`e_{vw}` is the feature vector of the
    bond between molecules :math:`v` and :math:`w`; :math:`x_v` is the feature vector of molecule :math:`v`;
    :math:`h_v^{(t)}` is the hidden representation of atom :math:`v` at iteration :math:`t`;
    :math:`m_v^{(t)}` is the message received by atom :math:`v` at iteration :math:`t`; and
    :math:`t \in \{1, \dots, T\}` is the number of message passing iterations.
    """

    def __init__(
        self,
        d_v: int = DEFAULT_HIDDEN_DIM,
        d_e: int | None = None,
        d_h: int = DEFAULT_HIDDEN_DIM,
        d_vd: int | None = None,
        bias: bool = False,
        depth: int = 1,
        activation: str | Activation = Activation.RELU,
    ):        
        super().__init__()
        self.save_hyperparameters()
        self.hparams["cls"] = self.__class__

        self.depth = depth
        self.tau = get_activation_function(activation)
        
        self.W_i = nn.Linear(d_v, d_h, bias)
        self.W_h = nn.Linear(d_h, d_h, bias) # TODO consider E
        self.W_o = None
        self.W_d = None

    def initialize(self, V: Tensor) -> Tensor:
        return self.W_i(V)

    def message(self, H: Tensor):
        # assume fully connected graph, TODO
        H = torch.transpose(H, 0, 1) # b x n x d
        M_t = H.unsqueeze(2).expand(-1, -1, H.size(1), -1) # b x n x n x d
        M_t = self.W_h(M_t)
        mask = ~torch.eye(H.size(1), dtype=bool, device=H.device).unsqueeze(0) # exclude self-loops (n x n)
        M_t = (M_t * mask.unsqueeze(-1)).sum(dim=1) # b x n x d
        M_t = torch.transpose(M_t, 0, 1) # n x b x d
        return M_t

    def update(self, M_t: Tensor, H_0: Tensor):
        H_t = self.tau(H_0 + M_t)
        return H_t

    def finalize(self, H_t: Tensor):
        return [h for h in H_t]

    def forward(self, V: list[Tensor]):
        H_0 = self.initialize(torch.stack(V))
        H = self.tau(H_0)
        for _ in range(self.depth):
            M = self.message(H)
            H = self.update(M, H_0)

        return self.finalize(H)



class MixtureAggregation(nn.Module, HasHParams):
    output_dim: int

    def __init__(
        self, 
        graph_agg: Aggregation, 
        groups: Sequence[Sequence[int]], 
        fp_dims: Sequence[int], 
        mixmp: MixtureMessagePassing | None, 
        *args, 
        **kwargs
    ):
        super().__init__()
        self.hparams = {
            "cls": self.__class__,
            "groups": groups,
            "fp_dims": fp_dims,
        }
        self.graph_agg = graph_agg
        self.groups = groups
        self.fp_dims = fp_dims
        self.mixmp = mixmp

    @abstractmethod
    def forward(
        self, Hs: list[Tensor], bmgs: list[BatchComponentMolGraph | BatchMolGraph]
    ) -> Tensor:
        """Aggregate component representations into a mixture representation"""

    def mol_forward(
        self, H_vs: list[Tensor], bmgs: list[BatchComponentMolGraph | BatchMolGraph]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        # Hs: n x b x d (but not each mixture has n components, so n x b can be incomplete, hence we need to synthetically adapt the sizes)
        Hs, w_fps, Hs_batch = zip(*[(self.graph_agg(H_v, torch.unique(bmg.batch, return_inverse=True)[1]), bmg.w_fps, torch.unique(bmg.batch)) if (isinstance(bmg, BatchComponentMolGraph) and (bmg.batch is not None)) else (self.graph_agg(H_v, torch.unique(bmg.batch, return_inverse=True)[1]), None, torch.unique(bmg.batch)) if (bmg.batch is not None) else (None, None, None) for H_v, bmg in zip(H_vs, bmgs)])
        Hs, w_fps, Hs_batch =  list(Hs), list(w_fps), list(Hs_batch)
        return Hs, w_fps, Hs_batch

    def complete_sparse_batch(
        self, Hs: list[Tensor], w_fps: list[Tensor], Hs_batch: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        # make Hs and w_fps the same size wrt n x b by adding zero-values/tensors
        Hs = self._complete_sparse_tensorlist(Hs, Hs_batch, self.fp_dims)
        if not all(f is None for f in w_fps):
            w_fps = self._complete_sparse_tensorlist(w_fps, Hs_batch, [None for _ in range(len(Hs_batch))])
        return Hs, w_fps, Hs_batch

    def _complete_sparse_tensorlist(
        self, Hs: list[Tensor], Hs_batch: list[Tensor], dim: list[int]
    ) -> list[Tensor]:
        #dim = max(H_b.a)
        batch_size = max(H_b.max().item() for H_b in Hs_batch if not (H_b is None))
        device = [H.device for H in Hs if not (H is None)][0] # workaround, TODO
        compl_Hs = []
        for n_idx, (n_H, n_H_batch) in enumerate(zip(Hs, Hs_batch)):
            if dim[n_idx]:
                compl_H = torch.zeros((batch_size+1, dim[n_idx]), dtype=torch.float32, device=device)
            else:
                compl_H = torch.zeros((batch_size+1), dtype=torch.float32, device=device)
            if (n_H is not None) and (n_H_batch is not None):
                compl_H[n_H_batch] = n_H
            compl_Hs.append(compl_H)
        return compl_Hs

class WeightedSumAggregation(MixtureAggregation):
    @property
    def output_dim(self) -> int:
        return sum(self.fp_dims[group[0]] for group in self.groups)

    def forward(
        self, H_vs: list[Tensor], bmgs: list[BatchComponentMolGraph | BatchMolGraph]
    ) -> Tensor:
        Hs, w_fps, Hs_batch = self.mol_forward(H_vs, bmgs)
        Hs, w_fps, Hs_batch = self.complete_sparse_batch(Hs, w_fps, Hs_batch)

        if not (self.mixmp is None):
            Hs = self.mixmp(Hs)

        combined_Hs = []
        for group in self.groups:
            if len(group) == 1:
                combined_Hs.append(Hs[group[0]])
                continue
            group_Hs = torch.stack([Hs[idx] for idx in group])  # n x b x d
            #import pdb
            #pdb.set_trace()
            group_w_fps = torch.stack([w_fps[idx] for idx in group])  # n x b
            #pdb.set_trace()
            # n: num. components in group, b: num. comp. in batch, d: output dim of message passing
            combined_H = torch.einsum("nb,nbd->bd", group_w_fps, group_Hs)
            combined_Hs.append(combined_H)
        #pdb.set_trace()
        return torch.cat(combined_Hs, 1)

class ConcatAggregation(MixtureAggregation):
    @property
    def components_in_mixture(self) -> set[int]:
        return {idx for group in self.groups if len(group) > 1 for idx in group}

    @property
    def output_dim(self) -> int:
        return sum(self.fp_dims) + len(self.components_in_mixture)

    def forward(
        self, H_vs: list[Tensor], bmgs: list[BatchComponentMolGraph | BatchMolGraph]
    ) -> Tensor:
        Hs, w_fps, Hs_batch = self.mol_forward(H_vs, bmgs)
        Hs, w_fps, Hs_batch = self.complete_sparse_batch(Hs, w_fps, Hs_batch)

        if not (self.mixmp is None):
            Hs = self.mixmp(Hs)

        w_fps = torch.stack([w_fps[idx] for idx in self.components_in_mixture], dim=1)
        return torch.cat(Hs + [w_fps], 1)

class DeepsetsAggregation(MixtureAggregation):
    r"""Deep sets aggregation of the graph-level representation:

    .. math::
        \mathbf h = \mathrm{MLP_{g}}(\sum_{c \in C} \mathrm{MLP_{l}}(\mathbf h_c))
    """
    def __init__(
        self, graph_agg: Aggregation, groups: Sequence[Sequence[int]], fp_dims: Sequence[int], 
        mixmp: MixtureMessagePassing | None, *args, **kwargs
    ):
        super().__init__(graph_agg, groups, fp_dims, mixmp, *args, **kwargs)
        
        self.MLPs_local = nn.ModuleList([])
        self.MLPs_global = nn.ModuleList([])
        for group in groups:
            # TODO: allow to set hparams for MLP by kwargs (e.g., hidden_dim, n_layers)
            hidden_dim = self.fp_dims[group[0]]
            self.MLPs_local.append(
                nn.Sequential(
                nn.Linear(self.fp_dims[group[0]], hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.fp_dims[group[0]], bias=False),
                )
            )
            self.MLPs_global.append(
                nn.Sequential(
                nn.Linear(self.fp_dims[group[0]], hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.fp_dims[group[0]], bias=False),
                )
            )

    @property
    def output_dim(self) -> int:
        return sum(self.fp_dims[group[0]] for group in self.groups)

    def forward(
        self, H_vs: list[Tensor], bmgs: list[BatchComponentMolGraph | BatchMolGraph]
    ) -> Tensor:
        Hs, w_fps, Hs_batch = self.mol_forward(H_vs, bmgs)
        Hs, w_fps, Hs_batch = self.complete_sparse_batch(Hs, w_fps, Hs_batch)
        
        if not (self.mixmp is None):
            Hs = self.mixmp(Hs)

        combined_Hs = []
        for g_idx, group in enumerate(self.groups):
            # use only global MLP if group only has one component, as local MLP would just be nested into global MLP
            if len(group) == 1:
                combined_Hs.append(self.MLPs_global[g_idx](Hs[group[0]]))
                continue
            group_w_Hs = torch.stack([self.MLPs_local[g_idx](w_fps[idx].unsqueeze(1) * Hs[idx]) for idx in group])  # n x b x d
            combined_H = torch.sum(group_w_Hs, dim=0)
            combined_Hs.append(self.MLPs_global[g_idx](combined_H))
        return torch.cat(combined_Hs, 1)

class AttentiveAggregation(MixtureAggregation):
    r"""Attentive aggregation of the graph-level representation:

    .. math::
        \mathbf h = \sum_{c \in C} \alpha_c \mathbf h_c

        \alpha_c = \mathrm{softmax}(\mathbf h_c)
    """
    def __init__(
        self, graph_agg: Aggregation, groups: Sequence[Sequence[int]], fp_dims: Sequence[int], 
        mixmp: MixtureMessagePassing | None, *args, **kwargs
    ):
        super().__init__(graph_agg, groups, fp_dims, mixmp, *args, **kwargs)
        
        self.Ws_a = nn.ModuleList([
            nn.Linear(self.fp_dims[group[0]], 1, bias=False) for group in groups
            ])

    @property
    def output_dim(self) -> int:
        return sum(self.fp_dims[group[0]] for group in self.groups)

    def forward(
        self, H_vs: list[Tensor], bmgs: list[BatchComponentMolGraph | BatchMolGraph]
    ) -> Tensor:
        Hs, w_fps, Hs_batch = self.mol_forward(H_vs, bmgs)
        Hs, w_fps, Hs_batch = self.complete_sparse_batch(Hs, w_fps, Hs_batch)

        if not (self.mixmp is None):
            Hs = self.mixmp(Hs)

        combined_Hs = []
        for g_idx, group in enumerate(self.groups):
            if len(group) == 1:
                combined_Hs.append((Hs[group[0]]))
                continue
            w_Hs = torch.stack([w_fps[idx].unsqueeze(1) * Hs[idx] for idx in group])  # n x b x d
            attention_logits = self.Ws_a[g_idx](w_Hs).exp().squeeze(2) # n x b
            # Ignore logits that correspond to completed zero-tensors due to missing components
            # Create a mask tensor
            mask = torch.zeros_like(attention_logits, dtype=torch.bool)
            # Fill the mask tensor based on index 
            for tmp_i, idx in enumerate(group):
                indices = Hs_batch[idx]
                if indices is not None:
                    mask[tmp_i, indices] = True
            # Apply the mask to the original tensor
            attention_logits = attention_logits * mask
            #attention_logits = torch.stack(self.complete_sparse_batch([a.unsqueeze(0) for a in attention_logits], Hs_batch, [None for _ in attention_logits]))
            Z = torch.sum(attention_logits, dim=0, keepdim=True)
            alphas = attention_logits / Z
            combined_H = torch.sum(alphas.unsqueeze(-1) * w_Hs, dim=0)
            #import pdb
            combined_Hs.append(combined_H)
        return torch.cat(combined_Hs, 1)

class Set2SetAggregation(MixtureAggregation):
    r"""Set2Set aggregation of the graph-level representation:

    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

        \alpha_{c,t} &= \mathrm{softmax}(\mathbf{h}_c \cdot \mathbf{q}_t)

        \mathbf{r}_t &= \sum_{c=1}^C \alpha_{c,t} \mathbf{h}_c

        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

    where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
    the dimensionality as the input.
    
    Note: This implementation follows PyTorch Geometric (cf. https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/aggr/set2set.html#Set2Set) and is based on `"Order Matters: Sequence to sequence for
    Sets" <https://arxiv.org/abs/1511.06391>`_ paper.
    """
    def __init__(
        self, graph_agg: Aggregation, groups: Sequence[Sequence[int]], fp_dims: Sequence[int], 
        mixmp: MixtureMessagePassing | None, *args, **kwargs
    ):
        super().__init__(graph_agg, groups, fp_dims, mixmp, *args, **kwargs)
        
        # TODO: allow to set hparams for Set2Set by kwargs (e.g., processing steps)
        self.processing_steps = 3
        self.lstms = nn.ModuleList([])
        for group in groups:
            in_channels = self.fp_dims[group[0]]
            out_channels = self.fp_dims[group[0]] * 2
            self.lstms.append(
                torch.nn.LSTM(out_channels, in_channels, **kwargs)
                )

    @property
    def output_dim(self) -> int:
        return sum(self.fp_dims[group[0]] * 2 if len(group) > 1 else self.fp_dims[group[0]] for group in self.groups)

    def forward(
        self, H_vs: list[Tensor], bmgs: list[BatchComponentMolGraph | BatchMolGraph]
    ) -> Tensor:
        Hs, w_fps, Hs_batch = self.mol_forward(H_vs, bmgs)
        Hs, w_fps, Hs_batch = self.complete_sparse_batch(Hs, w_fps, Hs_batch)

        if not (self.mixmp is None):
            Hs = self.mixmp(Hs)
            
        combined_Hs = []
        for g_idx, group in enumerate(self.groups):
            if len(group) == 1:
                combined_Hs.append((Hs[group[0]]))
                continue
            
            w_Hs = torch.stack([w_fps[idx].unsqueeze(1) * Hs[idx] for idx in group]) 
            w_Hs = torch.transpose(w_Hs, 0, 1) # b x n x d
            b_dim = w_Hs.size(0)
            d_dim = w_Hs.size(-1)

            h = (w_Hs.new_zeros((self.lstms[g_idx].num_layers, b_dim, d_dim)),
                w_Hs.new_zeros((self.lstms[g_idx].num_layers, b_dim, d_dim)))
            q_star = w_Hs.new_zeros(b_dim, d_dim * 2)

            for _ in range(self.processing_steps):
                q, h = self.lstms[g_idx](q_star.unsqueeze(0), h)

                q = q.squeeze(0) # b x d
                e = torch.sum(w_Hs * q.unsqueeze(1), dim=2) # b x n
                
                #a = torch.softmax(e, dim=1) # b x n, old version

                attention_logits = e.exp() #.squeeze(2) # b x n

                # Ignore logits that correspond to completed zero-tensors due to missing components
                # Create a mask tensor
                mask = torch.zeros_like(e, dtype=torch.bool)
                # Fill the mask tensor based on index tensors
                for tmp_i, idx in enumerate(group):
                    indices = Hs_batch[idx]
                    if indices is not None:
                        mask[indices, tmp_i] = True
                # Apply the mask to the original tensor
                attention_logits = attention_logits * mask
                Z = torch.sum(attention_logits, dim=1, keepdim=True)
                alphas = attention_logits / Z
                r = torch.sum(w_Hs * alphas.unsqueeze(2), dim=1) # b x d
                q_star = torch.cat([q, r], dim=1) # b x 2*d

            combined_Hs.append(q_star)
        return torch.cat(combined_Hs, 1)

class MixtureMPNN(MulticomponentMPNN):
    def __init__(
        self,
        message_passing: MulticomponentMessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        mix_mpn: MixtureMessagePassing | None = None,
        batch_norm: bool = False,
        metrics: Iterable[ChempropMetric] | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        X_d_transform: ScaleTransform | None = None,
    ):
        super().__init__(
            message_passing,
            agg,
            predictor,
            batch_norm,
            metrics,
            warmup_epochs,
            init_lr,
            max_lr,
            final_lr,
            X_d_transform,
        )
        self.agg: MixtureAggregation

    def fingerprint(
        self,
        bmgs: Iterable[BatchComponentMolGraph | BatchMolGraph],
        V_ds: Iterable[Tensor],
        X_d: Tensor | None = None,
    ) -> Tensor:
        H_vs: list[Tensor] = self.message_passing(bmgs, V_ds)
        H = self.agg(H_vs, bmgs)
        H = self.bn(H)
        return H if X_d is None else torch.cat((H, X_d), 1)

    @classmethod
    def _load(cls, path, map_location, **submodules):
        d = torch.load(path, map_location, weights_only=False)

        try:
            hparams = d["hyper_parameters"]
            state_dict = d["state_dict"]
        except KeyError:
            raise KeyError(f"Could not find hyper parameters and/or state dict in {path}.")

        hparams["message_passing"]["blocks"] = [
            block_hparams.pop("cls")(**block_hparams)
            for block_hparams in hparams["message_passing"]["blocks"]
        ]
        graph_agg_hparams = hparams["agg"]["graph_agg"]
        hparams["agg"]["graph_agg"] = graph_agg_hparams.pop("cls")(**graph_agg_hparams)
        mixmp_hparams = hparams["agg"]["mixmp"]
        hparams["agg"]["mixmp"] = mixmp_hparams.pop("cls")(**mixmp_hparams)
        submodules |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in ("message_passing", "agg", "predictor")
            if key not in submodules
        }

        if not hasattr(submodules["predictor"].criterion, "_defaults"):
            submodules["predictor"].criterion = submodules["predictor"].criterion.__class__(
                task_weights=submodules["predictor"].criterion.task_weights
            )

        return submodules, state_dict, hparams
    
# Example code for training a MixtureMPNN on a multicomponent dataset
from chemprop import nn as nnn
# training code 
chemprop_dir = Path.cwd().parent
input_path = "your_input_path.csv"
df_input_sample = pd.read_csv(input_path, low_memory=False)
smiles_columns = ["solute_smiles", "solvent_smiles", "solvent2_smiles"]
frac_columns = ["molefrac"]
target_columns = ["logS"]
smiss = df_input_sample.loc[:, smiles_columns].values
print(smiss)
fracs = df_input_sample.loc[:, frac_columns].copy()
fracs["molefrac"] = fracs["molefrac"].fillna(1.0)
fracs = fracs.values
ys = df_input_sample.loc[:, target_columns].values
extra_datapoint_descriptors = df_input_sample[['molefrac','temperature']].values
all_data = [[MoleculeDatapoint.from_smi(smis[0], y, x_d=X_d) for smis, y, X_d in zip(smiss, ys, extra_datapoint_descriptors)]]
all_data += [[ComponentDatapoint.from_smi(smis[1], w_fp=f[0]) for smis, f in zip(smiss, fracs)]]
all_data += [[ComponentDatapoint.from_smi(smis[2], w_fp=1-f[0]) if smis[2] else ComponentDatapoint(None) for smis, f in zip(smiss, fracs)]]
component_to_split_by = 0
mols = [d.mol for d in all_data[component_to_split_by]]
train_indices, val_indices, test_indices = make_split_indices(mols, "random", (0.85, 0.1, 0.05))
print(len(mols))
train_data, val_data, test_data = split_data_by_indices(all_data, train_indices, val_indices, test_indices)
train_data = train_data[0]
val_data = val_data[0]
test_data = test_data[0]
train_datasets = [
    MoleculeDataset(train_data[0]),
    ComponentDataset(train_data[1]),
    ComponentDataset(train_data[2]),
]
val_datasets = [
    MoleculeDataset(val_data[0]),
    ComponentDataset(val_data[1]),
    ComponentDataset(val_data[2]),
]
test_datasets = [
    MoleculeDataset(test_data[0]),
    ComponentDataset(test_data[1]),
    ComponentDataset(test_data[2]),
]
train_mcdset = MixtureDataset(train_datasets)
scaler = train_mcdset.normalize_targets()
val_mcdset = MixtureDataset(val_datasets)
val_mcdset.normalize_targets(scaler)
test_mcdset = MixtureDataset(test_datasets)
test_mcdset.normalize_targets(scaler)
extra_datapoint_descriptors_scaler = train_datasets[0].normalize_inputs("X_d")
val_datasets[0].normalize_inputs("X_d", extra_datapoint_descriptors_scaler)
test_datasets[0].normalize_inputs("X_d", extra_datapoint_descriptors_scaler)
train_loader = DataLoader(train_mcdset, batch_size=64, shuffle=True, collate_fn=collate_mixture)
val_loader = DataLoader(val_mcdset, batch_size=64, shuffle=False, collate_fn=collate_mixture)
test_loader = DataLoader(test_mcdset, batch_size=64, shuffle=False, collate_fn=collate_mixture)

mcmp = MulticomponentMessagePassing(
    blocks=[BondMessagePassing(), BondMessagePassing()],
    groups=[[0], [1, 2]],
    shared=False
)
mixmp = MixtureMessagePassing(
    d_v=mcmp.blocks[0].output_dim,
    d_h=mcmp.blocks[0].output_dim,
    depth=1
)
graph_agg = MeanAggregation()
mixagg = WeightedSumAggregation(
    graph_agg=graph_agg,
    groups=[[0], [1, 2]],
    fp_dims=[mcmp.blocks[0].output_dim] * 3,
    mixmp=mixmp,
)
output_transform = UnscaleTransform.from_standard_scaler(scaler)
ffn = RegressionFFN(
    input_dim=mixagg.output_dim + extra_datapoint_descriptors.shape[1],
    output_transform=output_transform
)
mcmpnn = MixtureMPNN(mcmp, mixagg, ffn)
trainer = pl.Trainer(
    logger=True,
    enable_checkpointing=True,
    enable_progress_bar=True,
    accelerator="auto",
    devices=1,
    max_epochs=100,
)
trainer.fit(mcmpnn, train_loader, val_loader)
model_dict = {
    "hyper_parameters": mcmpnn.hparams,
    "state_dict": mcmpnn.state_dict(),
    "target_scaler": scaler,
    "X_d_scaler": extra_datapoint_descriptors_scaler,
}
torch.save(model_dict, 'model_weight.pt')
print("Model saved.")
