import torch
from tap import Tap
import os
import pandas as pd


class CommonArgs(Tap):
    optimization: bool = False
    working_dir = os.getcwd()
    base_dir, _ = os.path.split(working_dir)
    dir: str = base_dir + '/'
    split_ratio: tuple = (0.89, 0.1, 0.01)
    seed: int = 0
    model_path: str = os.path.join(working_dir, 'trained_models') #working directory is until solvation_predictor
    output_name: str = "SolPropMixQMsegExp"
    output_dir = dir + output_name
    make_plots: bool = True
    scale: str = "standard"  # standard or minmax
    scale_features: bool = False
    use_same_scaler_for_features: bool = False
    # random or solute or scaffold or wo_solvents (the latter is based on random split) or kmeans or onecross
    split: str = "random"
    kmeans_split_base: str = "solvent"  # solvent or solute depending on if you want first or second molecule
    save_memory: bool = False

    # for featurization
    property: str = "solvation"  # alternatives are solvation, Tm and logS
    add_hydrogens_to_solvent: bool = False  # adds hydrogens to solvents (first column) if you have 2 input smiles
    mix: bool = False  # features are fractions of the different molecules in the same order
    ####################################################################################################################
    # for active learning
    uncertainty: bool = False  # calculate and output aleotoric uncertainties
    ensemble_variance: bool = False  # calculate and output ensemble variance, epi
    # number or adaptpercent-n for n% of training set
    active_learning_batch_size: str = "adaptpercent-10"
    active_learning_iterations: int = 100
    # how to select data, options are: epistemic, total and random, epi_mol, epi_scaled
    # (for epi unc on scaled predictions)
    data_selection: str = "epistemic"
    restart_al: bool = False
    active_learning_split_ratio = (
        0.3,
        0.4,
        0.3,
    )  # split between initial train data, experimental data and test set
    # for training
    epochs: int = 100 #10
    batch_size: int = 256
    loss_metric: str = "mse"

    # mpn or ffn or none or onlylast or mpn1 or onlylast1 if you have only one molecule
    learning_rates: tuple = (0.001, 0.0001, 0.001)  # initial, final, max
    warm_up_epochs: float = (
        2.0  # you need min 1 with adam optimizer and Noam learning rate scheduler
    )
    lr_scheduler: str = "Noam"  # Noam or Step or Exponential
    # in case of step
    step_size: int = 10
    step_decay: float = 0.2
    # in case of exponential
    exponential_decay: float = 0.1
    minimize_score: bool = True

    cuda: bool = False and (torch.backends.mps.is_available() or torch.cuda.is_available())
    if cuda:
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda')
    else:
        device = torch.device('cpu')
    gpu: int = 4
    # results
    print_weigths: bool = False
    postprocess: bool = False

    # for mpn
    depth: int = 4
    mpn_hidden: int = 200
    mpn_dropout: float = 0.00
    mpn_activation: str = "LeakyReLU"
    mpn_bias: bool = False
    morgan_fingerprint: str = (
        "None"  # None, only_solvent or All #if you want morgan fingerprints
    )
    morgan_bits: int = 16
    morgan_radius: int = 2
    aggregation: str = "mean"
    # make sure your solvent is the first in the input file
    # self.dummy_atom_for_single_atoms = True

    # for attention
    attention: bool = False  # True or false
    att_hidden: int = 200
    att_dropout: float = 0.0
    att_bias: bool = False
    att_activation: str = "ReLU"
    att_normalize: str = "sigmoid"  # sigmoid or softmax or logsigmoid of logsoftmax or None
    att_first_normalize: bool = False

    # for ffn
    ffn_hidden: int = 500
    ffn_num_layers: int = 4
    ffn_dropout: float = 0.00
    ffn_activation: str = "LeakyReLU"
    ffn_bias: bool = True


class TrainArgs(CommonArgs):
    # if the entire dataset is infinite dilute set this to True
    solute = True
    input_file: str = os.path.join(CommonArgs.base_dir, "Data/CombiSolvGH-exp_training_set.csv")
    num_folds: int = 10
    max_num_mols: int = 2 
    num_models: int = 1
    num_targets: int = 2
    f_mol_size: int = 2
    num_features: int = 0
    max_molecules: int = -1  # -1 for all
    pretraining_fix: str = "none"
    pretraining: bool = True # Load the weights of Roel here 
    if pretraining:
        pretraining_path: list = [
            os.path.join(CommonArgs.workdir_path, f"SolPropmixQM/fold_{i}/model0/model0.pt") for i in range(0, 10)
        ]
    # The headers of the input file
    solute_headers: list = ['inchi_solute']
    solvent_headers: list = ['inchi_solvent']
    target_headers: list = ["dGsolv_avg [kcal/mol]", "dHsolv_avg [kcal/mol]"]
    features_headers: list = []
    molefrac_headers: list = [] # molar fraction fuel 1	molar fraction fuel 2	molar fraction fuel 3
    delimiter: str = ","


class PredictArgs(CommonArgs):
    """
    A class were the arguments for prediction by the models are defined.
    """
    # The path to the input file
    input_file: str = os.path.join(CommonArgs.base_dir, 'Data', 'molecules_JC.csv')
    # The path to the directory that contains the trained models
    model_path_root: str = CommonArgs.working_dir + '/trained_models/dHfus'
    if not os.path.exists(model_path_root):
        os.makedirs(model_path_root)
    # List of the names of the trained models in the directory
    model_path = [f for f in os.listdir(model_path_root) if '.pt' in f]
    output_dir = os.path.join(CommonArgs.base_dir, "FusionPredictionsJC")  # The output directory
    get_molecular_embedding = "solute"  # save the embedding for solvents
    # If solute set to True, the first molecule is concatenated in the embedding (needed for infinite dilution)
    # False corresponds to the Fuel paper architecture and True to the mixed solvent Gsolv architecture
    solute = False
    max_num_mols: int = 1  # Maximum number of molecules per data point
    num_targets: int = 1  # Number of targets per data point and number of outputs of the model
    f_mol_size: int = 2
    num_features: int = 0
    max_molecules: int = 100 # Maximum of molecules that are read from the input file, -1 for all
    # The headers of the input file
    solute_headers: list = ["smiles"]
    solvent_headers: list = []
    target_headers: list = ["EnthalpyFusion"]
    features_headers: list = []
    molefrac_headers: list = []
    delimiter: str = ","

    
class GsolvHsolv(CommonArgs):
    pretraining_fix: str = "none"
    input_file: str = CommonArgs.dir + "Data/MultiTaskQM.csv"
    max_num_mols: int = 3
    solute: bool = True
    num_folds: int = 1
    num_targets: int = 2
    f_mol_size: int = 2
    num_features: int = 0


class Gsolv(CommonArgs):
    pretraining_fix: str = "none"
    input_file: str = CommonArgs.dir + "Data/MultiTaskQM.csv"
    max_num_mols: int = 3
    solute: bool = True
    num_folds: int = 1
    num_targets: int = 2
    f_mol_size: int = 2
    num_features: int = 0

class Hsolv(CommonArgs):
    pretraining_fix: str = "none"
    input_file: str = CommonArgs.dir + "Data/MultiTaskQM.csv"
    max_num_mols: int = 3
    solute: bool = True
    num_folds: int = 1
    num_targets: int = 2
    f_mol_size: int = 2
    num_features: int = 0

class GsolvAqueous(CommonArgs):
    pretraining_fix: str = "none"
    input_file: str = CommonArgs.dir + "Data/MultiTaskQM.csv"
    max_num_mols: int = 2
    solute: bool = True
    num_folds: int = 1
    num_targets: int = 2
    f_mol_size: int = 2
    num_features: int = 0


class LogSaq(CommonArgs):
    ## Check thi s one over here 
    ## Upload wieghts in here that might be releant
    pretraining_fix: str = "none"
    input_file: str = CommonArgs.dir + "Data/AqueousSolu.csv"
    solute: bool = False
    max_num_mols: int = 1
    num_folds: int = 1
    num_targets: int = 1
    f_mol_size: int = 2
    num_features: int = 0


class Solute(CommonArgs):
    pretraining_fix: str = "none"
    input_file: str = CommonArgs.dir + "Data/SoluteDB_selected_data.csv"
    num_folds: int = 1
    num_targets: int = 5
    f_mol_size: int = 2
    num_features: int = 0
    solute: bool = False
    max_num_mols: int = 1

