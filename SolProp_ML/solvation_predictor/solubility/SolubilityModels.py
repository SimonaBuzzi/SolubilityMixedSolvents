import os

from solvation_predictor.inp import TrainArgs, Gsolv, GsolvAqueous, Hsolv, LogSaq, Solute, GsolvHsolv
from solvation_predictor.train.train import load_checkpoint, load_scaler


class SolubilityModels:
    def __init__(
        self,
        reduced_number,
        load_ghsolv,
        load_g,
        load_h,
        load_saq,
        load_solute,
        logger=None,
        verbose=True,
    ):
        """
        Loads the required models for solvation free energy, enthalpy, and aqueous solubility.
            :param reduced_number: if true, only 3 models are considered per property to make predictions faster
            :param load_g: load models for solvation free energy
            :param load_h: load models for solvation enthalpy
            :param load_saq: load models for aqueous solubility
            :param load_solute: load models for solute parameters
            :param logger: logger file
            :param verbose: whether to show logger info or not
        """
        self.ghsolv_models = None
        self.g_models = None
        self.g_aq_models = None
        self.h_models = None
        self.saq_models = None
        self.solute_models = None
        self.logger = logger.info if logger is not None else print

        if load_ghsolv or load_g or load_h or load_saq or load_solute:
            print('reduce_number', reduced_number)
            self.ghsolv_models = (
                self.load_models("SolPropmix1MExp", GsolvHsolv, reduced_number=reduced_number, verbose=verbose)
                if load_ghsolv
                else None
            )
            # self.g_models = (
            #      self.load_models("SolPropmixQM", Gsolv, reduced_number=reduced_number, verbose=verbose)
            #      if load_g
            #      else None #here it was commented out
            #  )
            self.g_aq_models = (
                self.load_models("SolPropmix1MExp", GsolvAqueous, reduced_number=reduced_number, verbose=verbose)
                if load_g
                else None
            )
            # self.h_models = (
            #       self.load_models("SolPropmix1MExp", Hsolv, reduced_number=reduced_number, verbose=verbose) #here
            #       if load_h
            #      else None 
            #  )
            self.saq_models = (
                self.load_models("Aq_sol", LogSaq, reduced_number=reduced_number, verbose=verbose)
                if load_saq
                else None # weights changed #Aq_sol
            )
            self.solute_models = (
                self.load_models("Solutev6", Solute, reduced_number=reduced_number, verbose=verbose) #Abraham parameters. The first argument "Solutev2" refers to the weights of the model!
                if load_solute # This one is Solutev6
                else None
            )
     # debug
    def load_models(self, property_name, inp, reduced_number=False, verbose=True):
        """
        Loads the models for the given property and corresponding input arguments.
            :param property_name:
            :param inp:
            :param reduced_number:
            :param verbose:
        """
        number = 10 if not reduced_number else 3
        
        print(f"Loading solute models with property name: {property_name}")

        paths = [
            os.path.join("trained_models", property_name, "model" + str(i) + ".pt")
            for i in range(number)
        ]
        print(f"Model paths: {paths}")
        print("Checking if model files exist:")
        for p in paths:
            print(f"{p}: {os.path.exists(p)}")
        if verbose:
            self.logger(f"Loading {number} {property_name} models.")

        input_arguments = inp().parse_args()
        scalers = []
        models = []
        for p in paths:
            input_arguments.model_path = p
            scaler = load_scaler(p, from_package=True)
            print(f"Scaler loaded successfully for path: {p}")
            model = load_checkpoint(p, input_arguments, from_package=True)
            print(f"Model loaded successfully for path: {p}")
            scalers.append(scaler)
            models.append(model)
        return input_arguments, scalers, models

