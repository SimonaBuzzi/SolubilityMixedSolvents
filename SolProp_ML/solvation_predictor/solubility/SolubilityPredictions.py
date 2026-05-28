import numpy as np
import pandas as pd
from solvation_predictor.data.data import DatapointList, DataPoint, MolencoderDatabase
from solvation_predictor.solubility.SolubilityData import SolubilityData
from solvation_predictor.solubility.SolubilityModels import SolubilityModels
from solvation_predictor.train.evaluate import predict
from rdkit import Chem
from collections import OrderedDict


class SolubilityPredictions:
    def __init__(
        self,
        predict_aqueous = None, 
        predict_reference_solvents = None, 
        predict_t_dep = None, 
        predict_solute_parameters = True, 
        data: SolubilityData = None,
        models: SolubilityModels = None,
        logger=None,
        verbose=True,):
   
        """
        Make the machine learning model predictions
            :param data: data of the type SolubilityData
            :param models: models of the type SolubilityModels
            :param predict_aqueous: if you want to calculate solubility using the model predicted aqueous solubility
            :param predict_reference_solvents: if you want to calculate solubility using reference solvents
            :param predict_t_dep: if you want to calculate temperature dependent solubility
            :param predict_solute_parameters: if you want to predict solute parameters
            :param logger: logger file
            :param verbose: whether to show logger info or not
        """
        self.data = data if data is not None else None
        self.models = models if models is not None else None
        self.logger = logger.info if logger is not None else print

        self.ghsolv = None
        self.gsolv = None
        self.hsolv = None
        self.saq = None
        self.gsolv_aq = None
        self.gsolv_ref = None
        self.solute_parameters = None

        if self.data is not None and self.models is not None:
            self.make_all_model_predictions(
                predict_aqueous=predict_aqueous,
                predict_reference_solvents=predict_reference_solvents,
                predict_t_dep=predict_t_dep,
                predict_solute_parameters=predict_solute_parameters,
                verbose=verbose,
            )

    def set_data(self, data: SolubilityData):
        self.data = data

    def set_models(self, models: SolubilityModels):
        self.models = models

    def make_all_model_predictions(
        self,
        predict_aqueous: bool = True,
        predict_reference_solvents: bool = False,
        predict_t_dep: bool = True, #it was set it to False
        predict_solute_parameters: bool = True, #set it to false
        verbose=True,
    ):
        self.ghsolv = (
            self.make_property_predictions("GHsolvQM", self.models.ghsolv_models, verbose=verbose)
            if self.models.ghsolv_models is not None
            else None
        )
        # self.gsolv = (
        #     self.make_property_predictions("Gsolv", self.models.g_models, verbose=verbose)
        #      if self.models.g_models is not None
        #      else None
        #  )
        
        self.hsolv = (
            self.make_property_predictions("Hsolv", self.models.h_models, verbose=verbose)
            if self.models.h_models is not None
             else None
         )
        self.saq = (
            self.make_saq_predictions(verbose=verbose)
            if self.models.saq_models is not None
            else None
        )

        self.gsolv_aq = (
            self.make_gsolvaq_predictions(verbose=verbose)
            if predict_aqueous is not None
            else None
        )
        self.gsolv_ref = (
            self.make_gsolvref_predictions(verbose=verbose)
            if predict_reference_solvents and self.models.g_models is not None
            else None
        )

        self.solute_parameters = (
            self.make_soluteparameter_predictions( verbose=verbose)
            if predict_t_dep or predict_solute_parameters is not None
            else None
        )
    

    def make_property_predictions(self, property_name, models, verbose=False):
        if verbose:
            self.logger(f"Make {property_name} predictions")
        if models is None:
            raise ValueError(f"{property_name} models are not loaded, cannot make predictions.")
        if self.data.smiles_solvents2 is None:
            smiles = list(zip(self.data.smiles_solutes, self.data.smiles_solvents1))
        else:
            smiles = list(zip(self.data.smiles_solutes, self.data.smiles_solvents1, self.data.smiles_solvents2))
            smiles = list(zip(self.data.smiles_solutes, self.data.smiles_solvents1, self.data.smiles_solvents2, self.data.molefracs))
        results = self.make_predictions(smiles, models, self.data.molefracs)
        mean_predictions = [results[sm][0] for sm in smiles]
        variance_predictions = [results[sm][1] for sm in smiles]
        return mean_predictions, variance_predictions

    def make_gsolvaq_predictions(self, verbose=False):
        if verbose:
            self.logger("Make Gsolv aqueous predictions")
        if self.models.g_aq_models is None:
            raise ValueError("Gsolv models are not loaded, cannot make predictions")
        aq_smiles_pairs = [(sm, "O", 1.0) for sm in self.data.smiles_solutes]
        results = self.make_predictions(aq_smiles_pairs, self.models.g_aq_models)
        mean_predictions = [results[sm][0] for sm in aq_smiles_pairs]
        variance_predictions = [results[sm][1] for sm in aq_smiles_pairs]
        return mean_predictions, variance_predictions


    def make_gsolvref_predictions(self, verbose=False):
        if verbose:
            self.logger("Make Gsolv reference predictions")
        if self.models.g_models is None:
            raise ValueError("Gsolv models are not loaded, cannot make predictions")
        if self.data.reference_solvents is None:
            raise ValueError(
                "Gsolv reference predictions cannot be made because no refrence solvents are provided"
            )
        new_smiles_pairs = [
            (ref, sm)
            for ref, sm in zip(self.data.smiles_solutes, self.data.reference_solvents)
        ]
        unique_smiles_pairs = list(OrderedDict.fromkeys(new_smiles_pairs))
        results = self.make_gsolvaq_predictions(unique_smiles_pairs, self.models.g_models)
        # results = self.make_predictions(list(set(new_smiles_pairs)), self.models.g_models)
        mean_predictions = [results[sm][0] for sm in new_smiles_pairs]
        variance_predictions = [results[sm][1] for sm in new_smiles_pairs]
        return mean_predictions, variance_predictions

    def make_saq_predictions(self, verbose=False):
        if verbose:
            self.logger("Make logSaq predictions")
        if self.models.saq_models is None:
            raise ValueError("logSaq models are not loaded, cannot make predictions")
        aq_smiles_pairs = [(sm, 1.0) for sm in self.data.smiles_solutes]
        results = self.make_predictions(aq_smiles_pairs, self.models.saq_models)
        mean_predictions = [results[sm][0] for sm in aq_smiles_pairs]
        variance_predictions = [results[sm][1] for sm in aq_smiles_pairs]
        return mean_predictions, variance_predictions

    def make_predictions(self, smiles_set, models, molefracs=[]):
        results = dict()

        for sm in range(len(smiles_set)):
            smiles = smiles_set[sm][:-1]
            # smiles = smiles_set[sm]
            identifier = smiles_set[sm]
                
            if len(molefracs) > 0:
                molefrac = [molefracs[sm]]
            else:
                molefrac = []
            if not type(smiles) is tuple:
                smiles = [smiles]
            result = self.run_model(
                inp=models[0], models=models[2], scalers=models[1], smiles=smiles, molefrac=molefrac
            )
            if not type(smiles) is tuple:
                identifier = smiles[0]
            results[identifier] = result
        return results
    
    def run_model(self, inp=None, models=None, scalers=None, smiles=None, molefrac=[]):
        inp.num_features = 0
        preds = []
        database = MolencoderDatabase()
        if inp.num_targets == 1:
            for s, m in zip(scalers, models):
                data = DatapointList([DataPoint(smiles, None, None, molefrac, inp, database)])
                pred = predict(model=m, data=data, scaler=s, inp=inp)
                preds.append(pred)
            return np.mean(preds), np.var(preds)
        else:
            for s, m in zip(scalers, models):
                data = DatapointList([DataPoint(smiles, None, None, molefrac, inp, database)])
                pred = predict(model=m, data=data, scaler=s, inp=inp)
                preds.append(pred[0])
            return np.mean(preds, axis=0).tolist(), np.var(preds, axis=0).tolist()

    
    # def make_soluteparameter_predictions(self, verbose=True): #Here gives the error. Solute model is not loaded 
    #     if verbose:
    #         self.logger("Make solute parameter predictions")
    #     if self.models.solute_models is None:
    #         raise ValueError(
    #             "Solute parameter models are not loaded, cannot make predictions"
    #         )
        
        
    #     # unique_solute_smiles = set(self.data.smiles_solutes) #orginally set 
    #     unique_solute_smiles = list(set(self.data.smiles_solutes))

    #     results = self.make_predictions(unique_solute_smiles, self.models.solute_models)
    #     mean_predictions = [results[sm][0] for sm in self.data.smiles_solutes]
    #     variance_predictions = [results[sm][1] for sm in self.data.smiles_solutes]
    #     return mean_predictions, variance_predictions
    
    #old one rewrtitten 
    def make_soluteparameter_predictions(self, verbose=True):
        if verbose:
            self.logger('Make solute parameter predictions')        
        # print("Dataset (SMILES solutes):", self.data.smiles_solutes)
        # print("Solute Models:", self.models.solute_models)
        print("Solute Models:", self.models.solute_models)

        if self.models.solute_models is None:
            raise ValueError(
                "Solute parameter models are not loaded, cannot make predictions"
            )    
        smiles_pairs = [(sm, 1.0) for sm in self.data.smiles_solutes]
        results = self.make_predictions(smiles_pairs, self.models.solute_models)
        mean_predictions = [results[sm][0] for sm in smiles_pairs]
        variance_predictions = [results[sm][1] for sm in smiles_pairs]
        return mean_predictions, variance_predictions

