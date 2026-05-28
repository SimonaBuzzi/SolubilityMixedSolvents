
from solvation_predictor.solubility.SolubilityPredictions import SolubilityPredictions



sp = SolubilityPredictions()

try:
    sp.make_gsolvref_predictions()
    print("loading was successfully")
except Exception as e:
    print("An exeption occurred", e)


# def make_gsolvref_predictions(self, verbose=False):
#     if verbose:
#         self.logger("Make Gsolv reference predictions")
#     if self.models.g_models is None:
#         raise ValueError("Gsolv models are not loaded, cannot make predictions")
#     if self.data.reference_solvents is None:
#         raise ValueError(
#             "Gsolv reference predictions cannot be made because no refrence solvents are provided"
#         )
#     new_smiles_pairs = [
#         (ref, sm)
#         for ref, sm in zip(self.data.smiles_solutes, self.data.reference_solvents)
#     ]
#     unique_smiles_pairs = list(OrderedDict.fromkeys(new_smiles_pairs))
#     results = self.make_gsolvaq_predictions(unique_smiles_pairs, self.models.g_models)
#     # results = self.make_predictions(list(set(new_smiles_pairs)), self.models.g_models)
#     mean_predictions = [results[sm][0] for sm in new_smiles_pairs]
#     variance_predictions = [results[sm][1] for sm in new_smiles_pairs]
#     return mean_predictions, variance_predictions


# import solvation_predictor.solubility as solubility
# print(solubility)
# print(dir(solubility))
