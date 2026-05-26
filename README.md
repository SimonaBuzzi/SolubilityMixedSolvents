# SolProp_ML_mix

SolProp_ML_mix is a software package that combines machine learning and thermodynamics for the prediction of solubility-related properties in solvent mixtures. Starting from the aqueous solubility of a compound, along with Abraham parameters and solvation free energy and enthalpy, the tool calculates the solubility of a solute in a given solvent or solvent mixture. Each of these properties is predicted by a dedicated machine learning model.

Model weights for the solvation free energy/enthalpy model are bundled in the [conda package](https://anaconda.org/roelleenhouts/solprop_ml). The models are trained on the [Mixed Solvent Gsolv Data Collection](https://zenodo.org/records/14238055) and the [Solubility Data Collection](https://zenodo.org/record/5970538). Datasets for training the aqueous solubility model and Abraham parameters are available [here](https://pubs.acs.org/doi/full/10.1021/jacs.2c01768).

---

## Requirements

SolProp_ML_mix has been tested on **macOS** and **Linux**. It may not work on Windows.

---

## Installation

SolProp_ML_mix can be installed either from conda or directly from this repository. Both options require conda. If you do not have conda installed, download Miniconda from [https://conda.io/miniconda.html](https://conda.io/miniconda.html).

A SolProp conda package is available here: (https://anaconda.org/channels/simonabuzzi/packages/solprop_ml_mix/overview)

---

## Supported Solutes and Solvents

SolProp_ML_mix currently supports predictions for:

- **Solutes:** Electrically neutral compounds containing H, B, C, N, O, S, P, F, Cl, Br, and I
- **Solvents:** Nonionic liquid solvents

> **Note:** Predictions for solutes or solvents outside of these specifications may not be reliable.

---

## Example Prediction Files

Input and output definitions are described in sample Python files located under `SolProp/sample_files/`. Currently, only solvation free energy predictions in both pure and mixed solvents are available. Support for solid solubility predictions will be added in a future update.

---

## How to Cite

If you use this software in your research, please cite the relevant papers:

**SolProp:**
> Vermeire, F. H.; Chung, Y.; Green, W. H. Predicting Solubility Limits of Organic Solutes for a Wide Range of Solvents and Temperatures. *J. Am. Chem. Soc.* 2022. https://pubs.acs.org/doi/full/10.1021/jacs.2c01768

**MolPool:**
> Leenhouts, R. J.; Morgan, N.; Al Ibrahim, E.; Green, W. H.; Vermeire, F. H. Pooling Solvent Mixtures for Solvation Free Energy Predictions. *arXiv* 2024. https://arxiv.org/pdf/2412.01982

**SolProp-mix:**
> Buzzi, S.; Al Ibrahim, E.; Di Caprio, U.; et al. Predicting Solubility Curves in Solvent Mixtures Using Thermodynamic Cycles and Machine Learning. *ChemRxiv* 2026. https://doi.org/10.26434/chemrxiv.15003912/v1

---

## License

SolProp_ML_mix is a free, open-source software package distributed under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode) license.

---

## Contact

For questions or feedback, please contact:

- [Simona Buzzi](mailto:simona.buzzi@kuleuven.be)
- [Roel Leenhouts](mailto:roel.leenhouts@kuleuven.be)





