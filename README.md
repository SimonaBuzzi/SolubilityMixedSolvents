# SolProp
The SolProp package contains a code that combines machine learning and thermodynamics for the prediction of solubility related properties, which now is being extended to mixture properties. To make the predictions using the package the neural network model weights are needed. The package including the model weights is available as a [conda package](https://anaconda.org/roelleenhouts/solprop_ml). The models in SolProp are trained with databases from the [Mixed solvent Gsolv data collection](https://zenodo.org/records/14238055), and [Solubility data collection](https://zenodo.org/record/5970538).

## Requirements
SolProp_ML has been so far tested to work on Mac and Linux OS. It may not work on Windows.

## Installation
SolProp can be installed from conda and source(i.e. directly from this git repo). Both options require conda, so first install for example Miniconda from [https://conda.io/miniconda.html](https://conda.io/miniconda.html).

### Option 1:
1. Create a conda environment based on Python 3.9 using "conda create --name myenv python=3.9"
2. Activate the environment "conda activate myenv"
3. Install the conda package: "conda install roelleenhouts::solprop_ml"

### Option 2:
1. "git clone git@gitlab.kuleuven.be:creas/vermeiregroup/solprop.git"
2. "cd solprop" ("solprop" is the path to where you cloned the git repository)
3. "conda env create --name myenv python=3.9"
4. "Activate the environment "conda activate myenv"
5. Install the packages in requirements.txt
7. Download the machine learning model weights from [here](https://zenodo.org/records/14238055).
8. Copy the "SolPropmixExp" folder from the "ModelWeights" folder of the downloaded file and place them under "SolProp_ML/solvation_predictor/trained_models/"

## Supported solvents and solutes
SolProp_ML currently supports prediction for only electrically neutral solute compounds containing H, B, C, N, O, S, P, F, Cl, Br, and I and nonionic liquid solvents. Predictions for any out-of-range solvents and solutes won't be reliable.

## Example predictions files
The definitions of prediction inputs and outputs are described in a sample python file. Please refer to the `.py` file located under "Solprop/sample_files/". Currently only the solvation free energy predictions in both pure and mixed solvents are available. Updates will come to include the predictions for solid solubility.

## How to Cite
If you use this software for research, please cite the SolProp or mixture Gsolv paper (link to be added soon) as follows:

Vermeire, F. H.; Chung, Y.; Green, W. H. Predicting Solubility Limits of Organic Solutes for a Wide Range of Solvents
and Temperatures. https://pubs.acs.org/doi/full/10.1021/jacs.2c01768
Leenhouts, R. J.; Morgan, N; Al Ibrahim, E; Green, W. H.; Vermeire F. H. Pooling solvent mixtures for solvation free energy predictions. https://arxiv.org/pdf/2412.01982

## License Information
SolProp is a free, open-source software package distributed under the 
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode).

## Contact
For any questions, please contact [Simona Buzzi](mailto:simona.buzzi@kuleuven.be) or [Roel Leenhouts](mailto:roel.leenhouts@kuleuven.be) 





