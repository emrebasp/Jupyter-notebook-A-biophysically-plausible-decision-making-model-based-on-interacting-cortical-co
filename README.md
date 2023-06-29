# Jupyter notebook of a biophysically plausible decision-making model based on interacting cortical columns

Editor: Emre Baspinar (ORCID: 0000-0002-9518-3705).

Contributors: Gloria Cecchini, Ruben Moreno-Bote, Ignasi Cos, Alain Destexhe

This notebook contains the implementation of the biophysically realistic AdEx mean-field model proposed in [1] for the reward-driven consequential decision-making task presented in [2]. The implementation is in Python 3.8.5. The code writing and the preparation of the Jupyter notebook was done by Emre Baspinar (CNRS, NeuroPSI, Paris-Saclay). The model design was contributed by Emre Baspinar, Gloria Cecchini (U. of Barcelona), Rub ́en Moreno Bote (Pompeu Fabra University, Barcelona), Ignasi Cos (U. of Barcelona) and Alain Destexhe (CNRS, NeuroPSI, Paris-Saclay). The related manuscript [1] is a joint work of Emre Baspinar, Gloria Cecchini, Michael DePass (U. of Barcelona), Marta Andujar (U. of Rome), Pierpaolo Pani (U. of Rome), Stefano Ferraina (U. of Rome), Rub ́en Moreno Bote, Ignasi Cos and Alain Destexhe. See [1] for more information. This work was supported by Human Brain Project (European Union grant H2020-945539).

This notebook is updated regularly. Please feel free to contact if you have any feedback.

Zenodo link: https://doi.org/10.5281/zenodo.7682309

Contact: emre.baspinar@cnrs.fr

This notebook is licensed by Creative Commons Attribution 4.0 International Public License (CC BY 4.0). Please cite [1] and [3] if you use this notebook in your work.

Please feel free to contact for questions, comments and feedback. Contact: emre.baspinar@cnrs.fr

Description: This notebook provides the implementation of the implementation of the model, an example simulation of the model related to the behavioral experiments of the decision-making paradigm considered in [1, 2] and three case studies. The results of the example simulation are saved automatically in the notebook folder as .npy files in case of that the user wants to analyze the results. This option can be commented out without causing to the proper functioning of the rest of the notebook. The default values for the simulations are provided and they can be easily changed by the user.

Glossary

MainNotebook.ipnyb: This is the main notebook to run the simulations and the case studies. It contains explanations related to the model and to the simulations. This file uses the .py files given below.

cell library.py: It contains the parameters of the biophysical cell properties of the neurons. Do not change unless you add new cell types.

DiffOperator.py: It contains the stochastic differential equations of the AdEx mean-field equations corresponding to both cortical columns.

NeuronConnectivity.py: It contains the functions which we use to load the transfer functions of Regular Spiking (RS) and Fast Spiking (FS) cells. The transfer functions and their parameters are based on a fitting to experimental data, therefore the parameters should be kept fixed. Do not change this file.

SDEIntegrator.py: This is the Euler-Maruyama integrator. It integrates the SDEs found in DiffOperator.py in time.

syn and connec library.py: It contains the connectivity and synaptic properties of the neurons.

theoretical tools.py: It contains implementation of some analytical functions appearing in the AdEx mean- field equations.

In addition to these .py files, there are two .npy files in data folder: FS-cell CONFIG1 fit.npy and RS- cell CONFIG1 fit.npy. They contain the fitted parameters to the experimentally obtained RS and FS cell transfer functions. Do not change the folder of these files. Finally, showcaseData folder contains the data which MainNotebook.ipnyb uses for the case studies.

This notebook uses Matplotlib, Nump, Random, Os and Scipy.io libraries.

## References:

[1]: E. Baspinar, G. Cecchini, M. DePass, M. Andujar, P. Pani, S. Ferraina, R. Moreno-Bote, I. Cos, A. Destexhe, "A biologically plausible decision- making model based on interacting cortical columns", bioRxiv, 2023, doi: https://doi.org/10.1101/2023.02.28.530384 

[2]: G. Cecchini, M. DePass, E. Baspinar, M. Andujar, S. Ramawat, P. Pani, A. Destexhe, R. Moreno-Bote, I. Cos, "A theoretical formalization of consequence-based decision-making", bioRxiv, 2023, doi: https://doi.org/10.1101/2023.02.14.528595.

[3]: E. Baspinar, G. Cecchini, R. Moreno-Bote, I. Cos, A. Destexhe, "Jupyter notebook of a biophysically plausible decision-making model based on interacting cortical columns (1.0.0)", Zenodo, 2023, doi: https:// doi.org/10.5281/zenodo.7682309
