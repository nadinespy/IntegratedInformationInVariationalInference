# Integrated Information in Variational Inference
&nbsp;

This repository contains code for calculating integrated information in variational inference. We use a hybrid (i.e., partly numerical, partly analytical) approach to simulate the evolution of Gaussian parameters during variational inference, and use those as inputs for calculating integrated information as well as PhiID-based measures of emergence at each point in the evolution. This work (led by Nadine Spychala & Miguel Aguilera) is ongoing and not yet documented and tested.

The current up-to-date script file is `mec_var_inf_steady_state_param_sweep.ipynb` which uses functions from `mec_var_inf.py` under the `src` directory. The code requires Matlab engine to make use of Matlab functions. For this to work, Matlab needs to be installed locally. This code is not packaged up yet - local-specific directories need to be set at the top in both the script and module file. `mec_var_inf_steady_state_param_sweep_plotting.ipynb` does the plotting. 
