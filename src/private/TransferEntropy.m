function [ te ] = TransferEntropy(X, Y)
%%TRANSFERENTROPY Simple interface to the JIDT Gaussian transfer entropy
% calculator, adapted for the fMRI analysis pipeline.
%
% The aim is to have this function progressively extended to fill our
% transfer-entropic needs. For now it computes simple unembedded, delay=1
% multivariate Gaussian TE from X to Y.
%
% NOTE: assumes JIDT has been already added to the javaclasspath.
%
% Inputs:
%   X   -- Dx-by-T source data matrix of Dx dimensions for T timesteps
%   Y   -- Dy-by-T target data matrix of Dy dimensions for T timesteps
%
% Returns:
%  te -- scalar, transfer entropy
%
% Pedro Mediano, Dec 2019

%% Argument checks and parameter initialisation
if isempty(X) || ~ismatrix(X)
  error('Input X must be a 2D data matrix');
end
if isempty(Y) || ~ismatrix(Y)
  error('Input Y must be a 2D data matrix');
end
[Dx, T] = size(X);
[Dy, Ty] = size(Y);
if abs(T-Ty) > 1e-6
  error('Both input matrices must have same number of timesteps (columns)');
end
if T <= Dx || T <= Dy
  error(['One of your matrices has more dimensions than timesteps. ', ...
        'If this is true, you cant compute a reasonable covariance matrix. ', ...
        'If it is not true, you may have forgotten to transpose the matrix']);
end


%% Initialise calculator and compute TE
teCalc = infodynamics.measures.continuous.gaussian.TransferEntropyCalculatorMultiVariateGaussian();
teCalc.setProperty(teCalc.PROP_AUTO_EMBED_METHOD, teCalc.AUTO_EMBED_METHOD_NONE);

k = 1; k_tau = 1;
l = 1; l_tau = 1;
delay = 1;
teCalc.initialise(Dx, Dy, k, k_tau, l, l_tau, delay)
teCalc.setObservations(X', Y');
te = teCalc.computeAverageLocalOfObservations();

