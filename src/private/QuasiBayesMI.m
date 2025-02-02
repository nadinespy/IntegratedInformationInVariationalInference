function [ mi ] = QuasiBayesMI(x, y)
%%QUASIBAYESMI Quasi-Bayesian mutual information estimator for discrete data
%
%   MI = QUASIBAYESMI(X, Y) computes a quasi-bayesian estimator of MI between
%   two discrete samples, using the NSB1 estimator from Archer (2013). Input
%   data may have different bases and different number of dimensions but must
%   be of the same length, such that size(X) == [Dx, T] and size(Y) == [Dy, T].
%   Returns the result in bits.
%
% References:
%   Archer, E.; Park, I.M.; Pillow, J.W. Bayesian and Quasi-Bayesian Estimators
%   for Mutual Information from Discrete Data. Entropy 2013, 15, 1738-1755.
%
%   I Nemenman, F Shafee, and W Bialek. Entropy and inference, revisited. NIPS
%   14, 2002. arXiv: physics/0108025
%
% Pedro Mediano, Jan 2021

if size(x, 2) ~= size(y, 2)
  error('Input samples must have the same length.');
end
[Dx, T] = size(x);
Dy = size(y, 1);
if T <= Dx || T <= Dy
  error(sprintf(['Matrix X has %i dimensions and %i timesteps. ', ...
        'If this is true, your estimator may be heavily biased. ', ...
        'If it is not true, you may have forgotten to transpose the matrix'], ...
        min(Dx, Dy), T));
end

h1 = NSBEntropy(x');
h2 = NSBEntropy(y');
h12 = NSBEntropy([x', y']);

mi = h1 + h2 - h12;

