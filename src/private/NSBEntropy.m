function [ S_nsb ] = NSBEntropy(v)
%%NSBENTROPY Nemenman-Shafee-Bialek entropy estimator for discrete data
%
%   H = NSBENTROPY(V), where V is a T-by-D data matrix of D dimensions for T
%   timesteps, computes the joint entropy of discrete-valued sample V using the
%   NSB Bayesian estimator. Returns the result in bits.
%
% Reference:
%   I Nemenman, F Shafee, and W Bialek. Entropy and inference, revisited. NIPS
%   14, 2002. arXiv: physics/0108025
%
% Pedro Mediano, Sep 2020

addpath([fileparts(mfilename('fullpath')), '/nsb-entropy-oct']);

[~, ~, idx] = unique(v, 'rows');
n = accumarray(idx(:), 1)';

% n must be a row vector for the NSB function to work
assert(size(n,1) == 1);

% Preprocessing for NSB calculation, copied from nsb-entropy-oct/intest.m
K = length(n);
nx = n(n>0);
kx = ones(size(nx));
qfun = 1;
precision = 0.05;

% Use evalc to capture all the silly text that find_nsb_entropy prints
[~, S_nsb, dS_nsb, S_cl, dS_cl, xi_cl, S_ml,errcode] = evalc('find_nsb_entropy (kx, nx, K, precision,qfun)');

% Convert result from nats to bits
S_nsb = S_nsb / log(2);

