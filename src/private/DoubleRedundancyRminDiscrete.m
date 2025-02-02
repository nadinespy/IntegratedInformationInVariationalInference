function [ redred, localred ] = DoubleRedundancyRminDiscrete(varargin)
%%DOUBLEREDUNDANCYRMINDISCRETE Compute the PhiID double-redundancy of discrete
% input data using the Rmin PID. It uses a plug-in MI estimator.
%
% NOTE: assumes JIDT has been already added to the javaclasspath.
%
%   R = DOUBLEREDUNDANCYRMINDISCRETE(X, TAU), where X is a D-by-T data matrix of
%   D dimensions for T timesteps, and TAU is an integer integration timescale,
%   computes the double-redundancy across the minimum information bipartition
%   (MIB) of X. If TAU is not provided, it is set to 1.
%
%   R = DOUBLEREDUNDANCYRMINDISCRETE(X1, X2, Y1, Y2), where all inputs are
%   matrices with the same number of columns (i.e. same number of samples),
%   computes the double-redundancy of the mutual info between them, I(X1, X2;
%   Y1, Y2).
%
%   [R, L] = DOUBLEREDUNDANCYCCSDISCRETE(...) returns the local
%   double-redundancy values for each sample in the input.
%
% If input data is discrete-compatible (as per ISDISCRETE), it is passed
% directly to the underlying information-theoretic calculators. If it isn't
% (e.g. if it is real-valued data), it is mean-binarised first.
%
% Reference:
%   Mediano*, Rosas*, Carhart-Harris, Seth and Barrett (2019). Beyond
%   Integrated Information: A Taxonomy of Information Dynamics Phenomena.
%
% Pedro Mediano, Jan 2021

if nargin == 1
  R = private_TDRmin(varargin{1});
elseif nargin == 2
  R = private_TDRmin(varargin{1}, varargin{2});
elseif nargin == 4
  R = private_FourVectorRmin(varargin{1}, varargin{2}, varargin{3}, varargin{4});
else
  error('Wrong number of arguments. See `help DoubleRedundancyRminDiscrete` for help.');
end

redred = mean(R);

if nargout > 1
  localred = R;
end

end


%*********************************************************
%*********************************************************
function [ redred ] = private_TDRmin(X, tau)

% Argument checks and parameter initialisation
if isempty(X) || ~ismatrix(X)
  error('Input must be a 2D data matrix');
end
[D, T] = size(X);
if T <= D
  error(sprintf(['Your matrix has %i dimensions and %i timesteps. ', ...
        'If this is true, you cant compute a reasonable covariance matrix. ', ...
        'If it is not true, you may have forgotten to transpose the matrix'], D, T));
end
if nargin < 2 || isempty(tau)
  tau = 1;
end
integer_tau = ~isinf(tau) & floor(tau) == tau;
if ~integer_tau || tau < 1
  error('Timescale tau needs to be a positive integer.');
end

% Binarise the data, if not already discrete
if ~isdiscrete(X)
  X = 1*(X > mean(X, 2));
end


% Use JIDT to compute Phi and MIB
phiCalc = javaObject('infodynamics.measures.discrete.IntegratedInformationCalculatorDiscrete', 2, size(X, 1));
if tau > 1
  phiCalc.setProperty(phiCalc.PROP_TAU, num2str(tau));
end
phiCalc.setObservations(octaveToJavaIntMatrix(X'));
phi = phiCalc.computeAverageLocalOfObservations();
mib = phiCalc.getMinimumInformationPartition();

% Extract MIB partition indices
p1 = str2num(mib.get(0).toString()) + 1;
p2 = str2num(mib.get(1).toString()) + 1;

redred = private_FourVectorRmin(X(p1, 1:end-tau), X(p2, 1:end-tau), ...
                                X(p1, 1+tau:end), X(p2, 1+tau:end));

end


%*********************************************************
%*********************************************************
function [ redred ] = private_FourVectorRmin(X1, X2, Y1, Y2)

% Argument checks and parameter initialisation
T = size(X1, 2);
if size(X2, 2) ~= T || size(Y1, 2) ~= T || size(Y2, 2) ~= T
  error('All input vectors must have the same length');
end


% Binarise data (if not already discrete) and stack for easier handling
binarify = @(v) ensure_combined(isdiscrete(v)*v + (~isdiscrete(v))*(v > mean(v, 2)));
src = {binarify(X1), binarify(X2)};
tgt = {binarify(Y1), binarify(Y2)};


% Compute specificity (r^+) as the minimum over source local entropies
rplus = inf([T, 1]);
for i=1:length(src)
  hCalc = javaObject('infodynamics.measures.discrete.EntropyCalculatorDiscrete', max(src{i})+1);
  hCalc.initialise();
  rplus = min(rplus, hCalc.computeLocal(octaveToJavaIntArray(src{i}')));
end

% Compute ambiguity (r^-) as the minimum over source and target conditional entropies
rminus = inf([T, 1]);
for i=1:length(src)
  for j=1:length(tgt)

    joint = ensure_combined([src{i}; tgt{j}]);

    hCalc  = javaObject('infodynamics.measures.discrete.EntropyCalculatorDiscrete', max(tgt{i})+1);
    jhCalc = javaObject('infodynamics.measures.discrete.EntropyCalculatorDiscrete', max(joint)+1);
    
    hc = jhCalc.computeLocal(octaveToJavaIntArray(joint')) - hCalc.computeLocal(octaveToJavaIntArray(tgt{j}'));
    rminus = min(rminus, hc);
  end
end

% Take redundancy as their difference
redred = rplus - rminus;

end


%*********************************************************
%*********************************************************
function [ V ] = ensure_combined(U)
  if size(U, 1) == 1
    V = U;
  else
    [~,~,V] = unique(U', 'rows');
    V = V(:)' - 1;
  end
end
