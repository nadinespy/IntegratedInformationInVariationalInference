function [ redred, localred ] = DoubleRedundancyCCSDiscrete(varargin)
%%DOUBLEREDUNDANCYCCSDISCRETE Compute the PhiID double-redundancy of discrete
% input data using the CCS PID. It uses a plug-in MI estimator.
%
% NOTE: assumes JIDT has been already added to the javaclasspath.
%
%   R = DOUBLEREDUNDANCYCCSDISCRETE(X, TAU), where X is a D-by-T data matrix of
%   D dimensions for T timesteps, and TAU is an integer integration timescale,
%   computes the double-redundancy across the minimum information bipartition
%   (MIB) of X. If TAU is not provided, it is set to 1.
%
%   R = DOUBLEREDUNDANCYCCSDISCRETE(X1, X2, Y1, Y2), where all inputs are
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
  R = private_TDCCS(varargin{1});
elseif nargin == 2
  R = private_TDCCS(varargin{1}, varargin{2});
elseif nargin == 4
  R = private_FourVectorCCS(varargin{1}, varargin{2}, varargin{3}, varargin{4});
else
  error('Wrong number of arguments. See `help DoubleRedundancyCCSDiscrete` for help.');
end

redred = mean(R);

if nargout > 1
  localred = R;
end

end


%*********************************************************
%*********************************************************
function [ redred ] = private_TDCCS(X, tau)

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

redred = private_FourVectorCCS(X(p1, 1:end-tau), X(p2, 1:end-tau), ...
                                X(p1, 1+tau:end), X(p2, 1+tau:end));

end


%*********************************************************
%*********************************************************
function [ redred ] = private_FourVectorCCS(X1, X2, Y1, Y2)

% Argument checks and parameter initialisation
T = size(X1, 2);
if size(X2, 2) ~= T || size(Y1, 2) ~= T || size(Y2, 2) ~= T
  error('All input vectors must have the same length');
end


% Binarise data (if not already discrete) and stack for easier handling
binarify = @(v) ensure_combined(isdiscrete(v)*v + (~isdiscrete(v))*(v > mean(v, 2)));
X = [binarify(X1); binarify(X2); binarify(Y1); binarify(Y2)];


Ixytab = localmi(X, [1, 2], [3, 4]);

Ixta = localmi(X, 1, 3);
Ixtb = localmi(X, 1, 4);
Iyta = localmi(X, 2, 3);
Iytb = localmi(X, 2, 4);

Ixyta = localmi(X, [1, 2], 3);
Ixytb = localmi(X, [1, 2], 4);
Ixtab = localmi(X, 1, [3, 4]);
Iytab = localmi(X, 2, [3, 4]);

Rxytab = localred(Ixtab, Iytab, Ixytab);
Rabtxy = localred(Ixyta, Ixytb, Ixytab);
Rxyta  = localred(Ixta, Iyta, Ixyta);
Rxytb  = localred(Ixtb, Iytb, Ixytb);
Rabtx  = localred(Ixta, Ixtb, Ixtab);
Rabty  = localred(Iyta, Iytb, Iytab);


% This quantity equals redred - synsyn
double_coinfo = - Ixta - Ixtb - Iyta - Iytb + ...
                + Ixtab + Iytab + Ixyta + Ixytb - Ixytab + ...
                + Rxyta + Rxytb - Rxytab + ...
                + Rabtx + Rabty - Rabtxy;

redred = all(~diff([sign(Ixta), sign(Ixtb), sign(Iyta), sign(Iytb), sign(double_coinfo)],1,2),2).*double_coinfo;

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

function [ l ] = localmi(X, src, tgt)

  x = ensure_combined(X(src,:));
  y = ensure_combined(X(tgt,:));

  miCalc = javaObject('infodynamics.measures.discrete.MutualInformationCalculatorDiscrete', max(x)+1, max(y)+1, 0);
  miCalc.initialise();
  miCalc.addObservations(x', y');
  l = miCalc.computeLocalFromPreviousObservations(x', y');

end

function [ l ] = localred(mi1, mi2, mi12)
  c = mi12 - mi1 - mi2;
  signs = [sign(mi1), sign(mi2), sign(mi12), sign(-c)];
  l = all(signs == signs(:,1), 2).*(-c);
end

