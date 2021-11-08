function [ A, L ] = PhiIDFull(varargin)
%%PHIIDFULL Computes full PhiID decomposition of input data, assuming it
% follows a multivariate Gaussian distribution.
%
%   A = PHIIDFULL(X, TAU), where X is a D-by-T data matrix of D dimensions for
%   T timesteps, and TAU is an integer integration timescale, computes the
%   PhiID decomposition of the time-delayed mutual information of X. If TAU is
%   not provided, it is set to 1. If D > 2, PhiID is calculated across the
%   minimum information bipartition (MIB) of the system.
%
%   A = PHIIDFULL(X1, X2, Y1, Y2), where all inputs matrices with T columns,
%   computes the PhiID decomposition of the mutual information between them,
%   I(X1, X2; Y1, Y2).
%
%   A = PHIIDFULL(..., M) uses PhiID measure M (default: 'CCS')
%
%   [A, L] = PHIIDFULL(...) returns the local PhiID atoms for each sample in
%   the input.
%
% In all cases, results are returned in a struct A with all integrated
% information atoms. Atoms are named with a three-char string of the form QtP,
% where Q and P are one of r, x, y, or s (redundancy, unique X, unique Y or
% synergy, respectively). For example:
%
%            A.rtr is atom {1}{2}->{1}{2}
%            A.xty is atom {1}->{2}
%            A.stx is atom {12}->{1}
%            ...
%
% Reference:
%   Mediano*, Rosas*, Carhart-Harris, Seth and Barrett (2019). Beyond
%   Integrated Information: A Taxonomy of Information Dynamics Phenomena.
%
% Pedro Mediano and Andrea Luppi, Jan 2021

% Find JIDT and add relevant paths; calculate average PhiID atoms
p = strrep(mfilename('fullpath'), 'PhiIDFull', '');
if exist([p, '../elph_base'], 'dir')
  addpath([p, '../elph_base']);
end
if ~any(~cellfun('isempty', strfind(javaclasspath('-all'), 'infodynamics')))
  if exist([p, '../elph_base/infodynamics.jar'], 'file')
    javaaddpath([p, '../elph_base/infodynamics.jar']);
  elseif exist([p, 'private/infodynamics.jar'], 'file')
    javaaddpath([p, 'private/infodynamics.jar']);
  else
    error('Unable to find JIDT (infodynamics.jar).');
  end
end

if nargin == 1
  atoms = private_TDPhiID(varargin{1});
elseif nargin == 2
  atoms = private_TDPhiID(varargin{1}, varargin{2});
elseif nargin == 3
  atoms = private_TDPhiID(varargin{1}, varargin{2}, varargin{3});
elseif nargin == 4
  atoms = private_FourVectorPhiID(varargin{1}, varargin{2}, varargin{3}, varargin{4});
elseif nargin == 5
  atoms = private_FourVectorPhiID(varargin{1}, varargin{2}, varargin{3}, varargin{4}, varargin{5});
else
  error('Wrong number of arguments. See `help PhiIDFull` for help.');
end

if any(structfun(@(x) any(~isfinite(x)), atoms))
  warning('PhiID:Outlier', 'Outliers detected in PhiID computation. Results may be biased.');
end

% calculate average PhiID atoms
A = structfun(@(x) mean(x(isfinite(x))), atoms, 'UniformOutput', 0);
if nargout > 1
  L = atoms;
end

end


%*********************************************************
%*********************************************************
function [ atoms ] = private_TDPhiID(X, tau, measure)

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
if nargin < 3 || isempty(measure)
  measure = 'CCS';
end

% Create copy of the data scaled to unit variance (for numerical stability)
sX = X./repmat(std(X')', [1, T]);


% Use JIDT to compute Phi and MIB
phiCalc = javaObject('infodynamics.measures.continuous.gaussian.IntegratedInformationCalculatorGaussian');
if tau > 1
  phiCalc.setProperty(phiCalc.PROP_TAU, num2str(tau));
end
phi = phiCalc.compute(octaveToJavaDoubleMatrix(sX'));
mib = phiCalc.getMinimumInformationPartition();

% Extract MIB partition indices
p1 = str2num(mib.get(0).toString()) + 1;            % indices of partition 1, e.g., in an 8-element system, [1,2,7,8]
p2 = str2num(mib.get(1).toString()) + 1;            % indices of partition 2, e.g., in an 8-element system, [3,4,5,6]

% Stack data and call full PhiID function
atoms = private_FourVectorPhiID(sX(p1,1:end-tau), sX(p2,1:end-tau), ...
                                sX(p1,1+tau:end), sX(p2,1+tau:end), measure);

end


%*********************************************************
%*********************************************************

% returns local PhiID atoms
function [ atoms ] = private_FourVectorPhiID(X1, X2, Y1, Y2, measure)			   %X1: partition 1 at t
																					% X2: partition 2 at t
																					% Y1: partition 1 at t+1
																					% Y2: partition 2 at t+1                                                                                                                  

% Argument checks and parameter initialisation
checkmat = @(v) ~isempty(v) && ismatrix(v);
if ~(checkmat(X1) && checkmat(X2) && checkmat(Y1) && checkmat(Y2))
  error('All inputs must be non-empty data matrices');
end
T = size(X1, 2);
if size(X2, 2) ~= T || size(Y1, 2) ~= T || size(Y2, 2) ~= T
  error('All input matrices must have the same number of columns');
end
if nargin < 5 || isempty(measure)
  measure = 'CCS';
end

if strcmp(lower(measure), 'ccs')
  RedFun = @RedundancyCCS;
  DoubleRedFun = @DoubleRedundancyCCS;
elseif strcmp(lower(measure), 'mmi')
  RedFun = @RedundancyMMI;
  DoubleRedFun = @DoubleRedundancyMMI;
else
  error(['Unknown redundancy measure. Currently implemented measures are ''CCS'' and ''MMI''']);
end

% Indices of source and target variables in the joint covariance matrix
p1 = 1:size(X1, 1);								% indices of first source partition (e.g., in an 8-element system [1,2,3,4])
p2 = p1(end)+1:p1(end)+size(X2, 1);			     % indices of second source partition (e.g., in an 8-element system [5,6,7,8])
t1 = p2(end)+1:p2(end)+size(Y1, 1);				% indices of first target partition (e.g., " [9,10,11,12])
t2 = t1(end)+1:t1(end)+size(Y2, 1);				 % indices of second target partition (e.g., " [13,14,15,16])
D = t2(end);									  % total number of indices


% Create copy of the data scaled to unit variance (for numerical stability)
X = [X1; X2; Y1; Y2];							     % stack all variables from partitions row-wise
sX = X./repmat(std(X')', [1, T]);				       % we again scale to unit variance, because the matrix has changed 

% Compute mean and covariance for all the data
% (to be used by the local IT functions below)
S = cov(sX');									   % e.g., in an 8-element system, the size will be 16x16, it's the time-lagged covariance matrix             
mu = mean(sX');                                                  % e.g., " , the size will be 1x16
assert(all(size(mu) == [1, D]) && all(size(S) == [D, D]));

% Define local information-theoretic functions
h = @(idx) -log(mvnpdf(sX(idx,:)', mu(idx), S(idx,idx)));  % multivariate entropy
mi  = @(src, tgt) h(src) + h(tgt) - h([src, tgt]);			% mutual information (I(X;Y) = H(X) + H(Y) - H(X,Y)) (not further used below)

% Pre-compute entropies necessary for all IT quantities (all quantities
% have as many rows as time-steps in the time-series)
h_p1 = h(p1);												% entropy of partition 1 at t             H(1(t)) 
h_p2 = h(p2);												% entropy pf partiiton 2 at t             H(2(t))
h_t1 = h(t1);													% entropy of partition 1 at t+1         H(1(t+1))
h_t2 = h(t2);													% entropy of partition 2 at t+1         H(2(t+1))

h_p1p2 = h([p1 p2]);											% multivariate entropy (ME) of partition 1 & 2 at t         H(1(t),      2(t)) 
h_t1t2 = h([t1 t2]);											% ME of partition 1 & 2 at t+1                                       H(1(t+1), 2(t+1))
h_p1t1 = h([p1 t1]);											% ME of partition 1 at t & t+1                                       H(1(t),      1(t+1)) 
h_p1t2 = h([p1 t2]);											% ME of partition 1 at t & partition 2 at t+1                  H(1(t),      2(t+1)) 
h_p2t1 = h([p2 t1]);											% ME of partition 2 at t & partition 1 at t+1                  H(2(t),      1(t+1)) 
h_p2t2 = h([p2 t2]);											% ME of partition 2 at t & t+1                                       H(2(t),      2(t+1)) 

h_p1p2t1 = h([p1 p2 t1]);									% ME of partition 1 & 2 at t & partition 1 at t+1           H(1(t),       2(t),       1(t+1)) 
h_p1p2t2 = h([p1 p2 t2]);									% ME of partition 1 & 2 at t & partition 2 at t+1           H(1(t),       2(t),        2(t+1)) 
h_p1t1t2 = h([p1 t1 t2]);									 % ME of partition 1 at t & t+1 & partition 2 at t+1        H(1(t),       1(t+1),   2(t+1)) 
h_p2t1t2 = h([p2 t1 t2]);									 % ME of partition 2 at t & t+1 & partition 1 at t            H(2(t),       2(t+1),   1(t+1))

h_p1p2t1t2 = h([p1 p2 t1 t2]);							    % ME of partition 2 at t & t+1 & partition 1 at t & t+1  H(2(t),       2(t+1),   1(t),       (t+1))

% Compute local PhiID quantities as entropy combinations (all quantities have as many rows as time-steps in the time-series)
% variable names: 
% R: redundant information
% I: mutual information
% x: source variable 1
% y: source variable 2
% a: target variable 1
% b: target variable 2


Ixytab = h_p1p2 + h_t1t2 - h_p1p2t1t2;					    % all 16 atoms (this is what we're trying to decompose): I(1(t),1(t+1);2(t),2(t+1))      H(1(t),2(t)) + H(1(t+1),2(t+1)) - H(1(t),1(t+1),2(t),2(t+1))          

Ixta = h_p1 + h_t1 - h_p1t1;									% {1}{2}-->{1}{2} + {1}{2}-->{1} + {1}-->{1}{2} + {1}-->{1}:																					I(1(t);1(t+1))                          H(1(t)) + H(1(t+1)) - H(1(t),1(t+1))                                                                                        
Ixtb = h_p1 + h_t2 - h_p1t2;									% {1}{2}-->{1}{2} + {1}{2}-->{2} + {1}-->{1}{2} + {1}-->{2}:																					I(1(t);2(t+1))                          H(1(t)) + H(2(t+1)) - H(1(t),2(t+1))
Iyta = h_p2 + h_t1 - h_p2t1;									% {1}{2}-->{1}{2} + {1}{2}-->{1} + {2}-->{1}{2} + {2}-->{1}:																					I(2(t);1(t+1))                          H(2(t)) + H(1(t+1)) - H(2(t),1(t+1))
Iytb = h_p2 + h_t2 - h_p2t2;									% {1}{2}-->{1}{2} + {1}{2}-->{2} + {1}-->{1}{2} + {1}-->{2}+ {2}-->{1}{2} + {2}-->{2} + {12}-->{1}{2} + {12}-->{2}:				     I(2(t);2(t+1))                         H(2(t)) + H(2(t+1)) - H(2(t),2(t+1))

Ixyta = h_p1p2 + h_t1 - h_p1p2t1;							 % {1}{2}-->{1}{2} + {1}{2}-->{1} + {1}-->{1}{2} + {1}-->{1} + {2}-->{1}{2} + {2}-->{1} + {12}-->{1}{2} + {12}-->{1}:					 I(1(t),2(t);1(t+1))                  H(1(t),2(t)) + H(1(t+1)) - H(1(t),2(t),1(t+1))
Ixytb = h_p1p2 + h_t2 - h_p1p2t2;							 % {1}{2}-->{1}{2} + {1}{2}-->{2} + {1}-->{1}{2} + {1}-->{2}+ {2}-->{1}{2} + {2}-->{2} + {12}-->{1}{2} + {12}-->{2}:					 I(1(t),2(t);2(t+1))                  H(1(t),2(t)) + H(2(t+1)) - H(1(t),2(t),2(t+1))
Ixtab = h_p1 + h_t1t2 - h_p1t1t2;							  % {1}{2}-->{1}{2} + {1}{2}-->{1} + {1}{2}-->{2} + {1}{2}-->{12} + {1}-->{1}{2} + {1}-->{1} + {1}-->{2} + {1}-->{12}:				        I(1(t);1(t+1),2(t+1))              H(1(t)) + H(1(t+1),2(t+1)) - H(1(t),1(t+1),2(t+1))
Iytab = h_p2 + h_t1t2 - h_p2t1t2;							  % {1}{2}-->{1}{2} + {1}{2}-->{1} + {1}{2}-->{2} + {1}{2}-->{12} + {2}-->{1}{2} + {2}-->{1} + {2}-->{2} + {2}-->{12}:					  I(2(t);1(t+1),2(t+1))              H(2(t)) + H(1(t+1),2(t+1)) - H(2(t),1(t+1),2(t+1))

Rxyta  = RedFun(sX, p1, p2, t1, Ixta, Iyta, Ixyta);			   % {1}{2}-->{1}{2} + {1}{2}-->{1}:																													MMI: min of I(1(t);1(t+1))               & I(2(t);1(t+1))                 CCS: I(1(t),1(t+1);2(t),2(t+1)) - I(1(t);1(t+1),2(t+1)) - I(2(t);1(t+1),2(t+1))
Rxytb  = RedFun(sX, p1, p2, t2, Ixtb, Iytb, Ixytb);			   % {1}{2}-->{1}{2} + {1}{2}-->{2}:																													MMI: min of I(2(t);2(t+1))               & I(1(t);2(t+1))                 CCS: I(1(t),1(t+1);2(t),2(t+1)) - I(1(t),2(t);1(t+1)) - I(1(t),2(t);2(t+1))
Rxytab = RedFun(sX, p1, p2, [t1 t2], Ixtab, Iytab, Ixytab);	    % {1}{2}-->{1}{2} + {1}{2}-->{1} + {1}{2}-->{2} + {1}{2}-->{12}:																				MMI: min of I(1(t);1(t+1),2(t+1))     & I(2(t);1(t+1),2(t+1))     CCS: I(1(t),2(t);1(t+1)) - I(1(t);1(t+1)) - I(2(t);1(t+1))
Rabtx  = RedFun(sX, t1, t2, p1, Ixta, Ixtb, Ixtab);			    % {1}{2}-->{1}{2} + {1}-->{1}{2}:																													MMI: min of I(1(t);1(t+1))                & I(1(t);2(t+1))                CCS: I(1(t),2(t);2(t+1)) - I(2(t);2(t+1)) - I(1(t);2(t+1))
Rabty  = RedFun(sX, t1, t2, p2, Iyta, Iytb, Iytab);			    % {1}{2}-->{1}{2} + {2}-->{1}{2}:																													MMI: min of I(2(t);1(t+1))                & I(2(t);2(t+1))                CCS: I(1(t);1(t+1),2(t+1)) - I(1(t);1(t+1)) - I(1(t);2(t+1))
Rabtxy = RedFun(sX, t1, t2, [p1 p2], Ixyta, Ixytb, Ixytab);	    % {1}{2}-->{1}{2} + {1}-->{1}{2} + {2}-->{1}{2} + {12}-->{1}{2}:																				MMI: min of I(1(t),2(t);1(t+1))         & I(1(t),2(t);2(t+1))          CCS: I(2(t);1(t+1),2(t+1)) - I(2(t);1(t+1)) - I(2(t);2(t+1))



% Compute double-redundancy (the remaining PhiID quantity, and, in this case, PhiID atom to compute) with corresponding function
[~, rtr] = DoubleRedFun(sX(p1,:), sX(p2,:), sX(t1,:), sX(t2,:));     % MMI: min of MI between

                                                                                                  % partition1 at t & partition1 at t+1
                                                                                                  % partition1 at t & partition2 at t+1
                                                                                                  % partition2 at t & partition1 at t+1
                                                                                                  % partition2 at t & partition2 at t+1
                                                                                                  
                                                                                                  % CCS: calculating co-info: double_coinfo (redred - synsyn) = - Ixta - Ixtb - Iyta - Iytb + ...
                                                                                                  %                                                                 Ixtab + Iytab + Ixyta + Ixytb - Ixytab + ...
                                                                                                  %                                                                 + Rxyta + Rxytb - Rxytab + ...
                                                                                                  %                                                                 + Rabtx + Rabty - Rabtxy;
                                                                                                  %                                                                 signs = [sign(Ixta), sign(Ixtb), sign(Iyta), sign(Iytb), sign(double_coinfo)];
                                                                                                  %                                                                 redred = all(signs == signs(:,1), 2).*double_coinfo;

% Assemble and solve system of equations
% PhiID atoms:
reds = [rtr Rxyta Rxytb Rxytab Rabtx Rabty Rabtxy ...
        Ixta Ixtb Iyta Iytb Ixyta Ixytb Ixtab Iytab Ixytab];

% rtr:  {1}{2}-->{1}{2}
% rtx: {1}{2}-->{1}
% rty: {1}{2}-->{2}
% rts: {1}{2}-->{12}
% xtr: {1}-->{1}{2}
% xtx: {1}-->{1}
% xty: {1}-->{2} 
% xts: {1}-->{12}1 1 1 1 1 1 1 1
% ytr: {2}-->{1}{2}
% ytx: {2}-->{1}
% yty: {2}-->{2}
% yts: {2}-->{12} 
% str: {12}-->{1}{2}
% stx: {12}-->{1} 
% sty: {12}-->{2} 
% sts: {12}-->{12}  

% matrix M: each row corresponds to one of the 16 PhiID quantities; each column corresponds to one PhiID atom, thus, the rows indicates
% whether or not that atom is part of the respective PhiID quantity)
M = [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;	% rtr
	1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;	% Rxyta:		 {1}{2}-->{1}{2} + {1}{2}-->{1};																											MMI: min of I(1(t);1(t+1)) & I(2(t);1(t+1))					CCS: I(1(t),1(t+1);2(t),2(t+1)) - I(1(t);1(t+1),2(t+1)) - I(2(t);1(t+1),2(t+1))
	1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0;	% Rxytb:		 {1}{2}-->{1}{2} + {1}{2}-->{2};																											MMI: min of I(2(t);2(t+1)) & I(1(t);2(t+1))					CCS: I(1(t),1(t+1);2(t),2(t+1)) - I(1(t),2(t);1(t+1)) - I(1(t),2(t);2(t+1))
	1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0;	% Rxytab:		{1}{2}-->{1}{2} + {1}{2}-->{1} + {1}{2}-->{2} + {1}{2}-->{12};																		MMI: min of I(1(t);1(t+1),2(t+1)) & I(2(t);1(t+1),2(t+1))	  CCS: I(1(t),2(t);1(t+1)) - I(1(t);1(t+1)) - I(2(t);1(t+1))
	1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0;	% Rabtx:		 {1}{2}-->{1}{2} + {1}-->{1}{2};																											MMI: min of I(1(t);1(t+1)) & I(1(t);2(t+1))					CCS: I(1(t),2(t);2(t+1)) - I(2(t);2(t+1)) - I(1(t);2(t+1))
	1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0;	% Rabty:		 {1}{2}-->{1}{2} + {2}-->{1}{2};																											MMI: min of I(2(t);1(t+1)) & I(2(t);2(t+1))					CCS: I(1(t);1(t+1),2(t+1)) - I(1(t);1(t+1)) - I(1(t);2(t+1))
     1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0;	% Rabtxy:		{1}{2}-->{1}{2} + {1}-->{1}{2} + {2}-->{1}{2} + {12}-->{1}{2};																	     MMI: min of I(1(t),2(t);1(t+1)) & I(1(t),2(t);2(t+1))			CCS: I(2(t);1(t+1),2(t+1)) - I(2(t);1(t+1)) - I(2(t);2(t+1))
     1 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0;	% Ixta:		   {1}{2}-->{1}{2} + {1}{2}-->{1} + {1}-->{1}{2} + {1}-->{1};																			I(1(t);1(t+1)) ✓         
     1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0;	% Ixtb:		   {1}{2}-->{1}{2} + {1}{2}-->{2} + {1}-->{1}{2} + {1}-->{2};																			I(1(t);2(t+1)) ✓
     1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0;	% Iyta:		   {1}{2}-->{1}{2} + {1}{2}-->{1} + {2}-->{1}{2} + {2}-->{1};																			I(2(t);1(t+1)) ✓
     1 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0;	% Iytb:		   {1}{2}-->{1}{2} + {1}{2}-->{2} + {2}-->{1}{2} + {2}-->{2};																			I(2(t);2(t+1)) ✓        
     1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0;	% Ixyta:		  {1}{2}-->{1}{2} + {1}{2}-->{1} + {1}-->{1}{2} + {1}-->{1} + {2}-->{1}{2} + {2}-->{1} + {12}-->{1}{2} + {12}-->{1};			I(1(t),2(t);1(t+1)) ✓
     1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0;	% Ixytb:		  {1}{2}-->{1}{2} + {1}{2}-->{2} + {1}-->{1}{2} + {1}-->{2}+ {2}-->{1}{2} + {2}-->{2} + {12}-->{1}{2} + {12}-->{2};			I(1(t),2(t);2(t+1)) ✓
     1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0;	% Ixtab:		  {1}{2}-->{1}{2} + {1}{2}-->{1} + {1}{2}-->{2} + {1}{2}-->{12} + {1}-->{1}{2} + {1}-->{1} + {1}-->{2} + {1}-->{12};			I(1(t);1(t+1),2(t+1)) ✓
     1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0;	% Iytab:		  {1}{2}-->{1}{2} + {1}{2}-->{1} + {1}{2}-->{2} + {1}{2}-->{12} + {2}-->{1}{2} + {2}-->{1} + {2}-->{2} + {2}-->{12};			I(2(t);1(t+1),2(t+1)) ✓
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];	% Ixytab:		 all 16 atoms                                                                                                                                                                                                        I(1(t),1(t+1);2(t),2(t+1))

partials = linsolve(M, reds');     % solve system of linear equations: M * X = reds (16*16 x 16*time-steps = 16*time-steps; result gives local PhiID atoms)


% Sort the results and return
atoms = [];
atoms.rtr = partials(1,:);
atoms.rtx = partials(2,:);
atoms.rty = partials(3,:);
atoms.rts = partials(4,:);
atoms.xtr = partials(5,:);
atoms.xtx = partials(6,:);
atoms.xty = partials(7,:);
atoms.xts = partials(8,:);
atoms.ytr = partials(9,:);
atoms.ytx = partials(10,:);
atoms.yty = partials(11,:);
atoms.yts = partials(12,:);
atoms.str = partials(13,:);
atoms.stx = partials(14,:);
atoms.sty = partials(15,:);
atoms.sts = partials(16,:);

%alignComments()
end


%*********************************************************
% Utility functions to compute basic information-theoretic measures
%*********************************************************

% logarithm of the determinant of matrix A
function [ res ] = logdet(A)
  res = 2*sum(log(diag(chol(A))));
end

% joint entropy of a multivariate normal distribution with covariance S
function [ res ] = h(S, idx)
  res = 0.5*length(idx)*log(2*pi*exp(1)) + 0.5*logdet(S(idx,idx));
end

% mutual information between two (groups of) variables
function [ res ] = mi(S, src, tgt)
  res = h(S, src) + h(S, tgt) - h(S, [src, tgt]);
end

%*********************************************************
% A few PID (single-target) redundancy functions
%*********************************************************
function [ R ] = RedundancyMMI(bX, src1, src2, tgt, mi1, mi2, mi12)
  if mean(mi1) < mean(mi2)
    R = mi1;
  else
    R = mi2;
  end
end


function [ R ] = RedundancyCCS(S, src1, src2, tgt, mi1, mi2, mi12)

  c = mi12 - mi1 - mi2;                                         % co-information (should be equivalent to interaction information)
  signs = [sign(mi1), sign(mi2), sign(mi12), sign(-c)];
  R = all(signs == signs(:,1), 2).*(-c);                    % all() determines if the elements are all nonzero or logical 1 (true)
                                                                             % signs == signs(:,1) --> creates new matrix indicating whether columns in signs are equal to the first column of signs (all elements of first column in the new matrix would trivially be 1)
                                                                             % all(signs == signs(:,1), 2) --> determines whether all elements of a given row in the new matrix are nonzero or 1 

end

