function atoms = PhiIDFull_Analytical(varargin)
%%PHIIDFULL Computes full PhiID decomposition given a full time-lagged 
% covariance matrix, assuming it follows a multivariate Gaussian distribution.
%
%   A = PHIIDFULL(full_time_lagged_covarianc), where 
%   full_time_lagged_cov is a 2xD-by-2xD full time-lagged covariance matrix, 
%   computes the PhiID decomposition of the time-delayed mutual information 
%   of full_time_lagged_cov. If D > 2, PhiID is calculated across the 
%   minimum information bipartition (MIB) of the system.
%
%   A = PHIIDFULL(..., measure) uses PhiID measure M (default: 'MMI')
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
% Note: some function/variable names (private_FourVectorMPhiID(), private_TDPhiID()) 
% come from PhiIDFull_Continuous/Discrete where data is used as input, and 
% quantities have been first calculated locally, and then on average -
% this is not the case here, where, if using the full time-lagged covariance
% matrix, we can only look at averages. Should be changed at some point.
%
% Code adapted from:
%   Mediano*, Rosas*, Carhart-Harris, Seth and Barrett (2019). Beyond
%   Integrated Information: A Taxonomy of Information Dynamics Phenomena.
%
% Nadine Spychala, Sep 2022

	% Find JIDT and add relevant paths; calculate average PhiID atoms
	p = strrep(mfilename('fullpath'), 'PhiIDFull', '');
	if exist([p, '../elph_base'], 'dir')
		addpath([p, '../elph_base']);
	end
% 	if ~any(~cellfun('isempty', strfind(javaclasspath('-all'), 'infodynamics')))
% 		if exist([p, '../elph_base/infodynamics.jar'], 'file')
% 			javaaddpath([p, '../elph_base/infodynamics.jar']);
% 		elseif exist([p, 'private/infodynamics.jar'], 'file')
% 			javaaddpath([p, 'private/infodynamics.jar']);
% 		else
% 			error('Unable to find JIDT (infodynamics.jar).');
% 		end
% 	end

	if nargin == 1
		atoms = private_TDPhiID(varargin{1});
	elseif nargin == 2
		atoms = private_TDPhiID(varargin{1}, varargin{2});
	else
		error('Wrong number of arguments. See `help PhiIDFull_Analytical` for help.');
	end

end


%*********************************************************
%*********************************************************
function [ atoms ] = private_TDPhiID(full_time_lagged_cov, measure)

	% Argument checks and parameter initialisation
	if isempty(full_time_lagged_cov) || ~ismatrix(full_time_lagged_cov)
		error('Input covariances must be 2D matrices');
	end
	
	[D, T] = size(full_time_lagged_cov);
	if T < D || D > T
		error(sprintf(['full_time_lagged_cov has %i rows and %i columns. ', ...
			'If this is true, this is not a square matrix. '], D, T));
	end
	
	if nargin < 2 || isempty(measure)
		measure = 'MMI';
	end
	
	index_half_cov = size(full_time_lagged_cov,1)/2;
	cov_present = full_time_lagged_cov(1:index_half_cov, 1:index_half_cov);
	cov_present_past = full_time_lagged_cov(1:index_half_cov, index_half_cov+1:end);
	cov_past = full_time_lagged_cov(index_half_cov+1:end, index_half_cov+1:end);
	
	phi = get_phi_from_cov(cov_present, cov_past, cov_present_past);
	p1 = phi.Partition1;
	p2 = phi.Partition2;
	
	% Call full PhiID function
	atoms = private_FourVectorPhiID(full_time_lagged_cov, p1, p2, measure);
end


%*********************************************************
%*********************************************************

% returns local PhiID atoms
function [ atoms ] = private_FourVectorPhiID(full_time_lagged_cov, ...
		p1, p2, measure)
	
	% Argument checks and parameter initialisation
	checkmat = @(v) ~isempty(v) && ismatrix(v);
	if ~(checkmat(full_time_lagged_cov))
		error('Inputs must be non-empty matrices');
	end
	
	if nargin < 4 || isempty(measure)
		measure = 'MMI';
	end
	
	if strcmp(lower(measure), 'ccs')
		RedFun = @RedundancyCCS;
		DoubleRedFun = @DoubleRedundancyCCS_Analytical;
	elseif strcmp(lower(measure), 'mmi')
		RedFun = @RedundancyMMI;
		DoubleRedFun = @DoubleRedundancyMMI_Analytical;
	else
		error(['Unknown redundancy measure. Currently implemented measures are ''CCS'' and ''MMI''']);
	end
	
	[D, T] = size(full_time_lagged_cov);
	
	n_vars = 1:size(full_time_lagged_cov, 1)/2;
	p2 = setdiff(n_vars,p1);				% row/column indices for partition 2 at t
	t1 = p1 + size(full_time_lagged_cov, 1)/2;	% row/column indices for partition 1 at t+1
	t2 = p2 + size(full_time_lagged_cov, 1)/2;	% row/column indices for partition 2 at t+1
									% p1: row/column indices for partition 1 at t
	
	% Define local information-theoretic functions
	
	% % multivariate entropy, h() takes as an input the indices of the variables to consider in sX
	% h = @(idx) -log(mvnpdf(sX(idx,:)', means(idx), full_time_lagged_cov(idx,idx)));
	% % mutual information (I(X;Y) = H(X) + H(Y) - H(X,Y)) (not further used below)
	% mi  = @(src, tgt) h(src) + h(tgt) - h([src, tgt]);
	
	% analytical solution for average multivariate entropy using the covariance matrix:
	h = @(idx) 0.5 * size(full_time_lagged_cov(idx, idx),1) * log(2 * pi * exp(1)) + ...
		0.5 * log(det(full_time_lagged_cov(idx, idx)));
	
	% Pre-compute entropies necessary for all IT quantities (all quantities
	% have as many rows as time-steps in the time-series)
	h_p1 = h(p1);				% entropy of partition 1 at t					   H(1(t))
	h_p2 = h(p2);				% entropy pf partiiton 2 at t					   H(2(t))
	h_t1 = h(t1);				% entropy of partition 1 at t+1				   H(1(t+1))
	h_t2 = h(t2);				% entropy of partition 2 at t+1				   H(2(t+1))
	
	h_p1p2 = h([p1 p2]);			% multivariate entropy (ME) of partition 1 & 2 at t      H(1(t),      2(t))
	h_t1t2 = h([t1 t2]);			% ME of partition 1 & 2 at t+1                           H(1(t+1),    2(t+1))
	h_p1t1 = h([p1 t1]);			% ME of partition 1 at t & t+1                           H(1(t),      1(t+1))
	h_p1t2 = h([p1 t2]);			% ME of partition 1 at t & partition 2 at t+1            H(1(t),      2(t+1))
	h_p2t1 = h([p2 t1]);			% ME of partition 2 at t & partition 1 at t+1            H(2(t),      1(t+1))
	h_p2t2 = h([p2 t2]);			% ME of partition 2 at t & t+1                           H(2(t),      2(t+1))
	
	h_p1p2t1 = h([p1 p2 t1]);		% ME of partition 1 & 2 at t & partition 1 at t+1        H(1(t),      2(t),       1(t+1))
	h_p1p2t2 = h([p1 p2 t2]);		% ME of partition 1 & 2 at t & partition 2 at t+1        H(1(t),      2(t),       2(t+1))
	h_p1t1t2 = h([p1 t1 t2]);		% ME of partition 1 at t & t+1 & partition 2 at t+1      H(1(t),      1(t+1),     2(t+1))
	h_p2t1t2 = h([p2 t1 t2]);		% ME of partition 2 at t & t+1 & partition 1 at t        H(2(t),      2(t+1),     1(t+1))
	
	h_p1p2t1t2 = h([p1 p2 t1 t2]);	% ME of partition 2 at t & t+1 & partition 1 at t & t+1  H(2(t),      2(t+1),     1(t),       (t+1))
	
	% Compute local PhiID quantities as entropy combinations (all quantities have as many rows as time-steps in the time-series)
	% variable names:
	% R: redundant information
	% I: mutual information
	% x: source variable 1
	% y: source variable 2
	% a: target variable 1
	% b: target variable 2
	
	
	Ixytab = h_p1p2 + h_t1t2 - h_p1p2t1t2;	% all 16 atoms (this is what we're trying to decompose): I(1(t),1(t+1);2(t),2(t+1))      H(1(t),2(t)) + H(1(t+1),2(t+1)) - H(1(t),1(t+1),2(t),2(t+1))
	
	Ixta = h_p1 + h_t1 - h_p1t1;			% {1}{2}-->{1}{2} + {1}{2}-->{1} + {1}-->{1}{2} + {1}-->{1}:										I(1(t);1(t+1))                     H(1(t))      + H(1(t+1))       - H(1(t),1(t+1))
	Ixtb = h_p1 + h_t2 - h_p1t2;			% {1}{2}-->{1}{2} + {1}{2}-->{2} + {1}-->{1}{2} + {1}-->{2}:										I(1(t);2(t+1))                     H(1(t))      + H(2(t+1))       - H(1(t),2(t+1))
	Iyta = h_p2 + h_t1 - h_p2t1;			% {1}{2}-->{1}{2} + {1}{2}-->{1} + {2}-->{1}{2} + {2}-->{1}:										I(2(t);1(t+1))                     H(2(t))      + H(1(t+1))       - H(2(t),1(t+1))
	Iytb = h_p2 + h_t2 - h_p2t2;			% {1}{2}-->{1}{2} + {1}{2}-->{2} + {1}-->{1}{2} + {1}-->{2}+ {2}-->{1}{2} + {2}-->{2} + {12}-->{1}{2} + {12}-->{2}:	I(2(t);2(t+1))                     H(2(t))      + H(2(t+1))       - H(2(t),2(t+1))
	
	Ixyta = h_p1p2 + h_t1 - h_p1p2t1;		% {1}{2}-->{1}{2} + {1}{2}-->{1} + {1}-->{1}{2} + {1}-->{1} + {2}-->{1}{2} + {2}-->{1} + {12}-->{1}{2} + {12}-->{1}:	I(1(t),2(t);1(t+1))                H(1(t),2(t)) + H(1(t+1))        - H(1(t),2(t),1(t+1))
	Ixytb = h_p1p2 + h_t2 - h_p1p2t2;		% {1}{2}-->{1}{2} + {1}{2}-->{2} + {1}-->{1}{2} + {1}-->{2}+ {2}-->{1}{2} + {2}-->{2} + {12}-->{1}{2} + {12}-->{2}:	I(1(t),2(t);2(t+1))                H(1(t),2(t)) + H(2(t+1))        - H(1(t),2(t),2(t+1))
	Ixtab = h_p1 + h_t1t2 - h_p1t1t2;		% {1}{2}-->{1}{2} + {1}{2}-->{1} + {1}{2}-->{2} + {1}{2}-->{12} + {1}-->{1}{2} + {1}-->{1} + {1}-->{2} + {1}-->{12}:	I(1(t);1(t+1),2(t+1))              H(1(t))      + H(1(t+1),2(t+1)) - H(1(t),1(t+1),2(t+1))
	Iytab = h_p2 + h_t1t2 - h_p2t1t2;		% {1}{2}-->{1}{2} + {1}{2}-->{1} + {1}{2}-->{2} + {1}{2}-->{12} + {2}-->{1}{2} + {2}-->{1} + {2}-->{2} + {2}-->{12}:	I(2(t);1(t+1),2(t+1))              H(2(t))      + H(1(t+1),2(t+1)) - H(2(t),1(t+1),2(t+1))
	
	Rxyta  = RedFun(full_time_lagged_cov, p1, p2, t1, Ixta, Iyta, Ixyta);			% {1}{2}-->{1}{2} + {1}{2}-->{1}:								MMI: min of I(1(t);1(t+1))         & I(2(t);1(t+1))            CCS: I(1(t),2(t);1(t+1))        - I(2(t);1(t+1))        - I(1(t);1(t+1))
	Rxytb  = RedFun(full_time_lagged_cov, p1, p2, t2, Ixtb, Iytb, Ixytb);			% {1}{2}-->{1}{2} + {1}{2}-->{2}:								MMI: min of I(2(t);2(t+1))         & I(1(t);2(t+1))            CCS: I(1(t),2(t);2(t+1))        - I(1(t);2(t+1))	     - I(2(t);2(t+1))
	Rxytab = RedFun(full_time_lagged_cov, p1, p2, [t1 t2], Ixtab, Iytab, Ixytab);		% {1}{2}-->{1}{2} + {1}{2}-->{1} + {1}{2}-->{2} + {1}{2}-->{12}:			MMI: min of I(1(t);1(t+1),2(t+1))  & I(2(t);1(t+1),2(t+1))     CCS: I(1(t),1(t+1);2(t),2(t+1)) - I(2(t);1(t+1),2(t+1)) - I(1(t);1(t+1),2(t+1))
	Rabtx  = RedFun(full_time_lagged_cov, t1, t2, p1, Ixta, Ixtb, Ixtab);			% {1}{2}-->{1}{2} + {1}-->{1}{2}:								MMI: min of I(1(t);1(t+1))         & I(1(t);2(t+1))            CCS: I(1(t);1(t+1),2(t+1))	     - I(1(t);1(t+1))	     - I(1(t);2(t+1))
	Rabty  = RedFun(full_time_lagged_cov, t1, t2, p2, Iyta, Iytb, Iytab);			% {1}{2}-->{1}{2} + {2}-->{1}{2}:								MMI: min of I(2(t);1(t+1))         & I(2(t);2(t+1))            CCS: I(2(t);1(t+1),2(t+1))	     - I(2(t);1(t+1))	     - I(2(t);2(t+1))
	Rabtxy = RedFun(full_time_lagged_cov, t1, t2, [p1 p2], Ixyta, Ixytb, Ixytab);		% {1}{2}-->{1}{2} + {1}-->{1}{2} + {2}-->{1}{2} + {12}-->{1}{2}:			MMI: min of I(1(t),2(t);1(t+1))    & I(1(t),2(t);2(t+1))       CCS: I(1(t),1(t+1);2(t),2(t+1)) - I(1(t),2(t);1(t+1))   - I(1(t),2(t);2(t+1))
	
	% Compute double-redundancy (the remaining PhiID quantity, and, in this case, PhiID atom to compute) with corresponding function
	rtr = DoubleRedFun(full_time_lagged_cov, p1, p2);
	
	% MMI: min of MI between
	
	% partition1 at t & partition1 at t+1
	% partition1 at t & partition2 at t+1
	% partition2 at t & partition1 at t+1
	% partition2 at t & partition2 at t+1
	%
	% Example: we take the min between
	% I(X1(t),X2(t);X1(t+1),X2(t+1),
	% I(X3(t),X4(t);X1(t+1),X2(t+1),
	% I(X1(t),X2(t);X3(t+1),X4(t+1),
	% I(X3(t),X4(t);X3(t+1),X4(t+1),
	%
	% if for system X, [X1, X2] give one partition, and [X3, X4] give the other.
	%
	% CCS: calculating co-info (Double-Red - Double-Syn)
	% double_coinfo  = - Ixta - Ixtb - Iyta - Iytb + ...
	%                  Ixtab + Iytab + Ixyta + Ixytb - Ixytab + ...
	%                  + Rxyta + Rxytb - Rxytab + ...
	%                  + Rabtx + Rabty - Rabtxy;
	% signs = [sign(Ixta), sign(Ixtb), sign(Iyta), sign(Iytb), sign(double_coinfo)];
	% redred = all(signs == signs(:,1), 2).*double_coinfo;
	
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
		1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0;	% Rxytab:		 {1}{2}-->{1}{2} + {1}{2}-->{1} + {1}{2}-->{2} + {1}{2}-->{12};																		MMI: min of I(1(t);1(t+1),2(t+1)) & I(2(t);1(t+1),2(t+1))	  CCS: I(1(t),2(t);1(t+1)) - I(1(t);1(t+1)) - I(2(t);1(t+1))
		1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0;	% Rabtx:		 {1}{2}-->{1}{2} + {1}-->{1}{2};																											MMI: min of I(1(t);1(t+1)) & I(1(t);2(t+1))					CCS: I(1(t),2(t);2(t+1)) - I(2(t);2(t+1)) - I(1(t);2(t+1))
		1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0;	% Rabty:		 {1}{2}-->{1}{2} + {2}-->{1}{2};																											MMI: min of I(2(t);1(t+1)) & I(2(t);2(t+1))					CCS: I(1(t);1(t+1),2(t+1)) - I(1(t);1(t+1)) - I(1(t);2(t+1))
		1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0;	% Rabtxy:		 {1}{2}-->{1}{2} + {1}-->{1}{2} + {2}-->{1}{2} + {12}-->{1}{2};																	     MMI: min of I(1(t),2(t);1(t+1)) & I(1(t),2(t);2(t+1))			CCS: I(2(t);1(t+1),2(t+1)) - I(2(t);1(t+1)) - I(2(t);2(t+1))
		1 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0;	% Ixta:		 {1}{2}-->{1}{2} + {1}{2}-->{1} + {1}-->{1}{2} + {1}-->{1};																			I(1(t);1(t+1)) ✓
		1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0;	% Ixtb:		 {1}{2}-->{1}{2} + {1}{2}-->{2} + {1}-->{1}{2} + {1}-->{2};																			I(1(t);2(t+1)) ✓
		1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0;	% Iyta:		 {1}{2}-->{1}{2} + {1}{2}-->{1} + {2}-->{1}{2} + {2}-->{1};																			I(2(t);1(t+1)) ✓
		1 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0;	% Iytb:		 {1}{2}-->{1}{2} + {1}{2}-->{2} + {2}-->{1}{2} + {2}-->{2};																			I(2(t);2(t+1)) ✓
		1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0;	% Ixyta:		 {1}{2}-->{1}{2} + {1}{2}-->{1} + {1}-->{1}{2} + {1}-->{1} + {2}-->{1}{2} + {2}-->{1} + {12}-->{1}{2} + {12}-->{1};			I(1(t),2(t);1(t+1)) ✓
		1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0;	% Ixytb:		 {1}{2}-->{1}{2} + {1}{2}-->{2} + {1}-->{1}{2} + {1}-->{2}+ {2}-->{1}{2} + {2}-->{2} + {12}-->{1}{2} + {12}-->{2};			I(1(t),2(t);2(t+1)) ✓
		1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0;	% Ixtab:		 {1}{2}-->{1}{2} + {1}{2}-->{1} + {1}{2}-->{2} + {1}{2}-->{12} + {1}-->{1}{2} + {1}-->{1} + {1}-->{2} + {1}-->{12};			I(1(t);1(t+1),2(t+1)) ✓
		1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0;	% Iytab:		 {1}{2}-->{1}{2} + {1}{2}-->{1} + {1}{2}-->{2} + {1}{2}-->{12} + {2}-->{1}{2} + {2}-->{1} + {2}-->{2} + {2}-->{12};			I(2(t);1(t+1),2(t+1)) ✓
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

% joint entropy of a multivariate normal distribution with covariance full_time_lagged_cov
function [ res ] = h(full_time_lagged_cov, idx)
	res = 0.5*length(idx)*log(2*pi*exp(1)) + 0.5*logdet(full_time_lagged_cov(idx,idx));
end

% mutual information between two (groups of) variables
function [ res ] = mi(full_time_lagged_cov, src, tgt)
	res = h(full_time_lagged_cov, src) + h(full_time_lagged_cov, tgt) - h(full_time_lagged_cov, [src, tgt]);
end

%*********************************************************
% A few PID (single-target) redundancy functions
%*********************************************************
function [ R ] = RedundancyMMI(bX, src1, src2, tgt, mi1, mi2, mi12)
	
	if mi1 < mi2
		R = mi1;
	else
		R = mi2;
	end
	
end


function [ R ] = RedundancyCCS(full_time_lagged_cov, src1, src2, tgt, mi1, mi2, mi12)

	% Co-information (equivalent to interaction information): defined as
	% I(S1;T) + I(S2;T) - I(S1,S2;T) for sources S1, S2, and target T.
	% Negative values denote synergy, positive ones redundancy. We flip
	% signs for positive values to denote synergy such that
	% co-information = I(S1,S2;T) - I(S1;T) - I(S2;T)

	c = mi12 - mi1 - mi2;
  
	% store signs of all terms (sign of c is now flipped for positive 
	% values to indicate redundancy
	signs = [sign(mi1), sign(mi2), sign(mi12), sign(-c)];
	
	% all() determines if the elements are all nonzero or logical 1 
	% (true); 
	% signs == signs(:,1) --> creates new matrix indicating whether 
	% columns in signs are equal to the first column of signs (all 
	% (elements of first column in the new matrix would trivially be 1);
	% all(signs == signs(:,1), 2) --> determines whether all elements of
	% a given row in the new matrix are nonzero or 1 (yields column 
	% vector with ones and zeros);
	% multiply column vector with (-c) (yields zero for cases where signs 
	% had not been equal)  
	
	R = all(signs == signs(:,1), 2).*(-c);
end


