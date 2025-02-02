function redred = DoubleRedundancyCCS_Analytical(varargin)
%%DOUBLEREDUNDANCYMMI_ANALYTICAL Compute the PhiID double-redundancy given 
% a full time-lagged covariance matrix, assuming it follows a Gaussian 
% distribution and using the CCS PID.
%
% NOTE: assumes JIDT has been already added to the javaclasspath.
%
%   R = DOUBLEREDUNDANCYCCS_ANALYTICAL(full_time_lagged_cov, p1, p2), where
%   full_time_lagged_cov is a 2xD-by-2xD full time-lagged covariance matrix,  
%   and p1 the row/column indices of the covariance of partition 1 at t, and
%   p2 the row/column indices of the covariance of partition 2 at t. 
%
% Note: some function/variable names (private_FourVectorCCS(), private_TDCCS(),
% localred()) come from PhiIDFull_Continuous/Discrete where data is used as 
% input, and quantities have been first calculated locally, and then on average
% - this is not the case here, where, if using the full time-lagged covariance
% matrix, we can only look at averages. Should be changed at some point.
% 
% Adapted from:
%   Mediano*, Rosas*, Carhart-Harris, Seth and Barrett (2019). Beyond
%   Integrated Information: A Taxonomy of Information Dynamics Phenomena.
%
% Nadine Spychala, Sep 2022

	redred = private_FourVectorCCS(varargin{1}, varargin{2}, varargin{3});
end


%*********************************************************
%*********************************************************
function [ redred ] = private_TDCCS(full_time_lagged_cov, p1, p2)
	
	% Argument checks and parameter initialisation
	if isempty(full_time_lagged_cov) || ~ismatrix(full_time_lagged_cov)
		error('Input covariances must be 2D matrices');
	end
	
	[D, T] = size(full_time_lagged_cov);
	if T < D || D > T
		error(sprintf(['full_time_lagged_cov has %i rows and %i columns. ', ...
			'If this is true, this is not a square matrix. '], D, T));
	end
	
	if nargin < 2 || isempty(time_lag)
		time_lag = 1;
	end
	
	integer_time_lag = ~isinf(time_lag) & floor(time_lag) == time_lag;
	if ~integer_time_lag || time_lag < 1
		error('time-lag needs to be a positive integer.');
	end
	
	redred = private_FourVectorCCS(full_time_lagged_cov, p1, p2);
end


%*********************************************************
%*********************************************************
function [ redred ] = private_FourVectorCCS(full_time_lagged_cov, p1, p2)

	% Argument checks and parameter initialisation
	checkmat = @(v) ~isempty(v) && ismatrix(v);
	if ~(checkmat(full_time_lagged_cov))
		error('Inputs must be non-empty matrices');
	end
	
	if nargin < 5 || isempty(measure)
		measure = 'MMI';
	end
	
	if strcmp(lower(measure), 'ccs')
		RedFun = @RedundancyCCS;
		DoubleRedFun = @DoubleRedundancyCCS;
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

	% Define local information-theoretic functions: analytical solution for average 
	% multivariate entropy using the covariance matrix
	h = @(idx) 0.5 * size(full_time_lagged_cov(idx, idx),1) * log(2 * pi * exp(1)) + ...
		0.5 * log(det(full_time_lagged_cov(idx, idx)));
	
	% Pre-compute entropies necessary for all IT quantities
	h_p1 = h(p1);                      % entropy of partition 1 at t           H(1(t))
	h_p2 = h(p2);                      % entropy pf partiiton 2 at t           H(2(t))
	h_t1 = h(t1);                      % entropy of partition 1 at t+1         H(1(t+1))
	h_t2 = h(t2);                      % entropy of partition 2 at t+1         H(2(t+1))
	
	h_p1p2 = h([p1 p2]);               % multivariate entropy (ME) of partition 1 & 2 at t         H(1(t),   2(t))
	h_t1t2 = h([t1 t2]);               % ME of partition 1 & 2 at t+1                              H(1(t+1), 2(t+1))
	h_p1t1 = h([p1 t1]);               % ME of partition 1 at t & t+1                              H(1(t),   1(t+1))
	h_p1t2 = h([p1 t2]);               % ME of partition 1 at t & partition 2 at t+1               H(1(t),   2(t+1))
	h_p2t1 = h([p2 t1]);               % ME of partition 2 at t & partition 1 at t+1               H(2(t),   1(t+1))
	h_p2t2 = h([p2 t2]);               % ME of partition 2 at t & t+1                              H(2(t),   2(t+1))
	
	h_p1p2t1 = h([p1 p2 t1]);          % ME of partition 1 & 2 at t & partition 1 at t+1           H(1(t),   2(t),     1(t+1))
	h_p1p2t2 = h([p1 p2 t2]);          % ME of partition 1 & 2 at t & partition 2 at t+1           H(1(t),   2(t),     2(t+1))
	h_p1t1t2 = h([p1 t1 t2]);          % ME of partition 1 at t & t+1 & partition 2 at t+1         H(1(t),   1(t+1),   2(t+1))
	h_p2t1t2 = h([p2 t1 t2]);          % ME of partition 2 at t & t+1 & partition 1 at t           H(2(t),   2(t+1),   1(t+1))
	
	h_p1p2t1t2 = h([p1 p2 t1 t2]);     % ME of partition 2 at t & t+1 & partition 1 at t & t+1     H(2(t),   2(t+1),   1(t),     2(t+1))
	
	% Compute PhiID quantities as entropy combinations
	
	% Ixytab: all 16 atoms (this is what we're trying to decompose)
	Ixytab = h_p1p2 + h_t1t2 - h_p1p2t1t2;          % I(1(t),1(t+1);2(t),2(t+1))    H(1(t),2(t)) + H(1(t+1),2(t+1)) - H(1(t),1(t+1),2(t),2(t+1))          
	
	Ixta = h_p1 + h_t1 - h_p1t1;                    % I(1(t);1(t+1))                H(1(t)) + H(1(t+1)) - H(1(t),1(t+1))
	Ixtb = h_p1 + h_t2 - h_p1t2;                    % I(1(t);2(t+1))                H(1(t)) + H(2(t+1)) - H(1(t),2(t+1))
	Iyta = h_p2 + h_t1 - h_p2t1;                    % I(2(t);1(t+1))                H(2(t)) + H(1(t+1)) - H(2(t),1(t+1))
	Iytb = h_p2 + h_t2 - h_p2t2;                    % I(2(t);2(t+1))                H(2(t)) + H(2(t+1)) - H(2(t),2(t+1))
	
	Ixyta = h_p1p2 + h_t1 - h_p1p2t1;               % I(1(t),2(t);1(t+1))           H(1(t),2(t)) + H(1(t+1))        - H(1(t),2(t),1(t+1))
	Ixytb = h_p1p2 + h_t2 - h_p1p2t2;               % I(1(t),2(t);2(t+1))           H(1(t),2(t)) + H(2(t+1))        - H(1(t),2(t),2(t+1))
	Ixtab = h_p1 + h_t1t2 - h_p1t1t2;               % I(1(t);1(t+1),2(t+1))         H(1(t))      + H(1(t+1),2(t+1)) - H(1(t),1(t+1),2(t+1))
	Iytab = h_p2 + h_t1t2 - h_p2t1t2;               % I(2(t);1(t+1),2(t+1))         H(2(t))      + H(1(t+1),2(t+1)) - H(2(t),1(t+1),2(t+1))
	
	Rxytab = localred(Ixtab, Iytab, Ixytab);        % I(1(t),2(t);1(t+1))          - I(2(t);1(t+1))        - I(1(t);1(t+1))
	Rabtxy = localred(Ixyta, Ixytb, Ixytab);        % I(1(t),2(t);2(t+1))          - I(1(t);2(t+1))	       - I(2(t);2(t+1))
	Rxyta  = localred(Ixta, Iyta, Ixyta);           % I(1(t),1(t+1);2(t),2(t+1))   - I(2(t);1(t+1),2(t+1)) - I(1(t);1(t+1),2(t+1))
	Rxytb  = localred(Ixtb, Iytb, Ixytb);           % I(1(t);1(t+1),2(t+1))	       - I(1(t);1(t+1))	       - I(1(t);2(t+1))
	Rabtx  = localred(Ixta, Ixtb, Ixtab);           % I(2(t);1(t+1),2(t+1))	       - I(2(t);1(t+1))	       - I(2(t);2(t+1))
	Rabty  = localred(Iyta, Iytb, Iytab);           % I(1(t),1(t+1);2(t),2(t+1))   - I(1(t),2(t);1(t+1))   - I(1(t),2(t);2(t+1))
	
	% This quantity equals Double-Red - Double-Syn 
	double_coinfo = - Ixta - Ixtb - Iyta - Iytb + ...
		+ Ixtab + Iytab + Ixyta + Ixytb - Ixytab + ...
		+ Rxyta + Rxytb - Rxytab + ...
		+ Rabtx + Rabty - Rabtxy;
	
	% double_coinfo above is equal to:
	%	   - I(1(t);1(t+1)) - I(1(t);2(t+1)) - I(2(t);1(t+1)) - I(2(t);2(t+1))
	%	   + I(1(t);1(t+1),2(t+1)) + I(2(t);1(t+1),2(t+1)) + I(1(t),2(t);1(t+1)) + I(1(t),2(t);2(t+1)) - I(1(t),1(t+1);2(t),2(t+1))
	%	   + a bunch of redundancy terms derived for a univariate target
	
	
	signs = [sign(Ixta), sign(Ixtb), sign(Iyta), sign(Iytb), sign(double_coinfo)];
	% I(1(t);1(t+1)), I(1(t);2(t+1)), I(2(t);1(t+1)), I(2(t);2(t+1)), redred - synsyn
	redred = all(signs == signs(:,1), 2).*double_coinfo;

end

function [ l ] = localred(mi1, mi2, mi12)
  c = mi12 - mi1 - mi2;
  signs = [sign(mi1), sign(mi2), sign(mi12), sign(-c)];
  l = all(signs == signs(:,1), 2).*(-c);
end

