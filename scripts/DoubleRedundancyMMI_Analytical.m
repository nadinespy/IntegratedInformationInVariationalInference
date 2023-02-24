function redred = DoubleRedundancyMMI_Analytical(varargin)
%%DOUBLEREDUNDANCYMMI Compute the PhiID double-redundancy given 
% a full time-lagged covariance matrix, assuming it follows a Gaussian 
% distribution and using the MMI PID.
%
% NOTE: assumes JIDT has been already added to the javaclasspath.
%
%   R = DOUBLEREDUNDANCYMMI(full_time_lagged_cov, p1, p2), where
%   full_time_lagged_cov is a 2xD-by-2xD full time-lagged covariance matrix,  
%   and p1 the row/column indices of the covariance of partition 1 at t, and
%   p2 the row/column indices of the covariance of partition 2 at t. 
%
% Adapted from:
%   Mediano*, Rosas*, Carhart-Harris, Seth and Barrett (2019). Beyond
%   Integrated Information: A Taxonomy of Information Dynamics Phenomena.
%
% Nadine Spychala, Sep 2022

	redred = private_FourVectorMMI(varargin{1}, varargin{2}, varargin{3});

end


%*********************************************************
%*********************************************************
function [ redred ] = private_TDMMI(full_time_lagged_cov, p1, p2)

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
	
	redred = private_FourVectorMMI(full_time_lagged_cov, p1, p2);

end


%*********************************************************
%*********************************************************
function [ redred ] = private_FourVectorMMI(full_time_lagged_cov, p1, p2)

	% Argument checks and parameter initialisation
	if isempty(full_time_lagged_cov) || ~ismatrix(full_time_lagged_cov)
		error('Input covariances must be 2D matrices');
	end
	
	[D, T] = size(full_time_lagged_cov);
	if T < D || D > T
		error(sprintf(['full_time_lagged_cov has %i rows and %i columns. ', ...
			'If this is true, this is not a square matrix. '], D, T));
	end
	
	n_vars = 1:size(full_time_lagged_cov, 1)/2;
	p2 = setdiff(n_vars, p1);				% row/column indices for cov of partition 2 at t
	t1 = p1 + size(full_time_lagged_cov, 1)/2;	% row/column indices for cov of partition 1 at t+1
	t2 = p2 + size(full_time_lagged_cov, 1)/2;	% row/column indices for cov of partition 2 at t+1
	% p1: row/column indices for cov of partition 1 at t
	
	src = {p1, p2};
	tgt = {t1, t2};

	% Take double-redundancy as the minimum MI between either src or tgt
	redred = 1;  % Set to a large, but finite value
	
	for i=1:length(src)
		
		for j=1:length(tgt)
			
			s = src{i};
			t = tgt{j};
			
			% cond_cov_present_past = cov_past - (cov_present_past' / cov_present) * cov_present_past;
			% 			    = full_time_lagged_cov(t1,t1) - (full_time_lagged_cov(p1,t1)' / full_time_lagged_cov(p1,p1)) * full_time_lagged_cov(p1,t1);
			
			cond_cov = @(idx1, idx2) full_time_lagged_cov(idx2,idx2) - (full_time_lagged_cov(idx1,idx2)' / full_time_lagged_cov(idx1,idx1)) * full_time_lagged_cov(idx1,idx2);
			cond_cov_temp = cond_cov(s, t);
			
			cov = @(idx) full_time_lagged_cov(s, s);
			cov_temp = cov(s);
			
			mi = 0.5 * log(det(cov_temp) / det(cond_cov_temp));
			
			if mi < redred
				redred = mi;
			end
			
		end
	end
end