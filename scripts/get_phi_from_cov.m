function ret = get_phi_from_cov(cov_present, cov_past, cov_present_past)

	PhiNorm = 1000000;				% will be current minimum for normalized effective information
	nvar = size(cov_present,1);
	
	% get covariance of X_t-tau conditioned on X_t:
	% \Sigma_(X(t-1),X(t-1)) - \Sigma(X(t-1),X(t)) * \Sigma(X(t),X(t))^(-1) * \Sigma(X(t),X(t-1))
	cond_cov_present_past = cov_past - (cov_present_past' / cov_present) * cov_present_past;
	
	% get all possible bipartitions 
	bipartitions = [];
	myNums = [1:nvar];
	for i = 1:size(myNums,2)/2;
		combinationsSet{i} = nchoosek(myNums,i);
	end
	
	for g = 1:size(combinationsSet,2);
		blubb = combinationsSet{:,g};
		for f = 1:size(blubb,1)
			part1 = [blubb(f,:)];
			part2 = [setdiff(myNums, part1)];
			bipartition = {part1, part2};
			bipartitions = [bipartitions; bipartition];
		end
	end
	
	for g = 1:size(myNums,2)/(floor(nvar/2));
		
		blubb = g;
		for q = (blubb + 1):nvar
			part1 = [blubb, q];
			part2 = setdiff(myNums, part1);
			bipartition = {part1, part2};
			bipartitions = [bipartitions; bipartition];
		end
	end
	
	% get phi for all bipartitions
	for w = 1:length(bipartitions)
		
		% disp(w);
		
		part1 = bipartitions{w,1};
		part2 = bipartitions{w,2};
		
		% do everything for a specific bipartition
		clear covm_present;
		clear covn_present;
		clear covm_past;
		clear covn_past;
		clear cov_present_past_mm;
		clear cov_present_past_nn;
		clear cond_cov_present_past_mm;
		clear cond_cov_present_past_nn;
		
		covm_present = cov_present(part1, part1);			% insert subset from cov_present
		covn_present = cov_present(part2, part2);
		
		covm_past = cov_past(part1, part1);				% insert subset from cov_past
		covn_past = cov_past(part2, part2);
		
		cov_present_past_mm = cov_present_past(part1, part1);	% insert subset from cov_present_past
		cov_present_past_nn = cov_present_past(part2, part2);
		

		% get variance of x1_t-tau conditioned on x1_t:
		%\Sigma_(x1(t-tau),x1(t-tau)) - \Sigma_x1(t-tau)x1(t) * \Sigma_(x1(t),x1(t))^(-1) * \Sigma_x1(t)x1(t-tau)
		cond_cov_present_past_mm = covm_past - (cov_present_past_mm' / covm_present) * cov_present_past_mm;
		% get variance of x2_t-tau conditioned on x2_t:
		% \Sigma_(x2(t-tau),x2(t-tau)) - \Sigma_x2(t-tau)x2(t) * \Sigma_(x2(t),x2(t))^(-1) * \Sigma_x1(t)x2(t-tau)
		cond_cov_present_past_nn = covn_past - (cov_present_past_nn' / covn_present) * cov_present_past_nn;
		
		% eq. (0.33) in Barrett & Seth (2011)
		phi = 0.5*log(det(cov_present) / det(cond_cov_present_past)) - 0.5*log ((det(covm_present) / det(cond_cov_present_past_mm)) + ...
			(det(covn_present) / det(cond_cov_present_past_nn)));
		
		if isinf(phi) == true
			phi = NaN;
		end
		
		% normalisation factor for specific bipartition (see Barrett & Seth (2011))
		normm = 0.5*log(((2*pi*exp(1))^length(part1))*det(covm_present));
		normn = 0.5*log(((2*pi*exp(1))^length(part2))*det(covn_present));
		
		if normm<normn
			Normalise=normm;
		else
			Normalise=normn;
		end
		
		% normalised effective information
		phinorm = phi/Normalise;
		
		% record as Phi if minimum
		Phi = phi;
		Partition1 = part1;
		Partition2 = part2;
		
		if phinorm < PhiNorm
			PhiNorm = phinorm;
		end
	end
	
	% organize output structure
	ret.Partition1 = part1;
	ret.Partition2 = part2;
	ret.Phi = Phi;
	ret.type = 'td_normal';
    
end 