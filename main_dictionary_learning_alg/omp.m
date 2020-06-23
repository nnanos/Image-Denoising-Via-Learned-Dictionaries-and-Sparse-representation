function [sparse_x, support, iteration] = omp(y, A, K, S, err)


 	if nargin < 5
	   err    = 1e-5;
    end 
    
	sparse_x	  = zeros(size(A,2), 1);
	residual  = y;
	supp	  = [];
	iteration = 0; 
	
	while (norm(residual) > err && iteration < min(K, floor(size(A,1)/S))) 
		   iteration          = iteration + 1;
		   [~, idx]           = sort(abs(A' * residual), 'descend');
		   supp_temp          = union(supp, idx(1:S));

	   if (length(supp_temp) ~= length(supp))
           supp	              = supp_temp;
		   x_hat			  = A(:,supp)\y;
		   residual           = y - A(:,supp) * x_hat; 
       else
		   break;
       end
    end
    
 	sparse_x(supp)	          = A(:,supp)\y;
	[~, supp_idx]             = sort(abs(sparse_x), 'descend');
	support                   = supp_idx(1:K); 
	sparse_x                    = zeros(size(A,2), 1);
    sparse_x(support)           = A(:,support)\y;
end
