function sparse_x = omp(y, D, K, err)

residual = y ;
support = [] ;

sparse_x = zeros( size(D,2) , 1 );

tmp = D';


iteration = 1;
while ( norm(residual) > err ) && ( iteration <= K )
	iteration = iteration + 1;
	
	
	%idx is the index of the collumn (of D) that has the max inner product with the residual
	[~, idx] = max(abs(tmp * residual));
	
	selected_atom = tmp(idx,:);
	
	k = tmp(idx,:)*residual;
	
	sparse_x(idx) = k;
	
	support = [ support idx ];
	
	%orthogonal matcing persuit step
	%projecting the signal to the subspace spanned by the atoms that are in the support set
	Dsup = D(:,support);
	%residual = (eye(size(D,1)) - Dsup*inv(Dsup'*Dsup)*Dsup')*y;
    updated_coefs = Dsup\y;
    residual = y - ( Dsup * updated_coefs );
	
end	
	
end	
