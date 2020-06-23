function [D_new,X_new] = K_svd(D,X,Y)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

for k = 1 :size(D,2)
    E = Y  - ( D*X - D(:,k)*X(k,:) );
    ind = find(X(k,:));%finding the indices of the non zero entries of the k-th row of X
    
    if isempty(ind)
        continue; %if the k-th atom is not used from any signal continue to the next iteration
    end
    
    E_reduced = E(:,ind);
    [U,S,V] = svd(E_reduced);
    u = U(:,1);
    s = S(1,1);
    v = V(:,1);
    D(:,k) = u;
    X(k,:) = 0;
    X(k,ind) = s*v';
end    

D_new = D ;
X_new = X ;
end

