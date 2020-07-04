clear;

f = imread('barbara.jpg');

f = im2double(f);

signal_avg_power = (f(:)'*f(:))/length(f(:));
%in order to obtain a SNR=20db
sigma_squared = signal_avg_power/100;
noise = sqrt(sigma_squared).*randn(size(f));
J = f + noise;

%extracting all the overlapped patches-----------------------------
n0 = 256;
w = 8;
p = 256;
n = w*w;
q = 1;

[y,x] = meshgrid(1:q:n0-w/2, 1:q:n0-w/2);
m = size(x(:),1);
    
[dY,dX] = meshgrid(0:w-1,0:w-1);

Xp = repmat(dX,[1 1 m]) + repmat( reshape(x(:),[1 1 m]), [w w 1]);
Yp = repmat(dY,[1 1 m]) + repmat( reshape(y(:),[1 1 m]), [w w 1]); 

Xp(Xp>n0) = 2*n0-Xp(Xp>n0);
Yp(Yp>n0) = 2*n0-Yp(Yp>n0);

Y = J(Xp+(Yp-1)*n0);
Y = reshape(Y, [n, m]);
%-----------------------------------------------------------------

%centering and normalizing the data--------------------------
mean(Y);
mean_matrix = repmat(ans,64,1);
Y = Y - mean_matrix ;

norm_of_each_column = [];
for b = 1 : size(Y,2)
    norm_of_each_column = [ norm_of_each_column norm(Y(:,b)) ];
    Y(:,b) = Y(:,b)/norm_of_each_column(b);
end        
%------------------------------------------------------------------


%we initialize the Dictionary with the overcomplete DCT transform
D = dctmtx(256);
D = D';
D(65:256,:) = [];

X = ones(size(D,2),size(Y,2));


%Dictionary learning algorithm-------------------------------
for i = 1 : 10

    %normalizing all the atoms of the dictionary in order for the OMP to work properly
    for i = 1 : size(D,2)
        D(:,i) = D(:,i)./norm(D(:,i));
    end 
    
    %sparse coding 
    for k = 1 : size(Y,2)
        X(:,k) = omp(Y(:,k),D,6,1e-5);
    end 
    
    %dictionary update 
    [D,X] = K_svd(D , X , Y);
    
end    


    for k = 1 : size(Y,2)
        X(:,k) = omp(Y(:,k),D,6,1e-5);
    end
 
%----------------------------------------------------------

%reconstructing the image by averaging all the overlapped and denoised
%patches to obtain the corresponding pixels-----------------------------
temp = D*X ; 
for i = 1 : size(temp,2)
    temp(:,i) = temp(:,i).*norm_of_each_column(i);
end    
    
Y1 = reshape(temp + mean_matrix , [w w m]);


W = zeros(n0,n0);
f1 = zeros(n0,n0);
for i=1:m
    x = Xp(:,:,i); y = Yp(:,:,i);
    f1(x+(y-1)*n0) = f1(x+(y-1)*n0) + Y1(:,:,i);
    W(x+(y-1)*n0) = W(x+(y-1)*n0) + 1;
end
f1 = f1 ./ W;
%-----------------------------------------------------------------------

%computing PSNR 
cost = f(:) - f1(:);
MSE = (cost'*cost)/length(f(:));
PSNR = 20*log10(max(f(:))) - 10*log10(MSE)


figure,imshow((f1));
title('Denoised Image');
figure,imshow(J);
title('Noisy Image');
        
        
