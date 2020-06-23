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

mean(Y);
mean_matrix = repmat(ans,64,1);
Y = Y - mean_matrix ;
%------------------------------------------------------------------


%we initialize the Dictionary with the overcomplete DCT transform    
D = dctmtx(256);
D(65:256,:) = [];

X = ones(size(D,2),size(Y,2));


%Dictionary learning algorithm-------------------------------
for i = 1 : 10
    
    %sparse coding 
    for k = 1 : size(Y,2)
        [X(:,k),supp,iter] = omp(Y(:,k),D,6,1);
    end 
    
    %dictionary update 
    [D,X] = K_svd(D , X , Y);
    
end    


    for k = 1 : size(Y,2)
        [X(:,k),supp,iter] = omp(Y(:,k),D,6,1);
    end
 
%----------------------------------------------------------

%reconstructing the image by averaging all the overlapped and denoised
%patches to obtain the corresponding pixels-----------------------------
Y1 = reshape(D*X + mean_matrix , [w w m]);


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
title('Denoised Image \n');
figure,imshow(J);
title('Noisy Image');
        
        
