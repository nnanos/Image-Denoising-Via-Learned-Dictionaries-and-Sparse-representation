f = imread('barbara.jpg');

f = im2double(f);


load('trained_dictionary_barbara.mat');
load('sparse_representations_barbara.mat');
load('mean_values_for_each_initial_patch.mat');
load('Xp_coordinates.mat');
load('Yp_coordinates.mat');
load('noisy_image.mat')

Y1 = reshape(D*X + mean_matrix , [8 8 63504]);

W = zeros(256,256);
f1 = zeros(256,256);
for i=1:63504
    x = Xp(:,:,i); y = Yp(:,:,i);
    f1(x+(y-1)*256) = f1(x+(y-1)*256) + Y1(:,:,i);
    W(x+(y-1)*256) = W(x+(y-1)*256) + 1;
end
f1 = f1 ./ W;

%computing PSNR 
cost = f(:) - f1(:);
MSE = (cost'*cost)/length(f(:));
PSNR = 20*log10(max(f(:))) - 10*log10(MSE)


figure,imshow((f1));
title('Denoised Image');

figure,imshow((J));
title('Noisy Image');


