scale_factor = 0.5;          % image downscale factor
area = [ 80, 110, 570, 280 ];% image region to train foreground with
K = 16;                      % number of mixture components
alpha = 10;                 % maximum edge cost
sigma = 64;                % edge cost decay factor

I = imread('tiger1.jpg');
I = imresize(I, scale_factor);
Iback = I;
area = int16(area*scale_factor);
[ segm, prior ] = graphcut_segm(I, area, K, alpha, sigma);

mask_img = double(zeros(size(I,1), size(I,2)));
mask_img(area(2):area(4), area(1):area(3)) = 1;

Inew = mean_segments(Iback, segm);
I = overlay_bounds(Iback, segm);
subplot(3,1,1); imshow(Inew);
subplot(3,1,2); imshow(I);
bla = uint8(zeros(size(I)));
bla(:,:,1) = uint8(prior*256);
bla(:,:,2) = uint8(prior*256);
bla(:,:,3) = uint8(prior*256);

subplot(3,1,3); imshow(overlay_bounds(bla, mask_img));
