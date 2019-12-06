scale_factor = 1/2.0;       % image downscale factor
spatial_bandwidth = 5.0;  % spatial bandwidth
colour_bandwidth = 5;   % colour bandwidth
num_iterations = 100;      % number of mean-shift iterations
image_sigma = 1.0;        % image preblurring scale

I = imread('godthem.jpg');
I = imresize(I, scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma);
I = imfilter(I, h);

segm = mean_shift_segm(I, spatial_bandwidth, colour_bandwidth, num_iterations);
Inew = mean_segments(Iback, segm);
I = overlay_bounds(Iback, segm);
subplot(2,1,1); imshow(Inew);
subplot(2,1,2); imshow(I);


