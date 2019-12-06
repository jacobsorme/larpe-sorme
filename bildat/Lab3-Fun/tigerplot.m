function [mean_colored,overlay_original] = tigerplot(img,K,L,scale_factor,image_sigma,seed)

I = imread(img);
I = imresize(I, 1/scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma);
I = imfilter(I, h);


[ segm, ~ ] = kmeans_segm(I, K, L, seed);
mean_colored = mean_segments(Iback, segm);
overlay_original = overlay_bounds(Iback/2, segm);
end

