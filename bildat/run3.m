%% Add Path & Stuff
addpath Functions
addpath Images-m
addpath Images-mat
addpath Images
addpath bildat_lab3/
addpath Lab3-Fun

%% K-means  Q: 1, 2
close all; clc; set(gcf,"Position",[80 80 900 600]);

K0 = 4;
L0 = 1000;
sigma0 = 1.0;
scale0 = 1.0;

K = [2 4 8 16];
L = [1 3 9 1000];
scale = [2 4 8 16];
sigma = [1 4 8 16];

seed = 100;
for i = 1:length(K)
    % L 
     subplot(4,4,i); 
     [I, ~] = tigerplot('godthem.jpg',K0,L(i),scale0,sigma0,seed);
     imshow(I); title("Max Iter. = "+L(i));
     
    % K 
    subplot(4,4,4 + i); 
    [I, ~] = tigerplot('tiger1.jpg',K(i),L0, 2.0,sigma0,seed);
    imshow(I); title("Clusters = "+K(i));
    
    % scale 
    subplot(4,4,8 + i); 
    [I, ~] = tigerplot('tiger1.jpg',K0,L0,scale(i),sigma0,seed);
    imshow(I); title("Downscale = "+scale(i));
    
    %sigma 
    subplot(4,4,12 + i); 
    [I, ~] = tigerplot('tiger3.jpg',8,L0,scale0,sigma(i),seed);
    imshow(I); title("Gauss sigma = "+sigma(i));
end
%% K-means  Q: 3, 4
close all; clc; set(gcf,"Position",[80 80 900 600]);

[I, Io] = tigerplot('orange.jpg',6, 100,1,1,1);
subplot(1,2,1); imshow(I);
subplot(1,2,2); imshow(Io);

%% Mean-shift
close all; clc; set(gcf,"Position",[80 80 800 300]);

scale_factor = 1/2.0;       % image downscale factor
spatial_bandwidth = 10;  % spatial bandwidth
colour_bandwidth = 4;   % colour bandwidth
num_iterations = 100;      % number of mean-shift iterations
image_sigma = 2;        % image preblurring scale

I = imread('tiger1.jpg');
I = imresize(I, scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma);
I = imfilter(I, h);

segm = mean_shift_segm(I, spatial_bandwidth, colour_bandwidth, num_iterations);
Inew = mean_segments(Iback, segm);
I = overlay_bounds(Iback, segm);

subplot(1,2,2); 
I = tigerplot('tiger1.jpg',6,1000,2,2,1);
imshow(I);
title("K-means, 6 centers")

subplot(1,2,1); imshow(Inew);
title("Mean-shift, \sigma_s= 10, \sigma_c= 4")  
%title("Downscale = "+scale(i));

%% Doger normalized cut
close all; clc; set(gcf,"Position",[80 80 800 900]);

% colour_bandwidth = 20.0; % color bandwidth
% radius = 5;              % maximum neighbourhood distance
% ncuts_thresh = 0.07;      % cutting threshold
% min_area = 350;          % minimum area of segment
% max_depth = 4;           % maximum splitting depth
% scale_factor = 0.5;      % image downscale factor
% image_sigma = 2.0;       % image preblurring scale
% 
% I = imread('tiger3.jpg');
% I = imresize(I, scale_factor);
% Iback = I;
% d = 2*ceil(image_sigma*2) + 1;
% h = fspecial('gaussian', [d d], image_sigma);
% I = imfilter(I, h);
% 
% segm = norm_cuts_segm(I, colour_bandwidth, radius, ncuts_thresh, min_area, max_depth);
% Inew = mean_segments(Iback, segm);
I = overlay_bounds(Iback, segm);
imshow(I);
x = 20; y = -30;
text(x,100+y,"colour bandwidth = 20.0", 'BackgroundColor',[1 1 1 0.8])
text(x,110+y,"radius = 5", 'BackgroundColor',[1 1 1 0.8])
text(x,120+y,"ncuts thresh = 0.07", 'BackgroundColor',[1 1 1 0.8])
text(x,130+y,"min area = 350", 'BackgroundColor',[1 1 1 0.8])
text(x,140+y,"max depth = 4", 'BackgroundColor',[1 1 1 0.8])
text(x,150+y,"scale factor = 0.5", 'BackgroundColor',[1 1 1 0.8])
text(x,160+y,"image sigma = 2.0", 'BackgroundColor',[1 1 1 0.8])


%% Orange normalized cut
close all; clc; set(gcf,"Position",[80 80 800 900]);

% colour_bandwidth = 20.0; % color bandwidth
% radius = 10;              % maximum neighbourhood distance
% ncuts_thresh = 0.07;      % cutting threshold
% min_area = 480;          % minimum area of segment
% max_depth = 8;           % maximum splitting depth
% scale_factor = 0.33;      % image downscale factor
% image_sigma = 2.0;       % image preblurring scale
% 
% I = imread('orange.jpg');
% I = imresize(I, scale_factor);
% Iback = I;
% d = 2*ceil(image_sigma*2) + 1;
% h = fspecial('gaussian', [d d], image_sigma);
% I = imfilter(I, h);
% 
% segm = norm_cuts_segm(I, colour_bandwidth, radius, ncuts_thresh, min_area, max_depth);
% Inew = mean_segments(Iback, segm);
% I = overlay_bounds(Iback, segm);
%subplot(1,2,1); 
x = 100; y = -30;
imshow(Inew);
text(x,100+y,"colour bandwidth = 20.0")
text(x,110+y,"radius = 10")
text(x,120+y,"ncuts thresh = 0.07")
text(x,130+y,"min area = 480")
text(x,140+y,"max depth = 8")
text(x,150+y,"scale factor = 0.33")
text(x,160+y,"image sigma = 2.0")
%title("Remember Trump and Putin? This is them now... feel old yet?");

%% Orange normalized cut
close all; clc; set(gcf,"Position",[80 80 800 900]);

radii = [0 7 10 15 20 25];%[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15];
for i = 1:length(radii)
    radius = radii(i);
    colour_bandwidth = 20.0; % color bandwidth
    ncuts_thresh = 0.07;      % cutting threshold
    min_area = 480;          % minimum area of segment
    max_depth = 8;           % maximum splitting depth
    scale_factor = 0.33;      % image downscale factor
    image_sigma = 2.0;       % image preblurring scale

    I = imread('orange.jpg');
    I = imresize(I, scale_factor);
    Iback = I;
    d = 2*ceil(image_sigma*2) + 1;
    h = fspecial('gaussian', [d d], image_sigma);
    I = imfilter(I, h);

    segm = norm_cuts_segm(I, colour_bandwidth, radius, ncuts_thresh, min_area, max_depth);
    Inew = mean_segments(Iback, segm);
    subplot(1,length(radii),i); 
    imshow(Inew);
    title("radius = " + radius)
end

%% Graph Cut Q 11
clear variables; close all; clc; set(gcf,"Position",[80 80 700 900]);

images = {'tiger1.jpg', 'orange.jpg', 'tiger3.jpg', 'bullar.jpg'};
scale_factors = [0.5 0.5 0.5 0.5];
areas = {[80 110 570 280], ... [270 210 500 420], 
    [230 230 320 320], [180 145 460 360], [150 45 380 240]};
K = [16 16 16 16];
alpha = [10 10 10 10];
sigma = [64 64 64 64];
%alpha = [10 10 10 30];
%sigma = [64 16 64 16];

for i = 1:4
    I = imread(images{i});
    I = imresize(I, scale_factors(i));
    Iback = I;
    area = int16(areas{i}*scale_factors(i));
    [ segm, prior ] = graphcut_segm(I, area, K(i), alpha(i), sigma(i));

    mask_img = double(zeros(size(I,1), size(I,2)));
    mask_img(area(2):area(4), area(1):area(3)) = 1;

    Inew = mean_segments(Iback, segm);
    I = overlay_bounds(Iback, segm);
    subplot(4,3,(i-1)*3 + 1); imshow(Inew);
    subplot(4,3,(i-1)*3 + 2); imshow(I);
    bla = uint8(zeros(size(I)));
    bla(:,:) = uint8([prior*256 prior*256 prior*256]);

    subplot(4,3,(i-1)*3 + 3); imshow(overlay_bounds(bla, mask_img));
end

%% Graph Cut Q 12
clear variables; close all; clc; set(gcf,"Position",[80 80 700 900]);

image = 'tiger3.jpg';
scale_factor = 0.5 ;
area = [180 145 460 360];
K = [1 3 7];
alpha = 10;
sigma = 64;

area = int16(area*scale_factor);
for i = 1:length(K)
    I = imread(image);
    I = imresize(I, scale_factor);
    Iback = I;
    
    [ segm, prior ] = graphcut_segm(I, area, K(i), alpha, sigma);

    mask_img = double(zeros(size(I,1), size(I,2)));
    mask_img(area(2):area(4), area(1):area(3)) = 1;

    Inew = mean_segments(Iback, segm);
    subplot(1,length(K),i); imshow(Inew);
    title("K = " + K(i));
end