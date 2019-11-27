%% Add Path & Stuff
addpath Functions
addpath Images-m
addpath Images-mat
addpath Images

%% 1 Difference Operators
close all; clc; set(gcf,"Position",[80 80 900 600]);

dx = [0 0 0; -0.5 0 0.5; 0 0 0];
dy = [0 -0.5 0 ; 0 0 0; 0 0.5 0 ];

subplot(1,3,1); showgrey(few256);
subplot(1,3,2); showgrey(conv2(few256,dx,'valid'));
subplot(1,3,3); showgrey(conv2(few256,dy,'valid'));

%saveas(gcf,"1.png")

%% 2 Point-wise thresholding of gradient magnitudes
close all; clc; set(gcf,"Position",[80 80 900 600]);

dx = [0 0 0; -0.5 0 0.5; 0 0 0];
dy = [0 -0.5 0 ; 0 0 0; 0 0.5 0 ];

thresholds = [2 5 8 11 14 17];
sigmas = [1 3 8];
%img = few256;
img = godthem256;

for i = 1:length(sigmas)
    for j =1:length(thresholds)
        subplot(length(sigmas),length(thresholds),(i-1)*length(thresholds) + j);
        showgrey(Lv(discgaussfft(img,sigmas(i))) > thresholds(j));
        title("\sigma: " + sigmas(i) + ",    t: " + thresholds(j))
    end
end
%saveas(gcf,"2.png");

%% 4 Computing differential geometry descriptors
close all; clc; set(gcf,"Position",[80 80 900 200]);

sigmas = [0.0001 1 4 16 64]; 
for i = 1:length(sigmas)
    subplot(1,length(sigmas),i);
    contour(Lvvtilde(discgaussfft(few256, sigmas(i) )), [0 0]);
    set(gca,'xtick',[]); set(gca,'ytick',[]); 
    axis('image'); axis('ij'); title("\sigma : " + sigmas(i))
end
%saveas(gcf,"5-1.png")

figure; set(gcf,"Position",[80 400 900 200]);
for i = 1:length(sigmas)
    subplot(1,length(sigmas),i);
    showgrey(Lvvvtilde(discgaussfft(few256, sigmas(i))) < 0);
    title("\sigma : " + sigmas(i))
end
%saveas(gcf,"5-2.png")


%% 5 Extraction of edge segments
close all; clc; set(gcf,"Position",[80 80 900 900]);
img = godthem256;
img(1,1) = -200; % Dim the image

curves = extractedges(img, 6, 3.2, '');
overlaycurves(img, curves);

%% 6 Hough Transform
close all; clc; clear variables; set(gcf,"Position",[80 80 900 700]);

imgsize = 256;
img = houghtest256;
img(1,1) = -200; % Dim the image
curves = extractedges(img,6, 5, '');

subplot(1, 3, 1)
overlaycurves(img, curves);
[lines, hough] = houghedgeline(img, 8, 6, 0.01, imgsize, 180, 10, 0);

subplot(1, 3, 2)
showgrey(log(1 + hough))

subplot(1, 3, 3)
overlaycurves(img, lines)
xlim([0 255]); ylim([0 255]);

%saveas(gcf,"6-1.png")

