%% Add Path & Stuff
addpath Functions
addpath Images-m
addpath Images-mat
addpath Images



%% 1.3 Basis Functions
close all; hold on; set(gcf,'Position',[100 100 900 800]);

fftwave(20,20,128);


%% 1.4 Linearity & 1.5 Multiplication
close all; hold on; set(gcf,'Position',[80 30 900 900]);

F = [ zeros(56, 128); ones(16, 128) zeros(16,0); zeros(56, 128)];
G = F';
H = F + 2 * G;
FG = F .* G;

Fhat = fft2(F); Ghat = fft2(G);
Hhat = fft2(H); FGhat = fft2(FG);

F = {{'$F$', F}; {'$G$', G}; {'$H$', H}; {"$FG$",FG};
    {'$\hat{F}$', log(1 + abs(Fhat))}
    {'$\hat{G}$', log(1 + abs(Ghat))}
    {"$\hat{H}$", log(1 + abs(Hhat))}
    {"$\hat{FG}$", log(1 + abs(FGhat))}
    {"$shift(\hat{F})$", log(1 + abs(fftshift(Fhat)))}
    {"$shift(\hat{G})$", log(1 + abs(fftshift(Ghat)))}
    {"$shift(\hat{H})$", log(1 + abs(fftshift(Hhat)))}
    {"$shift(\hat{FG})$", log(1 + abs(fftshift(FGhat)))}
    };

for i = 1:12
    subplot(4,4,i);
    showgrey(F{i}{2});
    title(F{i}{1},'Interpreter','Latex');
end

subplot(4,4,16);
ey = conv2(Fhat,Ghat,'full');
showgrey(fftshift(abs(ey(1:128, 1:128))));


%% 1.6 Scaling
close all; hold on; set(gcf,'Position',[80 30 900 900]);

F = [ zeros(56, 128); ones(16, 128); zeros(56, 128)];
F = F .* F';

rots = 4;

for i = 1:rots
   G = rot(F,(i-1)*30); 
   subplot(6,rots,i); showgrey(G);
   subplot(6,rots,rots+i); showfs(fft2(G));
end

F = [zeros(60, 128); ones(8, 128); zeros(60, 128)] .* ...
[zeros(128, 48) ones(128, 32) zeros(128, 48)];


for i = 1:rots
   G = rot(F,(i-1)*30); 
   subplot(6,rots,8+i); showgrey(G);
   subplot(6,rots,8+rots+i); showfs(fft2(G));
end

% Kan man göra en inf x inf fourier bild från en 128x128 bild? 
% Alla finns väl fler frekvenser än bara de i 128x128

F = [zeros(60, 128); ones(8, 128); zeros(60, 128)];% .* ...
%F  = [zeros(128, 48) ones(128, 32) zeros(128, 48)];

rots = 4;
for i = 1:rots
   G = rot(F,(i-1)*30); 
   subplot(6,rots,16+i); showgrey(G);
   subplot(6,rots,16+rots+i); showfs(fft2(G));
end

%% 1.8 Information in Fourier phase and Magnitude
close all; hold on; set(gcf,'Position',[80 30 900 900]);

s = 0.5;
figures = { phonecalc128.^(s), few128.^(s), nallo128.^(s) };

for i = 1:length(figures)
   subplot(3,3,3*(i-1) + 1)
   showgrey(figures{i});
   
   subplot(3,3,3*(i-1) + 2)
   showgrey(pow2image(figures{i}, 0.00001)); 
   
   subplot(3,3,3*(i-1) + 3)
   showgrey(randphaseimage(figures{i})); 
end


%% 2.3 Filtering Procedure
close all; hold on; set(gcf,'Position',[80 30 1200 900]);


t = [0.1 0.3 1 10 100];
for i = [0.1 0.3 1 10 100]
    psf = gaussfft(deltafcn(128, 128), i);
    variance(psf)
end
t = [1 4 16 64 256];

images = dir("Images-m");
images = {images.name};
images = images(8:end);
images = images(randperm(length(images),5));

for i = 1:length(images)
   for j = 1:length(t)
       subplot(length(images),length(t),length(t)*(i-1) +j);
       img = char(images{i});
       img = img(1:end-2);
        showgrey(gaussfft(eval(img),t(j)));
        title(img+ " t = "+ t(j));
   end
end

%% 3.1 Smoothing of noisy data
clc; close all; hold on; set(gcf,'Position',[70 20 950 800]);


images = dir("Images-m");
images = {images.name};
images = images(8:end);
img = char(images(randi(length(images))))
img = eval(img(1:end-2));

noises = {(img) (gaussnoise(img,16)) (sapnoise(img,0.1,255))};
titles = ["original", "gaussnoise", "sapnoise"];
settings = [10,9,0.09];

for i = 1:3
    subplot(3,4,4*(i-1)+1);
    showgrey(noises{i});
    title(titles(i))
    subplot(3,4,4*(i-1)+2);
    showgrey(gaussfft(noises{i},settings(1)));
    title("gaussfft " + settings(1));
    subplot(3,4,4*(i-1)+3);
    showgrey(medfilt(noises{i},settings(2)));
    title("medfilt, size " + settings(2));
    subplot(3,4,4*(i-1)+4);
    showgrey(ideal(noises{i},settings(3)));
    title("ideal, cutoff " + settings(3));
end

%% 3.2 Smoothing and subsampling
clc; close all; hold on; set(gcf,'Position',[70 20 1200 800]);


img = phonecalc256;
smoothimg = img;
N=5;
for i=1:N
    if i>1
        % generate subsampled versions
        img = rawsubsample(img);
        smoothimg = ideal(smoothimg,0.5);
        smoothimg = rawsubsample(smoothimg);
    end
    subplot(2, N, i)
    showgrey(img)
    title("original - subsample "+ i)
    subplot(2, N, i+N)
    showgrey(smoothimg)
    title("ideal - subsample "+ i)
end










