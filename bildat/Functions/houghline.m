function [linepar, acc] = ...
    houghline(curves, magnitude, ...
    nrho, ntheta, threshold, nlines, verbose)

imgsize = max(size(magnitude));
magnitude = abs(magnitude);

indices = curves(1,:) > 0;
x = curves(1,indices)';
y = curves(2,indices)';

% Theta discretization
thetas = linspace(-pi/2,pi/2,ntheta);

hough = zeros(nrho, ntheta); 

% Rho discretization.
r = x*cos(thetas) + y*sin(thetas);
% Each row in r contains the the rho values for corresponding
% theta values, for a single point r_i

rhos = linspace(floor(min(r(:))) - 1, ceil(max(r(:))) + 1, nrho);

% Map rho values to rho indices.
dr = (rhos(nrho) - rhos(1)) / nrho;
r = floor(r ./ dr);
r = r - (min(r(:)) - 1);

% Accumulate.
for i = 1:length(x)
    mag = magnitude(round(x(i)), round(y(i)));
    if mag >= threshold || 1 == 1
        for j = 1:ntheta
            h = @(x) x;
            hough(r(i,j),j) = hough(r(i,j),j) + 1;
        end
    end    
end
%hough = binsepsmoothiter(hough, 0.5, 2);

dothis = 0;
if dothis == 1
    degrees = 10;
    yy = nrho/2 - 20;
    xx = 1:2:degrees+1;
    hough = zeros(length(rhos), length(thetas)); 
    %hough(xx, yy) = 100;

    %nlines = round(degrees/2);
    nlines = 1;
    hough(50,90-45) = 100; 

end

[mpos, mvalue] = locmax8(hough);
[dummy, indices] = sort(mvalue, 'descend');

f = @(x,r,theta) (r - x.*cos(theta)) / sin(theta);

linepar = [];
for i = 1:nlines
    rho_i = mpos(indices(i), 1);
    theta_i = mpos(indices(i), 2);
    line = [0 0 imgsize;
            2 f([0 imgsize], rhos(rho_i), thetas(theta_i))];
    
    sprintf("Line: rho: %f   theta: %f", rhos(rho_i), rad2deg(thetas(theta_i)))    
    linepar = [linepar line];
end
acc = hough;

end