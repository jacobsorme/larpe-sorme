function [linepar, acc] = ...
    houghedgeline(img, scale, gradmagnthreshold, houghthreshold, ...
    nrho, ntheta, nlines, verbose)

[curves, mag] = extractedges(img,scale, gradmagnthreshold);
[linepar, acc] = houghline(curves, mag, nrho,ntheta, houghthreshold, ...
    nlines, verbose);

end

