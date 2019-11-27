function [curves, mag] = extractedges(inpic, scale, threshold, shape)

pic = discgaussfft(inpic, scale);
curves = zerocrosscurves(Lvvtilde(pic), -Lvvvtilde(pic));

mag = Lv(pic);
curves = thresholdcurves(curves, mag - threshold);

end
