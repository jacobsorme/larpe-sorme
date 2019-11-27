function pingpong = gaussfft(pic,t)
    %gauss = zeros(size(pic));
    s1 = size(pic,1) / 2;
    s2 = size(pic,1) / 2;
    [x,y] = meshgrid(-s1:s1-1,-s2:s2-1);
    gauss = (1/(2*pi*t))*exp( -(x.^2 + y.^2) ./ (2*t));

    pingpong = fftshift(ifft2( fft2(pic) .* fft2(gauss)));
end

