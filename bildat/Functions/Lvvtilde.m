function pixels = Lvvtilde(inpic)
dx = [  0 0 0 0 0;
        0 0 0 0 0;
        0 -.5 0 .5 0;
        0 0 0 0 0;
        0 0 0 0 0 ];
    
dy = [  0 0 0 0 0;
        0 0 -.5 0 0;
        0 0 0 0 0;
        0 0 .5 0 0
        0 0 0 0 0];

dxx = [ 0 0 0 0 0;
        0 0 0 0 0;
        0 1 -2 1 0;
        0 0 0 0 0
        0 0 0 0 0];
    
dyy = [ 0 0 0 0 0;
        0 0 1 0 0;
        0 0 -2 0 0;
        0 0 1 0 0;
        0 0 0 0 0];

dxy = conv2(dx,dy,'same');

Lx = filter2(dx, inpic,'same');
Ly = filter2(dy, inpic, 'same');

Lxx = filter2(dxx, inpic,'same');
Lyy = filter2(dyy, inpic,'same');
Lxy = filter2(dxy, inpic,'same');


%pixels = Lx.^2 + Ly.^2;
pixels = (Lx.^2).*Lxx + 2*Lx.*Ly.*Lxy + (Ly.^2).*Lyy;

end