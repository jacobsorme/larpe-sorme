function pixels = Lvvvtilde(inpic)
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

dxxx = conv2(dx,dxx,'same');
dyyy = conv2(dy,dyy,'same');

dxxy = conv2(dxx,dy,'same');
dxyy = conv2(dx,dyy,'same');

Lx = filter2(dx, inpic,'same');
Ly = filter2(dy, inpic, 'same');

Lxxx = filter2(dxxx, inpic, 'same');
Lxxy = filter2(dxxy, inpic,'same');
Lxyy = filter2(dxyy, inpic, 'same');
Lyyy = filter2(dyyy, inpic, 'same');

%pixels = Lx.^2 + Ly.^2;
%pixels = (Lx.^2).*Lxx + 2*Lx.*Ly.*Lxy + (Ly.^2).*Lyy;
pixels = (Lx.^3).*Lxxx + 3*(Lx.^2).*Ly.*Lxxy + 3*Lx.*(Ly.^2).*Lxyy + (Ly.^3).*Lyyy;
end