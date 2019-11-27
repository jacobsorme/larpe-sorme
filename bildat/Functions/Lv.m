function pixels = Lv(inpic)
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
Lx = filter2(dx, inpic,'same');
Ly = filter2(dy, inpic, 'same');
pixels = sqrt(Lx.^2 + Ly.^2);

end