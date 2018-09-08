clear all; close all; clc

% Kuramoto-Sivashinsky equation (from Trefethen)
% u_t = -u*u_x - u_xx - u_xxxx,  periodic BCs 

% Generate data from random initial conditions
start_iter = 21;
num_iter = 20;
N = 1024;
x = 32*pi*(1:N)'/N;
for iter = start_iter:(start_iter+num_iter)
    % generate random initial conditions
    num_coeff = 4;
    v = zeros(N,1);
    v(2:1+num_coeff) = (rand(num_coeff,1)-.5) + (rand(num_coeff,1)-.5)*i;
    u = N*real(ifft(v));
    v = fft(u);

    % % % % % %
    %Spatial grid and initial condition:
    h = 0.025;
    k = [0:N/2-1 0 -N/2+1:-1]'/16;
    L = k.^2 - k.^4;
    E = exp(h*L); E2 = exp(h*L/2);
    M = 16;
    r = exp(1i*pi*((1:M)-.5)/M);
    LR = h*L(:,ones(M,1)) + r(ones(N,1),:);
    Q = h*real(mean( (exp(LR/2)-1)./LR ,2)); 
    f1 = h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2)); 
    f2 = h*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
    f3 = h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));

    % Main time-stepping loop:
    uu = u; tt = 0;
    tmax = 100; nmax = round(tmax/h); nplt = floor((tmax/250)/h); g = -0.5i*k;
    for n = 1:nmax
    t = n*h;
    Nv = g.*fft(real(ifft(v)).^2);
    a = E2.*v + Q.*Nv;
    Na = g.*fft(real(ifft(a)).^2);
    b = E2.*v + Q.*Na;
    Nb = g.*fft(real(ifft(b)).^2);
    c = E2.*a + Q.*(2*Nb-Nv);
    Nc = g.*fft(real(ifft(c)).^2);
    v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3; if mod(n,nplt)==0
            u = real(ifft(v));
    uu = [uu,u]; tt = [tt,t]; end
    end

    save(['KS_data/N',num2str(N),'/iter',num2str(iter),'.mat'],'x','tt','uu', '-v7')
end
