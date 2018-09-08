clear all; close all; clc

% lambda-omega reaction-diffusion system
%  u_t = lam(A) u - ome(A) v + d1*(u_xx + u_yy) = 0
%  v_t = ome(A) u + lam(A) v + d2*(v_xx + v_yy) = 0
%
%  A^2 = u^2 + v^2 and
%  lam(A) = 1 - A^2
%  ome(A) = -beta*A^2
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);


t=0:0.05:10;
d1=0.1; d2=0.1; beta=1.0;
L=20; n=32; N=n*n;
x2=linspace(-L/2,L/2,n+1); x=x2(1:n); y=x;
kx=(2*pi/L)*[0:(n/2-1) -n/2:-1]; ky=kx;

% INITIAL CONDITIONS

[X,Y]=meshgrid(x,y);
[KX,KY]=meshgrid(kx,ky);
K2=KX.^2+KY.^2; K22=reshape(K2,N,1);

max_iter = 20;

for iter=1:max_iter
    u = zeros(length(x),length(y),length(t));
    v = zeros(length(x),length(y),length(t));

    %m=1; % number of spirals
    %u(:,:,1)=tanh(sqrt(X.^2+Y.^2)).*cos(m*angle(X+i*Y)-(sqrt(X.^2+Y.^2)));
    %v(:,:,1)=tanh(sqrt(X.^2+Y.^2)).*sin(m*angle(X+i*Y)-(sqrt(X.^2+Y.^2)));

    num_coeff = 3;
    start_coeff = 1;
    ufft = zeros(length(x), length(y));
    vfft = zeros(length(x), length(y));
    ufft(start_coeff:start_coeff-1+num_coeff,start_coeff:start_coeff-1+num_coeff) = (rand(num_coeff,1)-.5) + (rand(num_coeff,1)-.5)*i;
    vfft(start_coeff:start_coeff-1+num_coeff,start_coeff:start_coeff-1+num_coeff) = (rand(num_coeff,1)-.5) + (rand(num_coeff,1)-.5)*i;
    u(:,:,1)=N*real(ifft2(ufft));
    v(:,:,1)=N*real(ifft2(vfft));

    % REACTION-DIFFUSION
    uvt=[reshape(fft2(u(:,:,1)),1,N) reshape(fft2(v(:,:,1)),1,N)].';
    [t,uvsol]=ode45('reaction_diffusion_rhs',t,uvt,[],K22,d1,d2,beta,n,N);


    for j=1:length(t)-1
        ut=reshape((uvsol(j,1:N).'),n,n);
        vt=reshape((uvsol(j,(N+1):(2*N)).'),n,n);
        u(:,:,j+1)=real(ifft2(ut));
        v(:,:,j+1)=real(ifft2(vt));
    end

    save(['RD_data/N',num2str(n),'/iter',num2str(iter),'.mat'],'t','x','y','u','v','-v7')
end
