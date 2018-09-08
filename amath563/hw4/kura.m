clear all; close all; clc

% Kuramoto

t=0:0.05:100;



K=2;  % coupling strength
n=10; % number of oscillators
rad=ones(n,1);
thetai=2*randn(n,1);
omega=rand(n,1)+0.5;

A=rand(n,n);  
A=(A>0.5).*A;






[t,y]=ode45('kura_rhs',t,thetai,[],omega,n,K,A);

for j=1:length(t)
        ynow=y(j,:);
        polar(ynow.',rad,'o');
        drawnow
end

