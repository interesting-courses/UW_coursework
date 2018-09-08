clear all; close all; clc

dt=0.01;
t=0:dt:50; 
x0=[0.1 5];
mu=1.2;

[t,y]=ode45('rhs_dyn',t,x0,[],mu);

plot(t,y(:,1),t,y(:,2),'Linewidth',[2])

x1=y(:,1);
x2=y(:,2);


n=length(t);
for j=2:n-1
  x1dot(j-1)=(x1(j+1)-x1(j-1))/(2*dt);
  x2dot(j-1)=(x2(j+1)-x2(j-1))/(2*dt);
end

x1s=x1(2:n-1);
x2s=x2(2:n-1);
%A=[x1s x2s x1s.^2 x1s.*x2s x2s.^2 x1s.^3 (x2s.^2).*x1s x2s.^3];
A=[x1s x2s x1s.^2 x1s.*x2s x2s.^2 x1s.^3 (x1s.^2).*x2s (x2s.^2).*x1s x2s.^3];

xi1=A\x1dot.';
xi2=A\x2dot.';
subplot(2,1,1), bar(xi1)
subplot(2,1,2), bar(xi2)

