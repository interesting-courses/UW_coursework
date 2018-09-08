dt=t(2)-t(1);
X=real(usol);
[m,n]=size(X);

Xdot=zeros(m,n-2);
for jj=1:m  % walk through rows (space)
for j=2:n-1  % walk through time
   Xdot(jj,j-1)=( X(jj,j+1)-X(jj,j-1) )/(2*dt);
end
end

% derv matrices
dx=x(2)-x(1);

D=zeros(m,m); D2=zeros(m,m);
for j=1:m-1
  D(j,j+1)=1;
  D(j+1,j)=-1;
%
  D2(j,j+1)=1;
  D2(j+1,j)=1;
  D2(j,j)=-2;
end
D(m,1)=1;
D(1,m)=-1;
D=(1/(2*dx))*D;
%

D2(m,m)=-2;
D2(m,1)=1;
D2(1,m)=1;
D2=D2/(dx^2);

u=reshape( X(:,2:end-1).',(n-2)*m ,1 );

for jj=2:n-1
   ux(jj-1,:)=((D*X(:,jj)).');  % u_x
   uxx(jj-1,:)=((D2*X(:,jj)).');  % u_xx
   u2x(jj-1,:)=((D* (X(:,jj).^2) ).');  % (u^2)_x
end

% why didn't they do it like this before??
ux1 = D*X(:,2:n-1);
uxx1 = D2*X(:,2:n-1);
u2x1 = D*(X(:,2:n-1)).^2;

Ux=reshape(ux1',(n-2)*m,1);
Uxx=reshape(uxx,(n-2)*m,1);
U2x=reshape(u2x,(n-2)*m,1);

A=[u u.^2 u.^3 Ux Uxx U2x Ux.*u Ux.*Ux Ux.*Uxx];

Udot=reshape((Xdot.'),(n-2)*m,1);
xi=A\Udot;
%xi=lasso(A,Udot,'Lambda',0.01);
bar(xi)



