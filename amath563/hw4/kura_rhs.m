function rhs=kura_rhs(t,theta,dummy,omega,n,K,A)

coupling=zeros(n,1);
for j=1:n
  C=0;
  for jj=1:n
    C=C+A(jj,j)*sin(theta(jj)-theta(j));
  end
  coupling(j,1)=C;
end

rhs=omega+(K/n)*coupling;



