function rhs=rhs_dyn(t,x,dummy,mu)
rhs=[x(2); mu*(1-x(1)^2)*x(2)-x(1)];