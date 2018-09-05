addpath(fullfile(cd,'chebfun'))

d = domain(0,1);
L = chebop(@(x,u) -diff((1+x.^2).*diff(u)),d,0,0);
f = chebfun(@(x) 2*(3*x.^2-x+1),d);
u = L\f;

u_true = chebfun(@(x) x.*(1-x),d);

norm(u-u_true)
norm(u-u_true,'inf')

