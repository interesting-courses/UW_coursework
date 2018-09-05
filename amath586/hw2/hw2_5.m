% how does matlab files work...

[t,u] = ode23(@(t,u) [-1000*u(1) + u(2), -u(2)/10],[0,1],[1,2]);

hold on
plot(t,u(:,1))
plot(t,u(:,2))