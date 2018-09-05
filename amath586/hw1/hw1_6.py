#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=5)


#<startTeX>
sol = solve_ivp(lambda t,rf: [(1-0.02*rf[1])*rf[0], (-1+0.03*rf[0])*rf[1]],[0,20],[20,20],method='RK45',max_step=.1)
#<endTeX>

t = sol.t
[R,F] = sol.y

plt.figure()
plt.plot(t,R,color='k',linestyle='-')
plt.plot(t,F,color='k',linestyle='--')
plt.savefig('img/6/solution.pdf')

plt.figure()
plt.plot(F,R,color='k')
plt.savefig('img/6/phase.pdf')
