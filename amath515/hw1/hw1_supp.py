# This is supplementary material for Homework 1
import numpy as np
from scipy.special import factorial

# line search function
# -----------------------------------------------------------------------------
def lineSearch(x, g, p, func, t=0.5):
	"""
	Line Search Function
	simple descent line search

	input
	-----
	x : array_like
		Base point.
	g : array_like
		Gradient on the base point.
	p : array_like
		Given descent direction.
	func : function
		Input x and return funciton value.
	t : float, optional
		step size shrink ratio

	output
	------
	step_size : float or None
		When sucess return the step_size, otherwise return None.
	"""
	# initial step size
	step_size = 1.0
	# 
	m = g.dot(p)
	if m < 0:
		print('Line search: not a descent direction.')
		return None
	#
	f = func(x)
	y = x - step_size*p
	#
	while func(y) > f:
		step_size *= t
		if step_size < 1e-15:
			print('Line search: step size too small.')
			return None
		y = x - step_size*p
	#
	return step_size

# line search function - armijo line search
# -----------------------------------------------------------------------------
def lineSearch_armijo(x, g, p, func, c=0.01, t=0.5):
	"""
	Line Search Function
	armijo line search

	input
	-----
	x : array_like
		Base point.
	g : array_like
		Gradient on the base point.
	p : array_like
		Given descent direction.
	func : function
		Input x and return funciton value.
	c : float, optional
		has to strictly be between 0 and 1
	t : float, optional
		step size shrink ratio

	output
	------
	step_size : float or None
		When sucess return the step_size, otherwise return None.
	"""
	# save guard for c
	assert 0 < c < 1, 'c needs to strictly be in 0 and 1'
	# initial step size
	step_size = 1.0
	# 
	m = c*g.dot(p)
	if m < 0:
		print('Line search: not a descent direction.')
		return None
	#
	f = func(x)
	y = x - step_size*p
	#
	while func(y) > f - step_size*m:
		step_size *= t
		if step_size < 1e-15:
			print('Line search: step size too small.')
			return None
		y = x - step_size*p
	#
	return step_size


# sample logistic data
# -----------------------------------------------------------------------------
def sampleLGT(x, A):
	y = A.dot(x)
	p = 1.0/(1.0 + np.exp(-y))
	q = np.random.rand(y.size)
	b = np.zeros(y.size)
	b[q <= p] = 1.0
	return b

# sample Poisson data
# -----------------------------------------------------------------------------
def samplePSN(x, A):
	y = A.dot(x)
	p = np.random.rand(y.size)*np.exp(y)
	b = np.zeros(y.size)
	for i in range(b.size):
		k = 0
		q = p[i]**k/factorial(k)
		while p[i] > q:
			k += 1
			q += p[i]**k/factorial(k)
		b[i] = k
	return b
