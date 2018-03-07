# -*- coding: utf-8 -*-
"""
* This file is part of FreeKiteSim.
*
* FreeKiteSim -- A kite-power system power simulation software.
* Copyright (C) 2013 by Uwe Fechner, Delft University
* of Technology, The Netherlands. All rights reserved.
*
* FreeKiteSim is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 3 of the License, or (at your option) any later version.
*
* FreeKiteSim is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with SystemOptimizer; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
"""
""" Fast implementation of the following linear algebra functions for 3-dimensional arrays:
    sum2, sum3, sub2, sub3, mul2, mul3, div2, div3, neg_sum, copy2
    cross, cross3, dot, norm, normalize, normalize1, normalize2
    The number behind the name is the number of parameters for the functions, that do
    not allocate memory. These functions always store the result in the last parameter.
    Average speed-up factor compared to numpy between 2 and 50, heavily machine dependant,
    but also dependent on the call context: If these procedures are called from another
    procedure, that was also compiled with Numba they can be much faster than when called
    from standard Python code.
    This version was tested with Numba 0.18. """
# TODO: add test case for dot with 3x3 rotation matrix
# pylint: disable=E0611
from numba import jit, double
import math
import numpy as np


@jit(nopython=True)
def sum2(vec, result):
	""" Calculate the sum of two 3d vectors and store the result in the second parameter. """
	result[0] = vec[0] + result[0]
	result[1] = vec[1] + result[1]
	result[2] = vec[2] + result[2]


@jit(nopython=True)
def sum3(vec1, vec2, result):
	""" Calculate the sum of two 3d vectors and store the result in the third parameter. """
	result[0] = vec1[0] + vec2[0]
	result[1] = vec1[1] + vec2[1]
	result[2] = vec1[2] + vec2[2]


@jit(nopython=True)
def sub2(vec, result):
	""" Calculate the difference of two 3d vectors and store the result in the second parameter. """
	result[0] = result[0] - vec[0]
	result[1] = result[1] - vec[1]
	result[2] = result[2] - vec[2]


@jit(nopython=True)
def sub3(vec1, vec2, result):
	""" Calculate the difference of two 3d vectors and store the result in the third parameter. """
	result[0] = vec1[0] - vec2[0]
	result[1] = vec1[1] - vec2[1]
	result[2] = vec1[2] - vec2[2]


@jit(nopython=True)
def mul2(a, result):
	""" Calculate the product of a scalar and a 3d vector and store the result in the second parameter."""
	result[0] = a * result[0]
	result[1] = a * result[1]
	result[2] = a * result[2]


@jit(nopython=True)
def mul3(a, vec, result):
	""" Calculate the product of a scalar and a 3d vector. """
	result[0] = a * vec[0]
	result[1] = a * vec[1]
	result[2] = a * vec[2]


@jit(nopython=True)
def div2(a, result):
	""" Divide a 3d vector by a scalar and store the result in the second parameter."""
	result[0] = result[0] / a
	result[1] = result[1] / a
	result[2] = result[2] / a


@jit(nopython=True)
def div3(a, vec, result):
	""" Divide a 3d vector by a scalar and store the result in the third parameter. """
	result[0] = vec[0] / a
	result[1] = vec[1] / a
	result[2] = vec[2] / a


@jit(nopython=True)
def neg_sum(a, b, c, result):
	""" Calculate the sum of three vectors and multiply the result with -1. """
	result[0] = -(a[0] + b[0] + c[0])
	result[1] = -(a[1] + b[1] + c[1])
	result[2] = -(a[2] + b[2] + c[2])


@jit(nopython=True)
def copy2(a, result):
	""" Calculate the difference of two 3d vectors. """
	result[0] = a[0]
	result[1] = a[1]
	result[2] = a[2]


@jit
def cross(vec1, vec2):
	""" Calculate the cross product of two 3d vectors. """
	result = np.zeros(3)
	return cross_(vec1, vec2, result)


@jit(nopython=True)
def cross_(vec1, vec2, result):
	""" Calculate the cross product of two 3d vectors. """
	a1, a2, a3 = double(vec1[0]), double(vec1[1]), double(vec1[2])
	b1, b2, b3 = double(vec2[0]), double(vec2[1]), double(vec2[2])
	result[0] = a2 * b3 - a3 * b2
	result[1] = a3 * b1 - a1 * b3
	result[2] = a1 * b2 - a2 * b1
	return result


@jit(nopython=True)
def cross3(vec1, vec2, result):
	""" Calculate the cross product of two 3d vectors. """
	a1, a2, a3 = double(vec1[0]), double(vec1[1]), double(vec1[2])
	b1, b2, b3 = double(vec2[0]), double(vec2[1]), double(vec2[2])
	result[0] = a2 * b3 - a3 * b2
	result[1] = a3 * b1 - a1 * b3
	result[2] = a1 * b2 - a2 * b1


@jit(nopython=True)
def dot(vec1, vec2):
	""" Calculate the dot product of two 3d vectors. """
	return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]


@jit(nopython=True)
def norm(vec):
	""" Calculate the norm of a 3d vector. """
	return math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])


@jit()
def normalize(vec):
	""" Calculate the normalized vector (norm: one). """
	result = np.copy(vec)
	return normalize1(result)


@jit(nopython=True)
def normalize1(vec):
	norm_ = norm(vec)
	if norm_ < 1e-6:
		vec[0] = 0.
		vec[1] = 0.
		vec[2] = 0.
	else:
		vec[0] = vec[0] / norm_
		vec[1] = vec[1] / norm_
		vec[2] = vec[2] / norm_
	return vec


@jit(nopython=True)
def normalize2(vec, result):
	norm_ = norm(vec)
	if norm_ < 1e-6:
		result[0] = 0.
		result[1] = 0.
		result[2] = 0.
	else:
		result[0] = vec[0] / norm_
		result[1] = vec[1] / norm_
		result[2] = vec[2] / norm_


def init():
	""" call all functions once to compile them """
	vec1, vec2 = np.array((1.0, 2.0, 3.0)), np.array((2.0, 3.0, 4.0))
	result = np.zeros(3)
	a = 1.24
	sum2(vec1, result)
	sum3(vec1, vec2, result)
	sub2(vec1, result)
	sub3(vec1, vec2, result)
	mul2(a, result)
	mul3(a, vec1, result)
	div2(a, result)
	div3(a, vec1, result)
	result = normalize(vec2)
	normalize1(vec1)
	normalize2(vec1, result)
	cross(vec1, vec2)
	cross3(vec1, vec2, result)
	dot(vec1, vec2)
	norm(vec1)


init()

"""
Results on i7-3770 CPU @ 3.40GHz

time for empty loop  0.000405073165894

time for sum2 with numba [µs]:    0.25
time sum with numpy [µs]:         0.53
speedup of sum2 with numba:       2.13

time for numba norm  [µs]:        0.19
time for linalg norm [µs]:        5.17
speedup of norm with numba:      27.25

time for numba dot [µs]:          0.27
time for numpy dot [µs]:          0.68
speedup of dot with numba:        2.47

time for numba cross  [µs]:       1.39
time for numba cross3 [µs]:       0.30
time for numpy cross  [µs]:      14.75
speedup of cross with numba:     48.37

time for numba normalize  [µs]:   1.83
time for numba normalize2 [µs]:   0.27
time for numpy normalize  [µs]:   7.54
speedup of normalize with numba: 27.93
"""