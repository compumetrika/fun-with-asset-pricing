"""
This code is modify from the DiscreteRV code from Quant-econ.net. See bottom of
file for reference and license of original code.

Filename: discrete_rv.py
Authors: Thomas Sargent, John Stachurski
Generates an array of draws from a discrete random variable with a
specified vector of probabilities.

Modified by Nathan Palmer, March 2016
"""
from __future__ import division
import numpy as np
from numpy import cumsum
from numpy.random import uniform


class DiscreteRV(object):
    """
    Generates an array of draws from a discrete random variable with
    vector of probabilities given by q.
    Parameters
    ----------
    q : array_like(float)
        Nonnegative numbers that sum to 1
    Attributes
    ----------
    q : see Parameters
    Q : array_like(float)
        The cumulative sum of q
    """

    def __init__(self, q, vals, seed=None):
        self._q = np.asarray(q)
        self.Q = cumsum(q)
        self.vals = np.array(vals)
        self.RNG = np.random.RandomState(seed)

    def __repr__(self):
        return "DiscreteRV with {n} elements".format(n=self._q.size)

    def __str__(self):
        return self.__repr__()

    @property
    def q(self):
        """
        Getter method for q.
        """
        return self._q

    @q.setter
    def q(self, val):
        """
        Setter method for q.
        """
        self._q = np.asarray(val)
        self.Q = cumsum(val)

    def draw(self, k=1):
        """
        Returns k draws from q.
        For each such draw, the value i is returned with probability
        q[i].
        Parameters
        -----------
        k : scalar(int), optional
            Number of draws to be returned
        Returns
        -------
        array_like(int)
            An array of k independent draws from q
        """
        idraws = self.RNG.uniform(0, 1, size=k)
        return self.vals[self.Q.searchsorted(idraws)]


# ------------------------------------------------------------------------------
# DiscreteRV and any additional QuantEcon code:
#
# Copyright (c) 2013, 2014 Thomas J. Sargent and John Stachurski: BSD-3
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
#  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
#  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
#  WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
