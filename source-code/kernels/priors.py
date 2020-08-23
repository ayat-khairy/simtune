# -*- coding: utf-8 -*-
# Spearmint
#
# Academic and Non-Commercial Research Use Software License and Terms
# of Use
#
# Spearmint is a software package to perform Bayesian optimization
# according to specific algorithms (the “Software”).  The Software is
# designed to automatically run experiments (thus the code name
# 'spearmint') in a manner that iteratively adjusts a number of
# parameters so as to minimize some objective in as few runs as
# possible.
#
# The Software was developed by Ryan P. Adams, Michael Gelbart, and
# Jasper Snoek at Harvard University, Kevin Swersky at the
# University of Toronto (“Toronto”), and Hugo Larochelle at the
# Université de Sherbrooke (“Sherbrooke”), which assigned its rights
# in the Software to Socpra Sciences et Génie
# S.E.C. (“Socpra”). Pursuant to an inter-institutional agreement
# between the parties, it is distributed for free academic and
# non-commercial research use by the President and Fellows of Harvard
# College (“Harvard”).
#
# Using the Software indicates your agreement to be bound by the terms
# of this Software Use Agreement (“Agreement”). Absent your agreement
# to the terms below, you (the “End User”) have no rights to hold or
# use the Software whatsoever.
#
# Harvard agrees to grant hereunder the limited non-exclusive license
# to End User for the use of the Software in the performance of End
# User’s internal, non-commercial research and academic use at End
# User’s academic or not-for-profit research institution
# (“Institution”) on the following terms and conditions:
#
# 1.  NO REDISTRIBUTION. The Software remains the property Harvard,
# Toronto and Socpra, and except as set forth in Section 4, End User
# shall not publish, distribute, or otherwise transfer or make
# available the Software to any other party.
#
# 2.  NO COMMERCIAL USE. End User shall not use the Software for
# commercial purposes and any such use of the Software is expressly
# prohibited. This includes, but is not limited to, use of the
# Software in fee-for-service arrangements, core facilities or
# laboratories or to provide research services to (or in collaboration
# with) third parties for a fee, and in industry-sponsored
# collaborative research projects where any commercial rights are
# granted to the sponsor. If End User wishes to use the Software for
# commercial purposes or for any other restricted purpose, End User
# must execute a separate license agreement with Harvard.
#
# Requests for use of the Software for commercial purposes, please
# contact:
#
# Office of Technology Development
# Harvard University
# Smith Campus Center, Suite 727E
# 1350 Massachusetts Avenue
# Cambridge, MA 02138 USA
# Telephone: (617) 495-3067
# Facsimile: (617) 495-9568
# E-mail: otd@harvard.edu
#
# 3.  OWNERSHIP AND COPYRIGHT NOTICE. Harvard, Toronto and Socpra own
# all intellectual property in the Software. End User shall gain no
# ownership to the Software. End User shall not remove or delete and
# shall retain in the Software, in any modifications to Software and
# in any Derivative Works, the copyright, trademark, or other notices
# pertaining to Software as provided with the Software.
#
# 4.  DERIVATIVE WORKS. End User may create and use Derivative Works,
# as such term is defined under U.S. copyright laws, provided that any
# such Derivative Works shall be restricted to non-commercial,
# internal research and academic use at End User’s Institution. End
# User may distribute Derivative Works to other Institutions solely
# for the performance of non-commercial, internal research and
# academic use on terms substantially similar to this License and
# Terms of Use.
#
# 5.  FEEDBACK. In order to improve the Software, comments from End
# Users may be useful. End User agrees to provide Harvard with
# feedback on the End User’s use of the Software (e.g., any bugs in
# the Software, the user experience, etc.).  Harvard is permitted to
# use such information provided by End User in making changes and
# improvements to the Software without compensation or an accounting
# to End User.
#
# 6.  NON ASSERT. End User acknowledges that Harvard, Toronto and/or
# Sherbrooke or Socpra may develop modifications to the Software that
# may be based on the feedback provided by End User under Section 5
# above. Harvard, Toronto and Sherbrooke/Socpra shall not be
# restricted in any way by End User regarding their use of such
# information.  End User acknowledges the right of Harvard, Toronto
# and Sherbrooke/Socpra to prepare, publish, display, reproduce,
# transmit and or use modifications to the Software that may be
# substantially similar or functionally equivalent to End User’s
# modifications and/or improvements if any.  In the event that End
# User obtains patent protection for any modification or improvement
# to Software, End User agrees not to allege or enjoin infringement of
# End User’s patent against Harvard, Toronto or Sherbrooke or Socpra,
# or any of the researchers, medical or research staff, officers,
# directors and employees of those institutions.
#
# 7.  PUBLICATION & ATTRIBUTION. End User has the right to publish,
# present, or share results from the use of the Software.  In
# accordance with customary academic practice, End User will
# acknowledge Harvard, Toronto and Sherbrooke/Socpra as the providers
# of the Software and may cite the relevant reference(s) from the
# following list of publications:
#
# Practical Bayesian Optimization of Machine Learning Algorithms
# Jasper Snoek, Hugo Larochelle and Ryan Prescott Adams
# Neural Information Processing Systems, 2012
#
# Multi-Task Bayesian Optimization
# Kevin Swersky, Jasper Snoek and Ryan Prescott Adams
# Advances in Neural Information Processing Systems, 2013
#
# Input Warping for Bayesian Optimization of Non-stationary Functions
# Jasper Snoek, Kevin Swersky, Richard Zemel and Ryan Prescott Adams
# Preprint, arXiv:1402.0929, http://arxiv.org/abs/1402.0929, 2013
#
# Bayesian Optimization and Semiparametric Models with Applications to
# Assistive Technology Jasper Snoek, PhD Thesis, University of
# Toronto, 2013
#
# 8.  NO WARRANTIES. THE SOFTWARE IS PROVIDED "AS IS." TO THE FULLEST
# EXTENT PERMITTED BY LAW, HARVARD, TORONTO AND SHERBROOKE AND SOCPRA
# HEREBY DISCLAIM ALL WARRANTIES OF ANY KIND (EXPRESS, IMPLIED OR
# OTHERWISE) REGARDING THE SOFTWARE, INCLUDING BUT NOT LIMITED TO ANY
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, OWNERSHIP, AND NON-INFRINGEMENT.  HARVARD, TORONTO AND
# SHERBROOKE AND SOCPRA MAKE NO WARRANTY ABOUT THE ACCURACY,
# RELIABILITY, COMPLETENESS, TIMELINESS, SUFFICIENCY OR QUALITY OF THE
# SOFTWARE.  HARVARD, TORONTO AND SHERBROOKE AND SOCPRA DO NOT WARRANT
# THAT THE SOFTWARE WILL OPERATE WITHOUT ERROR OR INTERRUPTION.
#
# 9.  LIMITATIONS OF LIABILITY AND REMEDIES. USE OF THE SOFTWARE IS AT
# END USER’S OWN RISK. IF END USER IS DISSATISFIED WITH THE SOFTWARE,
# ITS EXCLUSIVE REMEDY IS TO STOP USING IT.  IN NO EVENT SHALL
# HARVARD, TORONTO OR SHERBROOKE OR SOCPRA BE LIABLE TO END USER OR
# ITS INSTITUTION, IN CONTRACT, TORT OR OTHERWISE, FOR ANY DIRECT,
# INDIRECT, SPECIAL, INCIDENTAL, CONSEQUENTIAL, PUNITIVE OR OTHER
# DAMAGES OF ANY KIND WHATSOEVER ARISING OUT OF OR IN CONNECTION WITH
# THE SOFTWARE, EVEN IF HARVARD, TORONTO OR SHERBROOKE OR SOCPRA IS
# NEGLIGENT OR OTHERWISE AT FAULT, AND REGARDLESS OF WHETHER HARVARD,
# TORONTO OR SHERBROOKE OR SOCPRA IS ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGES.
#
# 10. INDEMNIFICATION. To the extent permitted by law, End User shall
# indemnify, defend and hold harmless Harvard, Toronto and Sherbrooke
# and Socpra, their corporate affiliates, current or future directors,
# trustees, officers, faculty, medical and professional staff,
# employees, students and agents and their respective successors,
# heirs and assigns (the "Indemnitees"), against any liability,
# damage, loss or expense (including reasonable attorney's fees and
# expenses of litigation) incurred by or imposed upon the Indemnitees
# or any one of them in connection with any claims, suits, actions,
# demands or judgments arising from End User’s breach of this
# Agreement or its Institution’s use of the Software except to the
# extent caused by the gross negligence or willful misconduct of
# Harvard, Toronto or Sherbrooke or Socpra. This indemnification
# provision shall survive expiration or termination of this Agreement.
#
# 11. GOVERNING LAW. This Agreement shall be construed and governed by
# the laws of the Commonwealth of Massachusetts regardless of
# otherwise applicable choice of law standards.
#
# 12. NON-USE OF NAME.  Nothing in this License and Terms of Use shall
# be construed as granting End Users or their Institutions any rights
# or licenses to use any trademarks, service marks or logos associated
# with the Software.  You may not use the terms “Harvard” or
# “University of Toronto” or “Université de Sherbrooke” or “Socpra
# Sciences et Génie S.E.C.” (or a substantially similar term) in any
# way that is inconsistent with the permitted uses described
# herein. You agree not to use any name or emblem of Harvard, Toronto
# or Sherbrooke, or any of their subdivisions for any purpose, or to
# falsely suggest any relationship between End User (or its
# Institution) and Harvard, Toronto and/or Sherbrooke, or in any
# manner that would infringe or violate any of their rights.
#
# 13. End User represents and warrants that it has the legal authority
# to enter into this License and Terms of Use on behalf of itself and
# its Institution.


import numpy as np
import scipy.stats as sps
from operator import add # same as lambda x,y:x+y I think
# import scipy.special.gammaln as log_gamma

def prior_factory(prior_fun, prior_params):
    if prior_params is None:
        return prior_fun

    if not type(prior_params) == list:
        prior_params = [prior_params]

    def prior(hypers):
        return prior_fun(hypers, *prior_params)

    return prior



# This should do the same thing as above, right? ok one takes the args as more args, the other as a list I WILL WORRY ABOUT THIS LATER 
def prior_factory_mike(prior_fun, *prior_params):
    return lambda hypers: prior_fun(hypers, *prior_params)

# A factory that lets you compose (multiply) priors. 
# But the product of the priors is the sum of these log-priors. So we add them below.
# Input should be a list of tuples, each tuple of the form (prior_fun, prior_params)
# for example [(tophat, 0, 1), (horseshoe, 5)]
# Gives tophat(hypers, 0, 1) + horseshoe(hypers, 5)
def prior_factory_compose(list_of_priors):
    funs = [prior_factory_mike(*p) for p in list_of_priors]
    return lambda hypers: reduce(add, map(lambda x: x(hypers), funs))



def tophat(params, pmin, pmax):
    if np.any(params < pmin) or np.any(params > pmax):
        return -np.inf

    return 0.

compwise_tophat = tophat # Deprecated


def nonnegative(params):
    if np.any(params < 0):
        return -np.inf

    return 0.
    
def step(params, pmin):
    if np.any(params < pmin):
        return -np.inf

    return 0.



def horseshoe(params, scale):
    return np.sum(np.log(np.log(1 + (scale/params)**2)))

def one_sided_horseshoe(params, pmin, scale):
    if np.any(params < pmin):
        return -np.inf

    return horseshoe(params, scale)


def lognormal(params, scale, mean=0):
    # lognormal only for x>0
    if np.any(params <= 0):
        return -np.inf

    return -np.sum(np.log(params))-0.5*np.sum((np.log(params-mean)/scale)**2)
# Note on the lognormal: it is 0 as params-->0. Also,
# The derivative of the above w.r.t. the params is -1/x [1 + log(x)/scale^4], so
# this thing is NOT monotonically decreasing for ANY choice of scale. 
# (It's 0 at 0, goes up, and then back down).
# Another note: we used to use a bugged version that did not contain the first term
# namely, -0.5*np.sum((np.log(params-mean)/scale)**2)

# When used as a prior for a variable x, then the effective prior on sqrt(x) is lognormal
def lognormal_on_square(params, scale, mean=0):
    if np.any(params <= 0):
        return -np.inf

    return lognormal(np.sqrt(params), scale, mean=mean)


# See comment above.
def one_sided_lognormal_on_square(params, pmin, scale, mean=0):
    if np.any(params < pmin):
        return -np.inf
    
    return lognormal_on_square(params, scale, mean=mean)

def loglogistic(params, shape, scale=1):
    if np.any(params <= 0):
        return -np.inf

    return np.sum(sps.fisk.logpdf(params, shape, scale=scale))

def loglogistic_on_square(params, shape, scale=1):
    if np.any(params <= 0):
        return -np.inf

    return loglogistic(np.sqrt(params), shape, scale=scale)

def one_sided_loglogistic_on_square(params, pmin, scale=1):
    if np.any(params < pmin):
        return -np.inf
    
    return loglogistic_on_square(params, pmin, scale=scale)


# This is the form (1/mean) exp(-x/mean)
# not the form k*exp(-kx)
def exponential(params, mean):
    if np.any(params < 0):
        return -np.inf
    
    return -params/mean
    
# This distribution is like the chi distribution but TO THE POWER OF TWO!!!!!!!!!!!!!!!!!!
def chi_squared(params, k):
    if np.any(params < 0):
        return -np.inf

    return (k/2.-1)*np.log(params) - params/2.

# A sigmoidal function that is zero at x=0 and approaches 1 as x --> +/- inf
# It is an even (symmetric) function and equals 1/2 when x = transition_loc.
# It looks sort of like this:
#  |
# 1|               *******************
#  |         ****  
#  |     **
#  |   *
#  | *
# 0|__________________________________

def arbitrary_sigmoid_thing(params, sharpness, transition_loc):
    return np.sum(np.log(abs(params)**sharpness/(transition_loc**sharpness + abs(params)**sharpness)))

def gaussian(params, mean, scale):
    return np.sum(-0.5*((params-mean)/scale)**2)

def one_sided_gaussian(params, mean, scale):
    if np.any(np.less_equal(params, 0)):
        return -np.inf
    return np.sum(-0.5*((params-mean)/scale)**2)

def none(params):
    return 0.

