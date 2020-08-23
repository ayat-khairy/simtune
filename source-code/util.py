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

import re
import numpy        as np
import numpy.random as npr
import math
import sys

# This should eventually go somewhere else, but I don't know where yet
TERMINATION_SIGNAL = "terminate"



def slice_sample(init_x, logprob, sigma=1.0, step_out=True, max_steps_out=1000, 
                 compwise=False, verbose=False, returnFunEvals=False):
    
    # Keep track of the number of evaluations of the logprob function
    funEvals = {'funevals': 0} # sorry, i don't know how to actually do this properly with all these nested function. pls forgive me -MG
    
    # this is a 1-d sampling subrountine that only samples along the direction "direction"
    def direction_slice(direction, init_x):
        
        def dir_logprob(z): # logprob of the proposed point (x + dir*z) where z must be the step size
            funEvals['funevals'] += 1
            try:
                return logprob(direction*z + init_x)
            except:
                print 'ERROR: Logprob failed at input %s' % str(direction*z + init_x)
                raise
                
    
        # upper and lower are step sizes -- everything is measured relative to init_x
        upper = sigma*npr.rand()  # random thing above 0
        lower = upper - sigma     # random thing below 0
        llh_s = np.log(npr.rand()) + dir_logprob(0.0)  # = log(prob_current * rand) 
        # (above) uniformly sample vertically at init_x
    
    
        l_steps_out = 0
        u_steps_out = 0
        if step_out:
            # increase upper and decrease lower until they overshoot the curve
            while dir_logprob(lower) > llh_s and l_steps_out < max_steps_out:
                l_steps_out += 1
                lower       -= sigma  # make lower smaller by sigma
            while dir_logprob(upper) > llh_s and u_steps_out < max_steps_out:
                u_steps_out += 1
                upper       += sigma
        
        
        # rejection sample along the horizontal line (because we don't know the bounds exactly)
        steps_in = 0
        while True:
            steps_in += 1
            new_z     = (upper - lower)*npr.rand() + lower  # uniformly sample between upper and lower
            new_llh   = dir_logprob(new_z)  # new current logprob
            if np.isnan(new_llh):
                print new_z, direction*new_z + init_x, new_llh, llh_s, init_x, logprob(init_x)
                raise Exception("Slice sampler got a NaN logprob")
            if new_llh > llh_s:  # this is the termination condition
                break       # it says, if you got to a better place than you started, you're done
                
            # the below is only if you've overshot, meaning your uniform sample from the horizontal
            # slice ended up outside the curve because the bounds lower and upper were not tight
            elif new_z < 0:  # new_z is to the left of init_x
                lower = new_z  # shrink lower to it
            elif new_z > 0:
                upper = new_z
            else:  # upper and lower are both 0...
                raise Exception("Slice sampler shrank to zero!")

        if verbose:
            print "Steps Out:", l_steps_out, u_steps_out, " Steps In:", steps_in, "Final logprob:", new_llh

        # return new the point
        return new_z*direction + init_x  

    
    # begin main
    
    # # This causes an extra "logprob" function call -- might want to turn off for speed
    initial_llh = logprob(init_x)
    if verbose:
        sys.stderr.write('Logprob before sampling: %f\n' % initial_llh)
    if np.isneginf(initial_llh):
        sys.stderr.write('Values passed into slice sampler: %s\n' % init_x)
        raise Exception("Initial value passed into slice sampler has logprob = -inf")
    
    if not init_x.shape:  # if there is just one dimension, stick it in a numpy array
        init_x = np.array([init_x])

    dims = init_x.shape[0]
    if compwise:   # if component-wise (independent) sampling
        ordering = range(dims)
        npr.shuffle(ordering)
        cur_x = init_x.copy()
        for d in ordering:
            direction    = np.zeros((dims))
            direction[d] = 1.0
            cur_x = direction_slice(direction, cur_x)
            
    else:   # if not component-wise sampling
        direction = npr.randn(dims)
        direction = direction / np.sqrt(np.sum(direction**2))  # pick a unit vector in a random direction
        cur_x = direction_slice(direction, init_x)  # attempt to sample in that direction
    
    return (cur_x, funEvals['funevals']) if returnFunEvals else cur_x
    



# xx: the initial point
# sample_nu: a function that samples from the multivariate Gaussian prior
# log_like_fn: a function that computes the log likelihood of an input
# cur_log_like (optional): the current log likelihood
# angle_range: not sure
def elliptical_slice(xx, sample_nu, log_like_fn, cur_log_like=None, angle_range=0):
    D = xx.shape[0]

    if cur_log_like is None:
        cur_log_like = log_like_fn(xx)

    if np.isneginf(cur_log_like):
        raise Exception("Elliptical Slice Sampler: initial logprob is -inf for inputs %s" % xx)
    if np.isnan(cur_log_like):
        raise Exception("Elliptical Slice Sampler: initial logprob is NaN for inputs %s" % xx)

    nu = sample_nu()
    hh = np.log(npr.rand()) + cur_log_like  
    # log likelihood threshold -- LESS THAN THE INITIAL LOG LIKELIHOOD

    # Set up a bracket of angles and pick a first proposal.
    # "phi = (theta'-theta)" is a change in angle.
    if angle_range <= 0:
        # Bracket whole ellipse with both edges at first proposed point
        phi = npr.rand()*2*math.pi
        phi_min = phi - 2*math.pi
        phi_max = phi
    else:
        # Randomly center bracket on current point
        phi_min = -angle_range*npr.rand();
        phi_max = phi_min + angle_range;
        phi = npr.rand()*(phi_max - phi_min) + phi_min;

    # Slice sampling loop
    while True:
        # Compute xx for proposed angle difference 
        # and check if it's on the slice
        xx_prop = xx*np.cos(phi) + nu*np.sin(phi)

        cur_log_like = log_like_fn(xx_prop)
        
        if cur_log_like > hh:
            # New point is on slice, ** EXIT LOOP **
            return xx_prop, cur_log_like

        # Shrink slice to rejected point
        if phi > 0:
            phi_max = phi
        elif phi < 0:
            phi_min = phi
        else:
            sys.stderr.write('Initial x: %s\n' % xx)
            # sys.stderr.write('initial log like = %f\n' % initial_log_like)
            sys.stderr.write('Proposed x: %s\n' % xx_prop)
            sys.stderr.write('ESS log lik = %f\n' % cur_log_like)
            raise Exception('BUG DETECTED: Shrunk to current position '
                            'and still not acceptable.');

        # Propose new angle difference
        phi = npr.rand()*(phi_max - phi_min) + phi_min









# Check the gradients of function "fun" at location(s) "test_x"
#   fun: a function that takes in test_x and returns a tuple of the form (function value, gradient)
#       # the function value should have shape (n_points) and  the gradient should have shape (n_points, D)
#   test_x: the points. should have shape (n_points by D)
#       special case: if test_x is a single *flat* array, then gradient should also be a flat array
#   delta: finite difference step length
#   error_tol: tolerance for error of numerical to symbolic gradient
# Returns a boolean of whether or not the gradients seem OK
def check_grad(fun, test_x, error_tol=1e-3, delta=1e-5, verbose=False):
    if verbose:
        sys.stderr.write('Checking gradients...\n')
        
    state_before_checking = npr.get_state()
    fixed_seed = 5      # arbitrary
    
    npr.seed(fixed_seed)
    analytical_grad = fun(test_x)[1]
    D = test_x.shape[1] if test_x.ndim > 1 else test_x.size
    grad_check = np.zeros(analytical_grad.shape) if analytical_grad.size > 1 else np.zeros(1)
    for i in range(D):
        unit_vector = np.zeros(D)
        unit_vector[i] = delta
        npr.seed(fixed_seed)
        forward_val = fun(test_x + unit_vector)[0]
        npr.seed(fixed_seed)
        backward_val = fun(test_x - unit_vector)[0]
        grad_check_i = (forward_val - backward_val)/(2*delta)
        if test_x.ndim > 1:
            grad_check[:,i] = grad_check_i
        else:
            grad_check[i] = grad_check_i
    grad_diff = grad_check - analytical_grad
    err = np.sqrt(np.sum(grad_diff**2))

    if verbose:        
        sys.stderr.write('Analytical grad: %s\n' % str(analytical_grad))
        sys.stderr.write('Estimated grad:  %s\n' % str(grad_check))
        sys.stderr.write('L2-norm of gradient error = %g\n' % err)

    npr.set_state(state_before_checking)

    return err < error_tol



# For converting a string of args into a dict of args
# (one could then call parse_args on the output)
def unpack_args(str):
    if len(str) > 1:
        eq_re = re.compile("\s*=\s*")
        return dict(map(lambda x: eq_re.split(x),
                        re.compile("\s*,\s*").split(str)))
    else:
        return {}
            
# For parsing the input arguments to a Chooser. 
# "argTypes" is a dict with keys of argument names and
# values of tuples with the (argType, argDefaultValue)
# args is the dict of arguments passd in by the used
def parse_args(argTypes, args):
    opt = dict() # "options"
    for arg in argTypes:
        if arg in args:
            try:
                opt[arg] = argTypes[arg][0](args[arg])
            except:
                # If the argument cannot be parsed into the data type specified by argTypes (e.g., float)
                sys.stderr.write("Cannot parse user-specified value %s of argument %s" % (args[arg], arg))
        else:
            opt[arg] = argTypes[arg][1]
 
    return opt
