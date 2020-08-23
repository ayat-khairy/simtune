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

import os
from multi_gp import MultiGP
import sys
import util
import tempfile
import copy
import numpy          as np
import numpy.random   as npr
import scipy.linalg   as spla
import scipy.stats    as sps
import scipy.optimize as spo
import cPickle
import multiprocessing
from Locker import *


def f(x):   # The rosenbrock function
    return .5*x[0]

def fprime(x):
    return np.array((-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1] - x[0]**2)))



#print (sp.fmin_l_bfgs_b(f, [2, 2], fprime=fprime))
def optimize_pt(c, b, args, model):
    #import scipy.optimize as spo
    #imp = importlib.import_module ('scipy.optimize')

 try:
    #imp = __import__('importlib')
#    module  = imp.import_module ('frank_multi')
#    print ("loaded modeule path >>>> " , module)
#    myinstance = xclass()
    #global ret
    print (">>>> model >>> " , args)

    ret = spo.fmin_l_bfgs_b(
#            f,
            model.ei_optim,
            c.flatten(),
            args=args,
#            x0= [2,2],
 #           args = [],
            bounds=b,
            disp=0 )
#,approx_grad = True)
    print (">>>> model >>> " , model.ei_optim)
#    ret = spo.fmin_l_bfgs_b(f, [2, 2], bounds =b ,disp=0, fprime=fprime)
    print (">>>>>>>return " , ret)
 except Exception:
    
    print (sys.exc_info() [0])
 return ret[0]
 
    

def init(expt_dir, arg_string):
    args = util.unpack_args(arg_string)
    return MultiTaskEIOptChooser(expt_dir, **args)
"""
Chooser module for the Gaussian process expected improvement (EI)
acquisition function where points are sampled densely in the unit
hypercube and then a subset of the points are optimized to maximize EI
over hyperparameter samples.  Slice sampling is used to sample
Gaussian process hyperparameters.
"""
class MultiTaskEIOptChooser:

  

    def __init__(self, expt_dir, **kwargs):
        model = eval(kwargs.get('model','MultiGP'))

        self.grid_subset     = kwargs.get('grid_subset',20)
        self.model           = model(**kwargs)
        self.has_started     = False
        self.task_num        = int(kwargs.get('task_num',0))

        # Variables for bookkeeping
        self.locker          = Locker()
        self.state_pkl       = os.path.join(expt_dir, self.__module__ + ".pkl")
        self.stats_file      = os.path.join(expt_dir, 
                                   self.__module__ + "_hyperparameters.txt")
        self.best_by_task = list()
        self.evals_by_task = list()

        # Number of points to optimize EI over
        self.grid_subset     = int(self.grid_subset)

    def _index_map(self, u, items):
        return (np.floor((1-np.finfo(float).eps) * u * np.float(items))).astype('int')

    def _index_unmap(self, u, items):
        return float(float(u+1) / float(items))

    # Given a set of completed 'experiments' in the unit hypercube with
    # corresponding objective 'values', pick from the next experiment to
    # run according to the acquisition function.
#    @profile
    def next(self, grid, values, durations,
             candidates, pending, complete, valid=None, vmap=None):

        #print 'Candidates: %s' % candidates
        #print 'Pending: %s' % pending
        #print 'Complete: %s' % complete

        # Grab out the relevant sets.
        comp    = grid[complete,:]
        cand    = grid[candidates,:]
        pend    = grid[pending,:]
        vals    = values[complete]
        numcand = cand.shape[0]

        self.num_tasks = vmap.get_variable_property('Task','max') + 1

        # We find out which examples belong to which task from the first
        # dimension of the input
        self.taskindices = self._index_map(comp[:,0], self.num_tasks).astype(int)

        # Keep only the candidates for the current task.  This is maybe
        # a bit of a hack but in the future we can have candidates span
        # tasks.
        cand_indices = self._index_map(cand[:,0], self.num_tasks)
        inds         = np.equal(cand_indices, self.task_num)
        cand         = cand[inds,:]
        candidates   = candidates[inds]
        numcandsleft = cand.shape[0]

        # Pull off the first dimension indicating the task
        comp = comp[:,1:]
        cand = cand[:,1:]

        print 'Num cands: ' , cand.shape

        bests = np.zeros((1,self.num_tasks))
        evals = np.zeros((1,self.num_tasks))
        for i in xrange(self.num_tasks):
            tindex = np.equal(self.taskindices,i)
            if np.sum(tindex) == 0:
                best     = np.nan
                best_loc = np.nan
            else:
                best       = np.min(vals[tindex])
                best_index = best_index = vals[tindex].argmin()
                best_loc  = comp[tindex][best_index]
            bests[0,i] = best
            evals[0,i] = np.sum(tindex)
            
            print 'Task %d - best result: %f best location: %s (from %d evals)' % (i, best, best_loc, evals[0,i])


        self.best_by_task.append(bests)
        self.evals_by_task.append(evals)

        # Not enough observations anywhere. Pick the next job from the grid.
        if vals.shape[0] < 2:
            return int(candidates[0])

        ncomplete = np.equal(self.taskindices,self.task_num).sum()
        if ncomplete == 0:
            # No observations on the current task. Pick the best job from other tasks
            order = np.argsort(vals)
            newcand     = np.zeros(cand.shape[1]+1)
            newcand[0]  = self._index_unmap(self.task_num, self.num_tasks)
            newcand[1:] = comp[order[0],:]
            return (int(numcand), newcand)
        elif ncomplete == 1:
            # Pick the next job from the grid.
            return int(candidates[0])

        # Perform the real initialization.
        if not self.has_started:
            self.model.real_init(
                self.state_pkl, 
                self.locker,
                comp, 
                vals, 
                self.taskindices, 
                vmap.card()-1, 
                self.num_tasks)

            self.has_started = True

        if np.equal(self.taskindices,self.task_num).sum() <= 2:
            # Need to initialize new kernel.
            self.model.needs_burnin = True
            self.model.reset_hypers(self.taskindices, vals)

        # Spray a set of candidates around the min so far
        best_comp = np.argmin(vals[self.taskindices])
        cand2 = np.vstack((np.random.randn(10,comp.shape[1])*0.001 + 
                           comp[best_comp,:], cand))

        self.model.update(
                comp        = comp,
                vals        = vals,
                taskindices = self.taskindices,
                locker      = self.locker,
                state_pkl   = self.state_pkl,
                stats_file  = self.stats_file
                )

        overall_ei = self.model.ei_over_predictions(
                comp,
                vals,
                self.taskindices,
                cand2,
                self.task_num,
                compute_grad=False
                )

        inds  = np.argsort(overall_ei)[-self.grid_subset:]
        best_so_far = cand2[np.argmax(overall_ei),:]
        cand2 = cand2[inds,:]

        sys.stderr.write('Starting optimization with point %s...' % best_so_far)

        # Optimize each point in parallel
        b = []# optimization bounds
        for i in xrange(0, cand.shape[1]):
            b.append((0, 1))
        args = (comp,vals,self.taskindices,self.task_num)
        pool    = multiprocessing.Pool(self.grid_subset)
        results = [pool.apply_async(optimize_pt,args=(
                    c,b,args,copy.copy(self.model))) for c in cand2]
        print ("results >>> " , results[0])
        for res in results:
            cand = np.vstack((cand, res.get()))
        pool.close()
        
        overall_ei = self.model.ei_over_predictions(
                comp,
                vals,
                self.taskindices,
                cand,
                self.task_num,
                compute_grad=False
                )

        best_cand = np.argmax(overall_ei)

        sys.stderr.write('Finished optimization with point %s\n' % cand[best_cand,:])
        
        # Picked an optimized point
        if (best_cand >= numcandsleft):
            # Tell spearmint to run the given task at this point
            newcand     = np.zeros(cand.shape[1]+1)
            newcand[0]  = self._index_unmap(self.task_num, self.num_tasks)
            # Need to do this to prevent roundoff errors from optimization.
            newcand[1:] = np.maximum(np.minimum(cand[best_cand,:],1),0)
            return (int(numcand), newcand)

        # I think this is a bug...
        return int(candidates[best_cand])

