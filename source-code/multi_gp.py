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
import sys
import ast
import util
import tempfile
import cPickle
import numpy             as np
import numpy.random      as npr
import scipy.linalg      as spla
import scipy.stats       as sps

import kernels.matern       as matern
import kernels.sphere       as sphere
import kernels.multikernel  as multikernel
import kernels.priors       as priors
import kernels.simple_means as simple_means
        
class MultiGP(object):
    def __init__(self, **kwargs):
        self.mcmc_iters     = int(kwargs.get('mcmc_iters',10))
        self.num_dimensions = kwargs.get('num_dimensions')
        self.burnin         = int(kwargs.get('burnin',100))
        self.needs_burnin   = True
        self.num_samples    = self.mcmc_iters
        self.num_tasks      = kwargs.get('num_tasks')
        self.noiseless      = bool(ast.literal_eval(kwargs.get('noiseless', "False")))

    def init_kernels(self, taskindices, values):
        self.input_kernel = matern.Matern52(
                self.num_dimensions,
                ls_prior        = priors.tophat,
                ls_prior_params = [0,10],
                ls_default      = 1
                )

        self.task_kernel = sphere.Sphere(
                self.num_tasks,
                theta_prior        = priors.tophat,
                theta_prior_params = [0,1],
                theta_default      = 0.5
                )

        self.kernel = multikernel.MultiKernel(
                    self.input_kernel,
                    self.task_kernel,
                    amp_prior          = priors.lognormal,
                    amp_prior_params   = [1],
                    amp_default        = 1,
                    noise_prior        = priors.one_sided_horseshoe,
                    noise_prior_params = [1e-6,0.1],
                    noise_default      = 1e-6
                    )

        # Initial amplitudes
        for i in range(self.num_tasks):
                ind = (taskindices==i)
                if ind.sum() > 1:
                    self.kernel.amps[i] = np.sqrt(np.std(values[ind]))

    def init_mean(self, taskindices, values):
        self.mean = simple_means.MultiConstantMean(self.num_tasks)
        # Initial mean.
        for i in range(self.num_tasks):
            ind = (taskindices==i)
            if ind.sum() > 0:
                self.mean.means[i] = values[ind].mean()

    def real_init(self, state_pkl, locker, comp, values,
                  taskindices, num_dimensions, num_tasks):
        sys.stderr.write("Waiting to lock hyperparameter pickle...")
        locker.lock_wait(state_pkl)
        sys.stderr.write("...acquired\n")

        self.num_dimensions = num_dimensions
        self.num_tasks      = num_tasks
        self.init_kernels(taskindices, values)
        self.init_mean(taskindices, values)

        if os.path.exists(state_pkl):
            fh    = open(state_pkl, 'r')
            state = cPickle.load(fh)
            fh.close()

            self.mean.from_hypers(state['mean'])
            #self.ensure_mean_is_valid(taskindices, values)
            self.kernel.from_hypers(state['kernel'])
            self.input_kernel.from_hypers(state['input_kernel'])
            self.task_kernel.from_hypers(state['task_kernel'])

            self.needs_burnin = False
            
    def update(self,**kwargs):
        comp        = kwargs.get('comp')
        vals        = kwargs.get('vals')
        taskindices = kwargs.get('taskindices')
        locker      = kwargs.get('locker')
        state_pkl   = kwargs.get('state_pkl')
        stats_file  = kwargs.get('stats_file')
        
        # Possibly burn in.
        if self.needs_burnin:
            for mcmc_iter in xrange(self.burnin):
                self.sample_hypers(comp, vals, taskindices)
                sys.stderr.write("BURN %d/%d\n"% (mcmc_iter+1, self.burnin))
                self.mean.print_diagnostic()
                self.kernel.print_diagnostic()
                self.input_kernel.print_diagnostic()
                self.task_kernel.print_diagnostic()
            self.needs_burnin = False

        # Sample from hyperparameters.
        # Adjust the candidates to hit ei peaks
        self.mean.means_samples        = []
        self.kernel.amps_samples       = []
        self.kernel.noises_samples     = []
        self.input_kernel.ls_samples   = []
        self.task_kernel.theta_samples = []
        for mcmc_iter in xrange(self.mcmc_iters):
            self.sample_hypers(comp, vals, taskindices)
            self.mean.append_sample()
            self.kernel.append_sample()
            self.input_kernel.append_sample()
            self.task_kernel.append_sample()

            sys.stderr.write("%d/%d\n"% (mcmc_iter+1, self.mcmc_iters))
            self.mean.print_diagnostic()
            self.kernel.print_diagnostic()
            self.input_kernel.print_diagnostic()
            self.task_kernel.print_diagnostic()

        self.dump_hypers(locker, state_pkl, stats_file)

    def reset_hypers(self, taskindices, values):
        self.init_mean(taskindices, values)
        self.init_kernels(taskindices, values)

    def cov(self, x1, taskindicesx1, x2=None, taskindicesx2=None, params=None):
        if params is not None:
            ls           = params['ls']
            theta        = params['theta']
            amps         = params['amps']
            noises       = params['noises']

            self.input_kernel.ls   = ls
            self.task_kernel.theta = theta
            self.kernel.amps       = amps
            self.kernel.noises     = noises
                
        return self.kernel.kernel(x1, taskindicesx1, x2, taskindicesx2)

    def logprob(self, x, y, taskindicesx, params=None):
        if params is not None:
            means = params['means']

            self.mean.means = means

        C     = self.cov(x, taskindicesx, params=params)
        chol  = spla.cholesky(C, lower=True)
        solve = spla.cho_solve((chol, True), y - self.mean.extended_means(taskindicesx))
        lp    = -np.sum(np.log(np.diag(chol)))-0.5*np.dot(y - self.mean.extended_means(taskindicesx), solve)
        return lp

    def logprob_fun(self, params):
        # This is a bit hacky, but I create this function this way
        # because pickle doesn't allow nested functions in classes.
        x            = self.logprob_data['x']
        y            = self.logprob_data['y']
        taskindicesx = self.logprob_data['taskindicesx']
        return self.logprob(x, y, taskindicesx, params=params)

    
    def predict(self, x1, y, taskindicesx1, x2, taskindicesx2, compute_grad=False):
        print (">>>>> predicting >>>")
        if compute_grad:
            assert x2.shape[0] == 1, 'Gradient computation must involve one candidate.'

        if x2.ndim == 1:
            x2 = x2[None,:]

        # The primary covariances for prediction.
        x1_x1_cov = self.kernel.kernel(x1, taskindicesx1)      
        x1_x2_cov = self.kernel.kernel(x1, taskindicesx1,
                                       x2, taskindicesx2)
        print (">>>>1 >>>>")
        # Compute the required Cholesky.
        x1_x1_cov_chol = spla.cholesky(x1_x1_cov, lower=True)
        print (">>>>>2 >>>>>>>>")
        # Solve the linear systems.
        alpha  = spla.cho_solve((x1_x1_cov_chol, True), y - self.mean.extended_means(taskindicesx1))
        beta   = spla.solve_triangular(x1_x1_cov_chol, x1_x2_cov, lower=True)
        print (">>>>> 3>>>>>>" )
        print ("comute_grad >>>> " , compute_grad)
        # Predict the marginal means and variances at candidates.
        func_m = np.dot(x1_x2_cov.T, alpha) + self.mean.extended_means(taskindicesx2)
        func_v = self.kernel.kernel(x2, taskindicesx2, diag_test=True) - np.sum(beta**2, axis=0)
        print (">>>>>>4>>>>>>>")
        if not compute_grad:
            return func_m, func_v

        x1_x2_cov_grad = self.kernel.kernel_grad(x1, taskindicesx1, x2, taskindicesx2)
        print (">>>> x1_x2_cov_grad >>>> " , x1_x2_cov_grad)
        print (">>>>>>>>>>>>>>>5 >>>>>>>>>>>>>>>>>")
        # Assumes one candidate (x2 must have one point)
        grad_cross = np.squeeze(x1_x2_cov_grad)
        
        grad_xp_m = np.dot(alpha.transpose(),grad_cross)
        grad_xp_v = np.dot(-2*spla.cho_solve(
                (x1_x1_cov_chol, True), x1_x2_cov).ravel(), grad_cross)
        print (">>>>>> 6 >>>.")
        return func_m, func_v, grad_xp_m, grad_xp_v
    def set_sample(self, sample_num):
        if sample_num < self.num_samples:
            self.mean.set_sample(sample_num)
            self.kernel.set_sample(sample_num)
            self.input_kernel.set_sample(sample_num)
            self.task_kernel.set_sample(sample_num)
        else:
            raise Exception('Sample number out of bounds.')

    def sample_hypers(self, x, y, taskindicesx):
        params = {
                'means'  : self.mean.means,
                'amps'   : self.kernel.amps,
                'noises' : self.kernel.noises,
                'theta'  : self.task_kernel.theta,
                'ls'     : self.input_kernel.ls
                }
                
        self.logprob_data = {
                'x'            : x,
                'y'            : y,
                'taskindicesx' : taskindicesx
                }

        y_mins = -np.inf*np.ones(self.num_tasks)
        y_maxs = np.inf*np.ones(self.num_tasks)
        for i in range(self.num_tasks):
            ind = (taskindicesx==i)
            if ind.sum() > 0:
                y_mins[i] = y[ind].min()
                y_maxs[i] = y[ind].max()
        self.mean.mean_prior_params = [y_mins, y_maxs]

        # Sample means, amps and noises together
        self.kernel.sample_hypers_and_means(
                self.logprob_fun,
                params,
                util.slice_sample,
                self.mean,
                self.noiseless
                )
        self.input_kernel.sample_hypers(self.logprob_fun, params, util.slice_sample)
        self.task_kernel.sample_hypers(self.logprob_fun, params, util.slice_sample)

        sys.stderr.write('Logprob: %s\n' % self.logprob(x, y, taskindicesx, params))

    def ei_optim(self, cand, comp, vals, taskindices_comp, task_num):
        print ("ei_optim")
        print (">>>> cand >>> " , cand)
        return self.ei_over_predictions(comp, vals, taskindices_comp, cand, task_num, compute_grad=True)

    def ei_over_predictions(self, comp, vals, taskindices_comp, cand, task_num, compute_grad=True):
        summed_ei = 0
        print (">>>>> ei_over_predictions >>>> ")
        summed_grad_ei = np.squeeze(np.zeros(cand.shape))
        args = (comp, vals, taskindices_comp, cand, task_num)
        for i in xrange(self.num_samples):
            print (">>> for loop")
            self.set_sample(i)
            if compute_grad:
                print (">>>> comput_grad = True >>>> ")
                (ei,g_ei) = self.compute_ei(*args,compute_grad=True)
                print (">>>> ei_computed >>>> ")
                summed_grad_ei = summed_grad_ei + g_ei
            else:
                ei = self.compute_ei(*args,compute_grad=False)
            summed_ei += ei

        if compute_grad:
            print (">>> ret >>> " , (summed_ei / self.num_samples, summed_grad_ei / self.num_samples))
            return (summed_ei / self.num_samples, summed_grad_ei / self.num_samples)
        else:
            return summed_ei / self.num_samples

    def compute_ei(self, comp, vals, taskindices_comp, cand, task_num, compute_grad=True):
        print (">>>> computing EI")
        cand = np.reshape(cand, (-1, comp.shape[1])) #Wut? -K
        args = (comp, vals, taskindices_comp, cand, task_num*np.ones(cand.shape[0]))
        if not compute_grad:
            func_m, func_v = self.predict(*args, compute_grad=False)
        else:
            func_m, func_v, grad_xp_m, grad_xp_v = self.predict(*args, compute_grad=True)

        ei = np.zeros(func_m.shape[0])

        if compute_grad:
            grad_xp = np.zeros(grad_xp_m.shape)

        ind_comp = (taskindices_comp==task_num)
        assert ind_comp.sum() > 0, 'Need at least one observation for the given task: %d' % task_num

        best = vals[ind_comp].min()
        if not compute_grad:
            ei = self.expected_improvement(best, func_m, func_v)
        else:
            # Assumes only one point
            print ("compute EI, else>>>>>")
            gxpm        = grad_xp_m
            gxpv        = grad_xp_v
            ei, grad_xp = self.expected_improvement(best, func_m, func_v, gxpm, gxpv)

        print (">>>>EI >>>> " , ei)
        if not compute_grad:
            return ei
        else:
            return ei, grad_xp

    def expected_improvement(self, best, func_m, func_v, grad_xp_m=None, grad_xp_v=None):
        # Expected improvement
        func_s = np.sqrt(func_v)
        u      = (best - func_m) / func_s
        ncdf   = sps.norm.cdf(u)
        npdf   = sps.norm.pdf(u)
        ei     = func_s*( u*ncdf + npdf)

        if grad_xp_m is None:
            return ei

        # Gradients of ei w.r.t. mean and variance
        g_ei_m = -ncdf
        g_ei_s2 = 0.5*npdf / func_s

        grad_xp = (grad_xp_m*g_ei_m + grad_xp_v*g_ei_s2)
        ei = np.sum(ei)

        return -ei, -grad_xp.flatten()

    def check_grad_predict(self, comp, vals, taskindices_comp, cand, taskindices_cand):
        (fm,fm,gm,gv) = self.predict(comp, vals, taskindices_comp, cand, taskindices_cand, True)
        gm_est = gm*0
        gv_est = gv*0
        idx = np.zeros(cand.shape[1])
        eps = 1e-5
        for i in xrange(0,cand.shape[1]):
            idx[i] = eps
            (fm1,fv1) = self.predict(comp, vals, taskindices_comp, cand + idx, taskindices_cand)
            (fm2,fv2) = self.predict(comp, vals, taskindices_comp, cand - idx, taskindices_cand)
            gm_est[i] = (fm1 - fm2)/(2*eps)
            gv_est[i] = (fv1 - fv2)/(2*eps)
            idx[i] = 0
            print 'mean: %s' % ([gm[i], gm_est[i]])
            print 'var: %s' % ([gv[i], gv_est[i]])
        print 'computed grads mean', gm
        print 'finite diffs mean', gm_est
        print 'computed grads var', gv
        print 'finite diffs var', gv_est
        print np.linalg.norm(gm - gm_est)
        print np.linalg.norm(gv - gv_est)

    def check_grad_ei(self, comp, vals, taskindices_comp, cand, taskindices_cand):
        (ei,dx1) = self.ei_over_predictions(comp, vals, taskindices_comp, cand, taskindices_cand)
        dx2 = dx1*0
        idx = np.zeros(cand.shape[1])
        for i in xrange(0,cand.shape[1]):
            idx[i] = 1e-5
            (ei1,tmp) = self.ei_over_predictions(comp, vals, taskindices_comp, cand + idx, taskindices_cand)
            (ei2,tmp) = self.ei_over_predictions(comp, vals, taskindices_comp, cand - idx, taskindices_cand)
            dx2[i] = (ei1 - ei2)/(2*1e-5)
            idx[i] = 0
            print [dx1[i], dx2[i]]
        print 'computed grads', dx1
        print 'finite diffs', dx2
        print (dx1/dx2)
        print np.linalg.norm(dx1 - dx2)

    def dump_hypers(self,locker,state_pkl,stats_file):
        sys.stderr.write("Waiting to lock hyperparameter pickle...")
        locker.lock_wait(state_pkl)
        sys.stderr.write("...acquired\n")

        # Write the hyperparameters out to a Pickle.
        fh = tempfile.NamedTemporaryFile(mode='w', delete=False)

        hypers = {
                'mean'         : {},
                'kernel'       : {},
                'input_kernel' : {},
                'task_kernel'  : {}               
                }

        self.mean.to_hypers(hypers['mean'])
        self.kernel.to_hypers(hypers['kernel'])
        self.input_kernel.to_hypers(hypers['input_kernel'])
        self.task_kernel.to_hypers(hypers['task_kernel'])
        
        cPickle.dump(hypers, fh)

        fh.close()

        # Use an atomic move for better NFS happiness.
        cmd = 'mv "%s" "%s"' % (fh.name, state_pkl)
        os.system(cmd) # TODO: Should check system-dependent return status.

        locker.unlock(state_pkl)

        # Write the hyperparameters out to a human readable file as well
        fh = open(stats_file, 'w')
        self.mean.to_file(fh)
        self.kernel.to_file(fh)
        self.input_kernel.to_file(fh)
        self.task_kernel.to_file(fh)
        fh.close()

