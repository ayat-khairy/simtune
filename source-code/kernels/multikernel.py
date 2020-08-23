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

import sys
import numpy as np
import priors
import weave
import scipy.stats as sps
import scipy.special as spe
import logging

class MultiKernel:
    def __init__(self, input_kernel, task_kernel, **kwargs):
        # Set up the priors
        self.amp_prior          = kwargs.get('amp_prior', priors.lognormal)
        self.amp_prior_params   = kwargs.get('amp_prior_params', [1])
        amp_default             = kwargs.get('amp_default', 1)
        #self.amp_prior         = priors.prior_factory(amp_prior, amp_prior_params)
        self.noise_prior        = kwargs.get('noise_prior', priors.one_sided_horseshoe)
        self.noise_prior_params = kwargs.get('noise_prior_params', [1e-6,0.1])
        noise_default           = kwargs.get('noise_default',1e-6)
        #self.noise_prior       = priors.prior_factory(noise_prior,noise_prior_params)

        self.num_tasks      = task_kernel.num_tasks
        self.amps           = amp_default*np.ones(self.num_tasks)
        self.noises         = noise_default*np.ones(self.num_tasks)
        self.amps_samples   = []
        self.noises_samples = []
        self.input_kernel   = input_kernel
        self.task_kernel    = task_kernel

    def to_hypers(self, hypers):
        hypers['amps']   = self.amps
        hypers['noises'] = self.noises

    def from_hypers(self, hypers):
        self.amps   = hypers['amps']
        self.noises = hypers['noises']

    def set_sample(self, seed):
        self.amps   = self.amps_samples[seed]
        self.noises = self.noises_samples[seed]

    def append_sample(self):
        self.amps_samples.append(self.amps.copy())
        self.noises_samples.append(self.noises.copy())

    def reset_samples(self):
        self.amps_samples   = []
        self.noises_samples = []

    def logprob(self, hypers):
        amps   = hypers[:self.amps.shape[0]]
        noises = hypers[self.amps.shape[0]:self.amps.shape[0]+self.noises.shape[0]]
        means  = hypers[self.amps.shape[0]+self.noises.shape[0]:]

        self.params['amps']   = amps
        self.params['noises'] = noises
        self.params['means']  = means

        lp = self.amp_prior(amps, *self.amp_prior_params)
        lp = lp + self.noise_prior(noises, *self.noise_prior_params)
        lp = lp + self.simple_mean_obj.mean_prior(means, *self.simple_mean_obj.mean_prior_params)
        
        if np.isneginf(lp):
            return -np.inf

        lp = lp + self.logprob_fun(self.params)

        return lp

    def logprob_noiseless(self, hypers):
        amps   = hypers[:self.amps.shape[0]]
        means  = hypers[self.amps.shape[0]:]

        self.params['amps']   = amps
        self.params['means']  = means

        lp = self.amp_prior(amps, *self.amp_prior_params)
        lp = lp + self.simple_mean_obj.mean_prior(means, *self.simple_mean_obj.mean_prior_params)

        if np.isneginf(lp):
            return -np.inf

        lp = lp + self.logprob_fun(self.params)

        return lp

    def sample_hypers_and_means(self,
                                logprob_fun,
                                params,
                                slice_sample_fun,
                                simple_mean_obj,
                                noiseless=False):
        # Instead of using a nested function here we use this method
        # so that this class can be pickled. We also need to make
        # sure to reset the logprob_fun handle afterward.
        self.params           = params
        self.logprob_fun      = logprob_fun
        self.simple_mean_obj  = simple_mean_obj
        if not noiseless:
            hypers                = slice_sample_fun(np.hstack((self.amps,self.noises,simple_mean_obj.means)), self.logprob, compwise=False)
            self.amps             = hypers[:self.amps.shape[0]]
            self.noises           = hypers[self.amps.shape[0]:self.amps.shape[0]+self.noises.shape[0]]
            simple_mean_obj.means = hypers[self.amps.shape[0]+self.noises.shape[0]:]
        else:
            hypers                = slice_sample_fun(np.hstack((self.amps,simple_mean_obj.means)), self.logprob_noiseless, compwise=False)
            self.amps             = hypers[:self.amps.shape[0]]
            simple_mean_obj.means = hypers[self.amps.shape[0]:]
        params['amps']   = self.amps
        params['noises'] = self.noises
        params['means']  = simple_mean_obj.means
        self.logprob_fun = None

    def make_obs_matrices(self, x1, taskindicesx1, x2=None, taskindicesx2=None):
        O1 = np.zeros((x1.shape[0], self.num_tasks))
        O1[np.arange(x1.shape[0]),taskindicesx1.astype(int)] = 1

        if x2 is not None:
            O2 = np.zeros((x2.shape[0], self.num_tasks))
            O2[np.arange(x2.shape[0]),taskindicesx2.astype(int)] = 1
        else:
            O2 = O1

        return O1, O2

    def kernel(self, x1, taskindicesx1, x2=None, taskindicesx2=None, diag_test=False):
        O1, O2        = self.make_obs_matrices(x1, taskindicesx1, x2, taskindicesx2)
        taskindicesx1 = taskindicesx1.astype(int)

        if diag_test:
            K = (1+1e-6)*(self.amps[taskindicesx1]**2)

            return K
        
        self_kernel = (x2 is None)
        if x2 is None:
            x2            = x1
            taskindicesx2 = taskindicesx1
            O2            = O1
        taskindicesx2 = taskindicesx2.astype(int)

        KX = self.input_kernel.kernel(x1,x2)
        KT = self.task_kernel.kernel()

        K = KX*(self.amps[taskindicesx1][:,None]*O1).dot(KT).dot(self.amps[taskindicesx2]*O2.T)
        
        if self_kernel:
            K[np.diag_indices(x1.shape[0])] += self.noises[taskindicesx1] + 1e-6*self.amps[taskindicesx1]**2

        return K

    def kernel_grad(self, x1, taskindicesx1, x2=None, taskindicesx2=None):
        print ("kernel grad")
        O1, O2        = self.make_obs_matrices(x1, taskindicesx1, x2, taskindicesx2)
        taskindicesx1 = taskindicesx1.astype(int)
        taskindicesx2 = taskindicesx2.astype(int)
        print (">>>>>>11 >>>>>")
        dKX = self.input_kernel.kernel_grad(x1,x2)
        print (">>>>>>>>>>>>>>>>22>>>>>>>>>>")
        KT  = self.task_kernel.kernel()
        print (">>>>>>>>dKX>>>>>>>>" , dKX)
        print ("taskindicesx1" ,self.amps[taskindicesx1] )
        print ("taskindicesx2" ,self.amps[taskindicesx2] )
        print (">>>>>>>>>>>>O1>>>>>>>> " , O1)
        print (">>>>>>>>>>>>O2 >>>> " , O2)
        try:
           K = dKX*(self.amps[taskindicesx1][:,None]*O1).dot(KT).dot(self.amps[taskindicesx2]*O2.T)[:,:,None]
           print (">>>>>>>>>>>>>>>>>>> k >>>> " , k)

        except Exception:
           print (">>>>>>>excption before finishing hernel_grad !!!!")
           print (sys.exc_info() [0])

        
        return K

    def print_diagnostic(self):
        logging.info('amps: %s\n' % (self.amps))
        logging.info('noises: %s\n' % (self.noises))

    def to_file(self, fh):
        fh.write('-----------Amplitudes-------------\n')
        fh.write('Mean: %s\n' % np.mean(self.amps_samples,0))
        fh.write('Var: %s\n' % np.var(self.amps_samples,0))
        fh.write('\n')
        fh.write('-----------Noises-------------\n')
        fh.write('Mean: %s\n' % np.mean(self.noises_samples,0))
        fh.write('Var: %s\n' % np.var(self.noises_samples,0))
        fh.write('\n')
