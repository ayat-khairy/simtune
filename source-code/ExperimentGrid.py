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
import tempfile
import cPickle
from sobol import sobol

import numpy        as np
import numpy.random as npr
import glob

from spearmint_pb2 import *
from Locker        import *

CANDIDATE_STATE = 0
SUBMITTED_STATE = 1
RUNNING_STATE   = 2
COMPLETE_STATE  = 3
BROKEN_STATE    = -1

class ExperimentGrid:

    @staticmethod
    def job_running(expt_dir, id):
        expt_grid = ExperimentGrid(expt_dir)
        expt_grid.set_running(id)

    @staticmethod
    def job_complete(expt_dir, id, value, duration, valid=1, add_vals=None):
        expt_grid = ExperimentGrid(expt_dir)
        expt_grid.set_complete(id, value, duration, valid, add_vals)

    @staticmethod
    def job_broken(expt_dir, id):
        expt_grid = ExperimentGrid(expt_dir)
        expt_grid.set_broken(id)

    def __init__(self, expt_dir, variables=None, grid_size=None, grid_seed=1):
        self.expt_dir = expt_dir
        self.jobs_pkl = os.path.join(expt_dir, 'expt-grid.pkl')
        self.locker   = Locker()

        # Only one process at a time is allowed to have access to this.
        sys.stderr.write("Waiting to lock grid...")
        self.locker.lock_wait(self.jobs_pkl)
        sys.stderr.write("...acquired\n")

        # Does this exist already?
        if variables is not None and not os.path.exists(self.jobs_pkl):

            # Set up the grid for the first time.
            self.seed   = grid_seed
            self.vmap   = GridMap(variables, grid_size)
            self.grid   = self._hypercube_grid(self.vmap.card(), grid_size)
            self.status = np.zeros(grid_size, dtype=int) + CANDIDATE_STATE
            self.values = np.zeros(grid_size) + np.nan
            self.valid  = np.zeros(grid_size) + np.nan
            self.durs   = np.zeros(grid_size) + np.nan
            self.sgeids = np.zeros(grid_size, dtype=int)
            self.add_vals = [None for i in xrange(grid_size)]
            assert(len(self.add_vals) == self.grid.shape[0])

            # Save this out.
            self._save_jobs()
        else:

            # Load in from the pickle.
            try:
                self._load_jobs()
            except:
                # The jobs pickle was corrupted
                # This happens if optimization is somehow ungracefully killed
                # while the state is being written to disk.
                # Reinitialize and try to rebuild
                sys.stderr.write('Experiment state file was corrupted. Rebuilding...\n')
                jobfiles = glob.glob(os.path.join(expt_dir, 'jobs/*.pb'))
                maxjobindex = np.max(np.hstack((grid_size,
                    [np.int(jobfile.split('/')[-1][:-3]) for jobfile in jobfiles])))
                grid_size   = maxjobindex + 1
                self.seed   = grid_seed
                self.vmap   = GridMap(variables, grid_size)
                self.grid   = self._hypercube_grid(self.vmap.card(), grid_size)
                self.status = np.zeros(grid_size, dtype=int) + CANDIDATE_STATE
                self.values = np.zeros(grid_size) + np.nan
                self.valid  = np.zeros(grid_size) + np.nan
                self.durs   = np.zeros(grid_size) + np.nan
                self.sgeids = np.zeros(grid_size, dtype=int)
                self.add_vals = [None for i in xrange(grid_size)]
                
                for jfile in jobfiles:
                    sys.stderr.write('.')
                    job = self._load_job(jfile)
                    
                    param_vector = self.vmap.to_unit(job.param)
                    if job.id > grid_size:
                        job.id = self.add_to_grid(param_vector)
                    self.grid[job.id] = param_vector
                    assert(np.all(param_vector == self.grid[job.id]))

                    if job.status == 'complete':
                        self.set_complete(job.id, job.value, job.duration, job.valid,
                                          job.add_vals)

        # Save this out.
        self._save_jobs()

    def __del__(self):
        self._save_jobs()
        if self.locker.unlock(self.jobs_pkl):
            sys.stderr.write("Released lock on job grid.\n")
        else:
            raise Exception("Could not release lock on job grid.\n")

    def get_grid(self):
        return self.grid, self.values, self.durs, self.valid, self.add_vals

    def get_candidates(self):
        return np.nonzero(self.status == CANDIDATE_STATE)[0]

    def get_pending(self):
        return np.nonzero((self.status == SUBMITTED_STATE) | (self.status == RUNNING_STATE))[0]

    def get_complete(self):
        return np.nonzero(self.status == COMPLETE_STATE)[0]

    def get_broken(self):
        return np.nonzero(self.status == BROKEN_STATE)[0]

    def get_params(self, index):
        return self.vmap.get_params(self.grid[index,:])

    def get_best(self):
        finite = self.values[np.logical_and(np.isfinite(self.values), self.valid==1)]
        if len(finite) > 0:
            cur_min = np.min(finite)
            index   = np.nonzero(self.values==cur_min)[0][0]
            return cur_min, index
        else:
            return np.nan, -1

    def get_sgeid(self, id):
        return self.sgeids[id]

    def add_to_grid(self, candidate):
        # Set up the grid
        self.grid   = np.vstack((self.grid, candidate))
        self.status = np.append(self.status, np.zeros(1, dtype=int) + 
                                int(CANDIDATE_STATE))
        self.values = np.append(self.values, np.zeros(1)+np.nan)
        self.durs   = np.append(self.durs, np.zeros(1)+np.nan)
        self.valid  = np.append(self.valid, np.zeros(1)+np.nan)
        self.sgeids = np.append(self.sgeids, np.zeros(1,dtype=int))
        self.add_vals.append(None)

        # Save this out.
        self._save_jobs()
        return self.grid.shape[0]-1

    def set_candidate(self, id):
        self.status[id] = CANDIDATE_STATE
        self._save_jobs()

    def set_submitted(self, id, sgeid):
        self.status[id] = SUBMITTED_STATE
        self.sgeids[id] = sgeid
        self._save_jobs()

    def set_running(self, id):
        self.status[id] = RUNNING_STATE
        self._save_jobs()

    def set_complete(self, id, value, duration, valid=1, add_vals=None):
        self.status[id]   = COMPLETE_STATE
        self.values[id]   = value
        self.valid[id]    = valid
        self.durs[id]     = duration
        self.add_vals[id] = [i for i in add_vals]
        
        self._save_jobs()

    def set_broken(self, id):
        self.status[id] = BROKEN_STATE
        self._save_jobs()

    def _load_job(self, filename):
        fh = open(filename, 'rb')
        job = Job()
        job.ParseFromString(fh.read())

        return job

    def _load_jobs(self):
        fh   = open(self.jobs_pkl, 'r')
        jobs = cPickle.load(fh)
        fh.close()

        self.vmap     = jobs['vmap']
        self.grid     = jobs['grid']
        self.status   = jobs['status']
        self.values   = jobs['values']
        self.durs     = jobs['durs']
        self.valid    = jobs['valid']
        self.add_vals = jobs['add_vals']
        self.sgeids   = jobs['sgeids']

    def _save_jobs(self):        

        # Write everything to a temporary file first.
        fh = tempfile.NamedTemporaryFile(mode='w', delete=False)
        cPickle.dump({ 'vmap'      : self.vmap,
                       'grid'      : self.grid,
                       'status'    : self.status,
                       'values'    : self.values,
                       'durs'      : self.durs,
                       'valid'     : self.valid,
                       'add_vals'  : self.add_vals,
                       'sgeids'    : self.sgeids }, fh, protocol=cPickle.HIGHEST_PROTOCOL)
        fh.close()

        # Use an atomic move for better NFS happiness.
        cmd = 'mv "%s" "%s"' % (fh.name, self.jobs_pkl)
        os.system(cmd) # TODO: Should check system-dependent return status.

    def _hypercube_grid(self, dims, size):
        # Generate from a sobol sequence
        return sobol(size+self.seed, dims)[self.seed:]

class GridMap:
    
    def __init__(self, variables, grid_size):
        self.variables      = []
        self.variable_types = [] # Stores the variable types for each grid index.
        self.cardinality    = 0

        # Count the total number of dimensions and roll into new format.
        for variable in variables:
            self.cardinality += variable.size

            if variable.type == Experiment.ParameterSpec.INT:
                self.variables.append({ 'name' : variable.name,
                                        'size' : variable.size,
                                        'type' : 'int',
                                        'min'  : int(variable.min),
                                        'max'  : int(variable.max)})

            elif variable.type == Experiment.ParameterSpec.FLOAT:
                self.variables.append({ 'name' : variable.name,
                                        'size' : variable.size,
                                        'type' : 'float',
                                        'min'  : float(variable.min),
                                        'max'  : float(variable.max)})

            elif variable.type == Experiment.ParameterSpec.ENUM:
                self.variables.append({ 'name'    : variable.name,
                                        'size'    : variable.size,
                                        'type'    : 'enum',
                                        'options' : list(variable.options)})
            else:
                raise Exception("Unknown parameter type.")
        sys.stderr.write("Optimizing over %d dimensions\n" % (self.cardinality))

        for variable in self.variables:
            for dd in xrange(variable['size']):
                self.variable_types.append(variable['type'])
    
    # Convert a variable to the unit hypercube
    # Takes a single variable encoded as a list, assuming the ordering is 
    # the same as specified in the configuration file
    def to_unit(self, v):
        unit = np.zeros(self.cardinality)
        index  = 0

        k = 0
        for variable in self.variables:
            #param.name = variable['name']
            params = v[k]
            k = k+1
            if variable['type'] == 'int':
                for dd in xrange(variable['size']):
                    unit[index] = self._index_unmap(float(params.int_val[dd]) - variable['min'], (variable['max']-variable['min']))
                    index += 1

            elif variable['type'] == 'float':
                for dd in xrange(variable['size']):
                    unit[index] = (float(params.dbl_val[dd]) - variable['min'])/(variable['max']-variable['min'])
                    index += 1

            elif variable['type'] == 'enum':
                for dd in xrange(variable['size']):
                    unit[index] = variable['options'].index(params.str_val[dd])
                    index += 1

            else:
                raise Exception("Unknown parameter type.")
            
        return unit

    def get_params(self, u):
        if u.shape[0] != self.cardinality:
            raise Exception("Hypercube dimensionality is incorrect.")

        params = []
        index  = 0
        for variable in self.variables:
            param = Parameter()
            
            param.name = variable['name']

            if variable['type'] == 'int':
                for dd in xrange(variable['size']):
                    param.int_val.append(variable['min'] + self._index_map(u[index], variable['max']-variable['min']+1))
                    index += 1

            elif variable['type'] == 'float':
                for dd in xrange(variable['size']):
                    param.dbl_val.append(variable['min'] + u[index]*(variable['max']-variable['min']))
                    index += 1

            elif variable['type'] == 'enum':
                for dd in xrange(variable['size']):
                    ii = self._index_map(u[index], len(variable['options']))
                    index += 1
                    param.str_val.append(variable['options'][ii])

            else:
                raise Exception("Unknown parameter type.")
            
            params.append(param)

        return params

    # Return indices into enum type variables.
    def get_enum_indices(self):
        index = np.zeros(self.cardinality)
        ind = 0
        nenums = 0
        for variable in self.variables:
            if variable['type'] == 'enum':
                nenums += 1
                index[ind:ind+variable['size']] = nenums
            ind += variable['size']
        return index

    def get_variable_property(self, name, prop):
        for variable in self.variables:
            if variable['name'] == name:
                return variable[prop]

        raise Exception('Variable %s with property %s not found' % (name,prop))
            
    def card(self):
        return self.cardinality

    def get_param_from_index(self, u, index):
        variable = self.variable_types[index]
        if variable['type'] == 'int':
            param = (variable['min'] + self._index_map(u, variable['max']-variable['min']+1))

        return param

    def _index_map(self, u, items):
        return int(np.floor((1-np.finfo(float).eps) * u * float(items)))

    def _index_unmap(self, u, items):
        return float(float(u) / float(items))
