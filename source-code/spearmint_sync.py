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

import optparse
import tempfile
import datetime
import subprocess
import time
import imp
import os
import re
import Locker
from google.protobuf import text_format
from spearmint_pb2   import *
from ExperimentGrid  import *

# System dependent modules
MCR_LOCATION = "/home/matlab/v715" # hack

#
# There are two things going on here.  There are "experiments", which are
# large-scale things that live in a directory and in this case correspond
# to the task of minimizing a complicated function.  These experiments
# contain "jobs" which are individual function evaluations.  The set of
# all possible jobs, regardless of whether they have been run or not, is
# the "grid".  This grid is managed by an instance of the class
# ExperimentGrid.
#
# The spearmint.py script can run in two modes, which reflect experiments
# vs jobs.  When run with the --wrapper argument, it will try to run a
# single job.  This is not meant to be run by hand, but is intended to be
# run by a job queueing system.  Without this argument, it runs in its main
# controller mode, which determines the jobs that should be executed and
# submits them to the queueing system.
#

def main():
    parser = optparse.OptionParser(usage="usage: %prog [options] directory")

    parser.add_option("--max-concurrent", dest="max_concurrent",
                      help="Maximum number of concurrent jobs.",
                      type="int", default=1)
    parser.add_option("--max-finished-jobs", dest="max_finished_jobs",
                      type="int", default=10000)
    parser.add_option("--method", dest="chooser_module",
                      help="Method for choosing experiments.",
                      type="string", default="MultiTaskEIOptChooser")
    parser.add_option("--method-args", dest="chooser_args",
                      help="Arguments to pass to chooser module.",
                      type="string", default="")
    parser.add_option("--grid-size", dest="grid_size",
                      help="Number of experiments in initial grid.",
                      type="int", default=20000)
    parser.add_option("--grid-seed", dest="grid_seed",
                      help="The seed used to initialize initial grid.",
                      type="int", default=1)
    parser.add_option("--config", dest="config_file",
                      help="Configuration file name.",
                      type="string", default="config.pb")
    parser.add_option("--wrapper", dest="wrapper",
                      help="Run in job-wrapper mode.",
                      action="store_true")
    parser.add_option("--polling-time", dest="polling_time",
                      help="The time in-between successive polls for results.",
                      type="float", default=3)

    (options, args) = parser.parse_args()

    if options.wrapper:
        # Possibly run in job wrapper mode.
        main_wrapper(options, args)

    else:
        # Otherwise run in controller mode.
        main_controller(options, args)
    
##############################################################################
##############################################################################
def main_wrapper(options, args):
    sys.stderr.write("Running in wrapper mode for '%s'\n" % (args[0]))

    # This happens when the job is actually executing.  Now we are
    # going to do a little bookkeeping and then spin off the actual
    # job that does whatever it is we're trying to achieve.

    # Load in the protocol buffer spec for this job and experiment.
    job_file = args[0]
    job      = load_job(job_file)

    ExperimentGrid.job_running(job.expt_dir, job.id)
    
    # Update metadata.
    job.start_t = int(time.time())
    job.status  = 'running'
    save_job(job_file, job)

    ##########################################################################
    success    = False
    start_time = time.time()

    try:
        if job.language == MATLAB:
            # Run it as a Matlab function.
            function_call = "matlab_wrapper('%s'),quit;" % (job_file)
            matlab_cmd    = ('matlab -nosplash -nodesktop -r "%s"' % 
                             (function_call))
            sys.stderr.write(matlab_cmd + "\n")
            subprocess.check_call(matlab_cmd, shell=True)

        elif job.language == PYTHON:
            # Run a Python function
            sys.stderr.write("Running python job.\n")

            # Add directory to the system path.
            sys.path.append(os.path.realpath(job.expt_dir))

            # Change into the directory.
            os.chdir(job.expt_dir)
            sys.stderr.write("Changed into dir %s\n" % (os.getcwd()))

            # Convert the PB object into useful parameters.
            params = {}
            for param in job.param:
                dbl_vals = param.dbl_val._values
                int_vals = param.int_val._values
                str_vals = param.str_val._values

                if len(dbl_vals) > 0:
                    params[param.name] = np.array(dbl_vals)
                elif len(int_vals) > 0:
                    params[param.name] = np.array(int_vals, dtype=int)
                elif len(str_vals) > 0:
                    params[param.name] = str_vals
                else:
                    raise Exception("Unknown parameter type.")

            # Load up this module and run
            module  = __import__(job.name)
            result = module.main(job.id, params)

            # Change back out.
            os.chdir('..')

            # Store the result.

            # DEPRECATED.  Will be removed shortly
            if isinstance(result, tuple):
                result, isvalid = result
                job.valid = np.int(isvalid)

            # If the function returns a list of values, then we
            # keep all values beyond the first in an 'additional
            # values' list - which the chooser can deal with.
            if isinstance(result, list):
                addvals = [result[i] for i in xrange(1,len(result))]
                result = result[0]                
                for i in addvals:
                    job.add_vals.append(i)

            job.value = result            
            save_job(job_file, job)

            sys.stderr.write("Got result %f\n" % (result))

        elif job.language == SHELL:
            # Change into the directory.
            os.chdir(job.expt_dir)

            cmd = './%s %s' % (job.name, job_file)
            sys.stderr.write("Executing command '%s'\n" % (cmd))

            subprocess.check_call(cmd, shell=True)

        elif job.language == MCR:

            # Change into the directory.
            os.chdir(job.expt_dir)

            if os.environ.has_key('MATLAB'):
                mcr_loc = os.environ['MATLAB']
            else:
                mcr_loc = MCR_LOCATION

            cmd = './run_%s.sh %s %s' % (job.name, mcr_loc, job_file)
            sys.stderr.write("Executing command '%s'\n" % (cmd))
            subprocess.check_call(cmd, shell=True)

        else:
            raise Exception("That function type has not been implemented.")

        success = True
    except:
        import traceback
        traceback.print_exc()
        sys.stderr.write("Problem executing the function\n")
        print sys.exc_info()
        
    end_time = time.time()
    duration = end_time - start_time
    ##########################################################################

    job = load_job(job_file)
    sys.stderr.write("Job file reloaded.\n")

    if not job.HasField("value"):
        sys.stderr.write("Could not find value in output file.\n")
        success = False

    if success:
        sys.stderr.write("Completed successfully in %0.2f seconds. [%f]\n" 
                         % (duration, job.value))

        # Update the status for this job.
        ExperimentGrid.job_complete(job.expt_dir, job.id,
                                    job.value, duration, job.valid,
                                    job.add_vals)
    
        # Update metadata.
        job.end_t    = int(time.time())
        job.status   = 'complete'
        job.duration = duration

    else:
        sys.stderr.write("Job failed in %0.2f seconds.\n" % (duration))

        # Update the status for this job.
        ExperimentGrid.job_broken(job.expt_dir, job.id)
    
        # Update metadata.
        job.end_t    = int(time.time())
        job.status   = 'broken'
        job.duration = duration

    save_job(job_file, job)

##############################################################################
##############################################################################
def main_controller(options, args):

    expt_dir  = os.path.realpath(args[0])
    work_dir  = os.path.realpath('.')
    expt_name = os.path.basename(expt_dir)

    if not os.path.exists(expt_dir):
        sys.stderr.write("Cannot find experiment directory '%s'. "
                         "Aborting.\n" % (expt_dir))
        sys.exit(-1)

    # Load up the chooser module.
    module  = __import__(options.chooser_module)
    chooser = module.init(expt_dir, options.chooser_args)
 
    # Loop until we run out of jobs.
    while True:
        attempt_dispatch(expt_name, expt_dir, work_dir, chooser, options)
        # This is polling frequency. A higher frequency means that the algorithm
        # picks up results more quickly after they finish, but also significantly 
        # increases overhead.
        time.sleep(options.polling_time)
 
def attempt_dispatch(expt_name, expt_dir, work_dir, chooser, options):
    sys.stderr.write("\n")
    
    expt_file = os.path.join(expt_dir, options.config_file)
    expt      = load_expt(expt_file)

    # Build the experiment grid.
    expt_grid = ExperimentGrid(expt_dir,
                               expt.variable,
                               options.grid_size,
                               options.grid_seed)

    # Print out the current best function value.
    best_val, best_job = expt_grid.get_best()
    if best_job >= 0:
        sys.stderr.write("Current best: %f (job %d)\n" % (best_val, best_job))
    else:
        sys.stderr.write("Current best: No results returned yet.\n")

    # Gets you everything - NaN for unknown values & durations.
    grid, values, durations, valid, add_vals = expt_grid.get_grid()
    
    # Returns lists of indices.
    candidates = expt_grid.get_candidates()
    pending    = expt_grid.get_pending()
    complete   = expt_grid.get_complete()
    sys.stderr.write("%d candidates   %d pending   %d complete\n" %
                     (candidates.shape[0], pending.shape[0], complete.shape[0]))

    # Verify that pending jobs are actually running.
    for job_id in pending:
        sgeid = expt_grid.get_sgeid(job_id)
        reset_job = False
        
        try:
            # Send an alive signal to proc (note this could kill it in windows)
            os.kill(sgeid, 0)
        except OSError:
            # Job is no longer running but still in the candidate list. Assume it crashed out.
            expt_grid.set_candidate(job_id)

    # Track the time series of optimization.
    trace_fh = open(os.path.join(expt_dir, 'trace.csv'), 'a')
    trace_fh.write("%d,%f,%d,%d,%d,%d\n"
                   % (time.time(), best_val, best_job,
                      candidates.shape[0], pending.shape[0], complete.shape[0]))
    trace_fh.close()

    # Print out the best job results
    best_job_fh = open(os.path.join(expt_dir, 'best_job_and_result.txt'), 'w')
    best_job_fh.write("Best result: %f\nJob-id: %d\nParameters: \n" % 
                      (best_val, best_job))
    for best_params in expt_grid.get_params(best_job):
        best_job_fh.write(str(best_params) + '\n')
    best_job_fh.close()

    if complete.shape[0] >= options.max_finished_jobs:
        sys.stderr.write("Maximum number of finished jobs (%d) reached."
                         "Exiting\n" % options.max_finished_jobs)
        sys.exit(0)

    if candidates.shape[0] == 0:
        sys.stderr.write("There are no candidates left.  Exiting.\n")
        sys.exit(0)

    if pending.shape[0] >= options.max_concurrent:
        sys.stderr.write("Maximum number of jobs (%d) pending.\n"
                         % (options.max_concurrent))
        return
    #n_complete = complete.shape[0] 
    #if n_complete >= 10:
    #     sys.exit(0)
    # Ask the chooser to actually pick one.
    job_id = chooser.next(grid, values, durations, candidates, pending,
                          complete, add_vals, vmap=expt_grid.vmap)

    # If the job_id is a tuple, then the chooser picked a new job.
    # We have to add this to our grid
    if isinstance(job_id, tuple):
        (job_id, candidate) = job_id
        job_id = expt_grid.add_to_grid(candidate)

    sys.stderr.write("Selected job %d from the grid.\n" % (job_id))

    # Convert this back into an interpretable job and add metadata.
    job = Job()
    job.id        = job_id
    job.expt_dir  = expt_dir
    job.name      = expt.name
    job.language  = expt.language
    job.status    = 'submitted'
    job.submit_t  = int(time.time())
    job.param.extend(expt_grid.get_params(job_id))

    # Make sure we have a job subdirectory.
    job_subdir = os.path.join(expt_dir, 'jobs')
    if not os.path.exists(job_subdir):
        os.mkdir(job_subdir)

    # Name this job file.
    job_file = os.path.join(job_subdir,
                            '%08d.pb' % (job_id))

    # Store the job file.
    save_job(job_file, job)

    # Make sure there is a directory for output.
    output_subdir = os.path.join(expt_dir, 'output')
    if not os.path.exists(output_subdir):
        os.mkdir(output_subdir)
    output_file = os.path.join(output_subdir,
                               '%08d.out' % (job_id))

    process = job_submit("%s-%08d" % (expt_name, job_id),
                         output_file,
                         job_file, work_dir)
    process.poll()
    if process.returncode is not None and process.returncode < 0:
        sys.stderr.write("Failed to submit job or job crashed "
                         "with return code %d !\n" % process.returncode)
        sys.stderr.write("Deleting job file.\n")
        os.unlink(job_file)
        return
    else:
        sys.stderr.write("Submitted job as process: %d\n" % process.pid)

    # Now, update the experiment status to submitted.
    expt_grid.set_submitted(job_id, process.pid)

    return

def load_expt(filename):
    fh = open(filename, 'rb')
    expt = Experiment()
    text_format.Merge(fh.read(), expt)
    fh.close()
    return expt

def load_job(filename):
    fh = open(filename, 'rb')
    job = Job()
    #text_format.Merge(fh.read(), job)
    job.ParseFromString(fh.read())
    fh.close()
    return job

def save_expt(filename, expt):
    fh = tempfile.NamedTemporaryFile(mode='w', delete=False)
    fh.write(text_format.MessageToString(expt))
    fh.close()
    cmd = 'mv "%s" "%s"' % (fh.name, filename)
    subprocess.check_call(cmd, shell=True)

def save_job(filename, job):
    fh = tempfile.NamedTemporaryFile(mode='w', delete=False)
    #fh.write(text_format.MessageToString(job))
    fh.write(job.SerializeToString())
    fh.close()
    cmd = 'mv "%s" "%s"' % (fh.name, filename)
    subprocess.check_call(cmd, shell=True)

def job_submit(name, output_file, job_file, working_dir):

    cmd = ('''python spearmint_sync.py --wrapper "%s" > %s''' % 
           (job_file, output_file))
    output_file = open(output_file, 'w')

    # Submit the job.
    #locker = Locker()
    #locker.unlock(working_dir + '/expt-grid.pkl')
    process = subprocess.Popen(cmd,
                               stdout=output_file,
                               stderr=output_file, shell=True)

    return process

if __name__ == '__main__':
    main()
