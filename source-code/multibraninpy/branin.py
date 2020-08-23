import numpy as np
import sys
import math
import time

def branin(x, task):
  print 'Input: ', x
  print 'Task: ', task

  if x[0] <0.0 or x[0] > 1:
    return np.NaN

  if x[1] <0 or x[1] > 1:
    return np.NaN

  iy = 0
  if task == 1:
    x[0] = x[0] + 0.1
    x[1] = x[1] + 0.1
    iy = 10

  x[0] = x[0]*15
  x[1] = (x[1]*15)-5

  result = iy + np.square(x[1] - (5.1/(4*np.square(math.pi)))*np.square(x[0]) + (5/math.pi)*x[0] - 6) + 10*(1-(1./(8*math.pi)))*np.cos(x[0]) + 10;

  print 'Result: ', result

  # Take some time to compute...
  time.sleep(0.1)
  return result

# Write a function like this called 'main'
def main(job_id, params):
  print params
  return branin(params['X'], params['Task'][0])
