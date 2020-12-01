## Global job properties
universe     = vanilla
notification = error
notify_user  = awlee22@amherst.edu
initialdir   = /mnt/scratch/awlee22/conservation-bagging-svcs
getenv = True
executable   = run

## Job properties
output = /home/awlee22/cluster-scratch/conservation-bagging-svcs/condor_results/pmlb/out
error  = /home/awlee22/cluster-scratch/conservation-bagging-svcs/condor_results/pmlb/err
log    = /home/awlee22/cluster-scratch/conservation-bagging-svcs/condor_results/pmlb/log
arguments = pmlb_driver.py
queue