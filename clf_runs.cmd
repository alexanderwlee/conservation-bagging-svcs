## Global job properties
universe     = vanilla
notification = error
notify_user  = awlee22@amherst.edu
initialdir   = /mnt/scratch/awlee22/conservation-bagging-svcs
getenv = True
executable   = run

## Job properties
output = /home/awlee22/cluster-scratch/conservation-bagging-svcs/condor_results/clf/out
error  = /home/awlee22/cluster-scratch/conservation-bagging-svcs/condor_results/clf/err
log    = /home/awlee22/cluster-scratch/conservation-bagging-svcs/condor_results/clf/log
arguments = clf_driver.py
queue