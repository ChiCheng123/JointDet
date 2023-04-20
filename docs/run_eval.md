# Prerequisites

**Please ensure environment and data are ready.**

# Train and Test

Train JointDet w/ RDM with 8 GPUs 
```
./tools/dist_train.sh ./exp_cfg/jointdet/person_head_w_rdm.py 8
```

Train JointDet w/o RDM with 8 GPUs 
```
./tools/dist_train.sh ./exp_cfg/jointdet/person_head_wo_rdm.py 8
```

Eval JointDet w/ RDM with 8 GPUs
```
./tools/dist_test.sh ./exp_cfg/jointdet/person_head_w_rdm.py /PATH/TO/CHECKPOINT.pth 8 --eval bbox
```

Eval JointDet w/o RDM with 8 GPUs
```
./tools/dist_test.sh ./exp_cfg/jointdet/person_head_wo_rdm.py /PATH/TO/CHECKPOINT.pth 8 --eval bbox
```