

## CrowdHuman

**Download**

Download CrowdHuman data and annotation [HERE](https://www.crowdhuman.org/download.html). 
Organize as following folder structure
```
JointDet
├── datasets/
│   ├── CrowdHuman/
│   │   ├── train/
│   │   ├── val/
│   │   ├── annotation_train.odgt
│   │   ├── annotation_val.odgt
│   │   ├── convert_crowdhuman_to_coco.py
```

**Prepare coco-format data**

```shell
cd /PATH/TO/JointDet/datasets/CrowdHuman
python convert_crowdhuman_to_coco.py
```

After running the above code, `{train,val}.json` will be generated as following.
```
JointDet
├── datasets/
│   ├── CrowdHuman/
│   │   ├── train/
│   │   ├── train.json
│   │   ├── val/
│   │   ├── val.json
│   │   ├── annotation_train.odgt
│   │   ├── annotation_val.odgt
│   │   ├── convert_crowdhuman_to_coco.py
```