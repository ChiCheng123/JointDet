# Installation Instructions

**a. Create a python virtual environment and activate .**
```shell
python3.8 -m virtualenv jointdet
cd /path/to/jointdet_virtualenv_path
source bin/activate
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.6.2
```

**d. Install mmdetection from source code within JointDet project.**
```shell
cd /PATH/TO/JointDet
pip install -e .
```

**e. Install tensorboard and scikit-image.**
```shell
pip install tensorboard scikit-image
```
