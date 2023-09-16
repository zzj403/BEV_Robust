
# Env We Use
We use python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 in all conda envs.
Here we give you the code we use to install the envs. Hope be helpful for you.

* TransFusion
```
conda create -n transfusion-adv python=3.8 -y
conda activate transfusion-adv
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge

cd TransFusion
# download  mmcv v1.3.10.zip and unzip
# download  mmdet v2.11.0.zip and unzip

cd mmcv-1.3.10
MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -e . 
cd ..

cd mmdet-2.11.0
FORCE_CUDA=1 pip install -e . 
cd ..


pip install orjson
pip install scipy 
pip install numba==0.48.0 llvmlite==0.31.0 lyft-dataset-sdk==0.0.8

```
!!!! attention here !!!!

you must apply our patch file before 'python setup.py develop' to ensure that you will complie differential voxelization in next step!
```

python setup.py develop
pip install numpy==1.21.5

pip uninstall setuptools -y
pip install setuptools==59.5.0
```


* BEVFusion
```
conda create -n bevfusion-adv python=3.8 -y
conda activate bevfusion-adv
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge

pip install pillow==8.4.0

pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
pip install mmdet==2.20.0

pip install torchpack tqdm
pip install nuscenes-devkit  numba 
pip install mpi4py
pip install orjson

```
!!!! attention here !!!!

you must apply our patch file before 'python setup.py develop' to ensure that you will complie differential voxelization in next step!
```

python setup.py develop
pip install numpy==1.21.5
```


* BEVFormer
```
conda create -n bevformer-adv python=3.8 -y
conda activate bevformer-adv 
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge -y

pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1

pip install scipy
pip install numba==0.48.0 llvmlite==0.31.0 lyft-dataset-sdk==0.0.8
pip install numpy==1.21.5
pip install IPython
pip install orjson

git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py develop

```

* DETR3D
```

conda create -n detr3d-adv python=3.8 -y
conda activate detr3d-adv 
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge -y

pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1


pip install scipy
pip install numba==0.48.0 llvmlite==0.31.0 lyft-dataset-sdk==0.0.8
pip install numpy==1.21.5
pip install IPython

pip install orjson

cd mmdetection3d
python setup.py develop


pip unistall scikit-image
pip istall scikit-image
```




* BEVDet
```
conda create -n bevdet-adv python=3.8 -y
conda activate bevdet-adv
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge -y


pip install mmcv-full==1.3.16 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1



pip install orjson
pip install scipy 

python setup.py develop
pip install numba==0.48.0 llvmlite==0.31.0 lyft-dataset-sdk==0.0.8
pip install numpy==1.21.5

pip uninstall setuptools -y
pip install setuptools==59.5.0


```

