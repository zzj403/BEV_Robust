# BEV_Robust
Offical code for CVPR 2023 Paper Understanding the Robustness of 3D Object Detection With Bird's-Eye-View Representations in Autonomous Driving ([cvpr_url](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_Understanding_the_Robustness_of_3D_Object_Detection_With_Birds-Eye-View_Representations_CVPR_2023_paper.html)) ([arXiv_url](https://arxiv.org/abs/2303.17297))


## Repo Explaination
For you and we can clearly see what have been changed by us, we link original repo as submodules and give you repo changes file: XXXX_changes.patch

You can use these patch files like this (transfusion for example):
```bash
cd TransFusion
patch -p1 < ../transfusion_changes.patch
```

By the way, we save .patch files by
```bash
git add .  # add changes for a short time
git diff --cached --name-only  # show added changes
git diff --cached > ../transfusion_changes.patch # save changes to outside file
git reset --mixed # recover changes 
```

Common attack codes are in [`./apis_common`](./apis_common). Common tool codes are in [`./extend_common`](./extend_common). These codes are used in model repos by soft link.

Special attack codes are in model repo's `mmdeted/apis`. Special tool codes are in model repo's `extend`.


## Patch Attack Mask and Info 
The mask and info of adversarial patches are obtained all in TransFusion repo.
You should generate mask and info in TransFusion, and then modify the mask/info path in all patch attack codes.



## Common Attack Code and Unique Attack Code
Most model+attack pairs share common attack code in [`./apis_common`](./apis_common), but there are some exception: BEVFormer, FCOS3D.


### BEVFormer
* BEVFormer patch_class
* BEVFormer patch_temporal

BEVFormer uses history info like 'prev_bev' and others. We keep the update of 'prev_bev' in forward_test(), not include it in forward_train()(in which we get attack loss). So we should run an extra forward_test(), after every time we change the frame in temporal-related attacks (patch_class, patch_temporal). 

Though, BEVFormer is not special in other attack, like FGSM and PGD. Because we eval (forward_test) immediately after each frame training in these attacks.


### FCOS3D
* FCOS3D patch_temporal

FCOS3D uses nus-mono dataset which has different scenes order with nus dataset. So the scene change detection code is different with other models. We use a stand-alone code to do patch_temporal for FCOS3D

## Make Voxelization Differential
We replace the voxelization code in BEVFusion and TransFusion with a differetial version. This voxelization is modified from a old version of voxelization. We 

## Env We Use
Refer to [Install.md](./Install.md) 

## Prepare dataset and checkpoint
Dataset and checkpoints prepare please refer to each model repo. 


## Patch Mask and Info Code
For example, TransFusion
```bash
cd TransFusion
conda activate transfusion-adv
```
Save instance patch mask, for instance patch attack
```bash
python tools/test_save_instance_patch_mask_launcher.py configs/transfusion_nusc_voxel_LC_adv.py ckpt/transfusion_LC.pth patch_instance_mask_dir
```
Save patch info, for other patch attacks
```bash
python tools/test_save_patch_info_launcher.py configs/transfusion_nusc_voxel_LC_instancetoken.py ckpt/transfusion_LC.pth patch_info_dir
```


## Attack Code Demo
TransFusion
```bash
cd TransFusion
conda activate transfusion-adv
```

fgsm eps=8
```bash
CUDA_VISIBLE_DEVICES=0 python tools/test_fgsm_img_launcher.py configs/transfusion_nusc_voxel_LC_adv.py ckpt/transfusion_LC.pth adv_fgsm_img_results 8
```

pgd eps=8 step=10
```bash
CUDA_VISIBLE_DEVICES=0 python tools/test_pgd_img_launcher.py configs/transfusion_nusc_voxel_LC_adv.py ckpt/transfusion_LC.pth adv_pgd_img_results 8 10
```

pgd img_eps=8 point_eps=0.5 step=10
```bash
CUDA_VISIBLE_DEVICES=0 python tools/test_pgd_imgpoint_launcher.py configs/transfusion_nusc_voxel_LC_adv.py ckpt/transfusion_LC.pth adv_pgd_imgpoint_results 8 0.5 10
```

instance patch patch_size=040 (for 0.4, 001 002 005 010 020 040 are supported) attack_step=10
```bash
CUDA_VISIBLE_DEVICES=0 python tools/test_patch_instance_launcher.py configs/transfusion_nusc_voxel_LC_adv.py ckpt/transfusion_LC.pth adv_instance_patch_results 040 10
```

class patch patch_size=0.05 lr=0.1.
(please carefully SAVE the attack log, eval log is in the attack log.)
```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u tools/test_patch_class_launcher.py configs/transfusion_nusc_voxel_LC_adv.py ckpt/transfusion_LC.pth adv_class_patch 0.05 0.1  >> adv_class_patch_size0.05_lr0.1.log 2>&1 &
```

temporal patch patch_size=0.05 lr=0.1 step=3
```bash
CUDA_VISIBLE_DEVICES=0 python tools/test_patch_temporal_launcher.py configs/transfusion_nusc_voxel_LC_adv.py ckpt/transfusion_LC.pth adv_temporal_patch_results 0.05 0.1 3
```

overlap patch patch_size=0.05 lr=0.1 step=20
```bash
CUDA_VISIBLE_DEVICES=0 python tools/test_patch_overlap_launcher.py configs/transfusion_nusc_voxel_LC_adv.py ckpt/transfusion_LC.pth adv_overlap_patch_results 0.05 0.1 20
```

## Eval Code Demo
For all attacks except class patch and overlap patch, we use the following code to eval:

for example: transfusion fgsm
```bash
python tools/test_scatterd_eval.py configs/transfusion_nusc_voxel_LC_adv.py fgsm_img_resultseps8 -eval bbox
```

For class patch, eval is in the attack code, please carefully save the attack log and search for eval results.

For overlap patch, please refer to BEVFormer repo to find the evaluation code for overlap object detection.


## Other Things
To save the work during paper preparing, we use BEVDepth implementation in BEVDet repo. Because the pytorch_lightning used in original BEVDepth repo is too hard to modified with adversarial features. 
Thanks to [BEVDet](https://github.com/HuangJunJie2017/BEVDet) which give us another implementation for BEVDepth. And we still thanks the bravo work of [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth).

## Acknowledgement
Many thanks to these excellent projects:
- [TransFusion](https://github.com/XuyangBai/TransFusion)
- [BEVFusion](https://github.com/mit-han-lab/bevfusion)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [DETR3D](https://github.com/WangYueFt/detr3d)
- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)


## Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@inproceedings{zhu2023understanding,
  title={Understanding the Robustness of 3D Object Detection With Bird's-Eye-View Representations in Autonomous Driving},
  author={Zhu, Zijian and Zhang, Yichi and Chen, Hai and Dong, Yinpeng and Zhao, Shu and Ding, Wenbo and Zhong, Jiachen and Zheng, Shibao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21600--21610},
  year={2023}
}
```