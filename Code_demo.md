

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

