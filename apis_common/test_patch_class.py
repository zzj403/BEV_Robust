import mmcv
import torch
import numpy as np
import PIL.Image as Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
import cv2
import time
import os
import pickle
from extend.custom_func import *
from extend_common.img_check import img_diff_print
from extend_common.time_counter import time_counter
from extend_common.patch_apply import apply_patches_by_info
from extend_common.path_string_split import split_path_string_to_multiname



def single_gpu_test(model, data_loader,
                    patch_save_prefix=None, 
                    area_rate_str=None,
                    optim_lr=None
                    ):
    
    model.eval()
    dataset = data_loader.dataset
    device = model.src_device_obj
    
    patch_save_dir = patch_save_prefix +'_area'+area_rate_str+'_lr'+optim_lr
    os.makedirs(patch_save_dir, exist_ok=True)

    optim_lr = float(optim_lr)


    # 为每一个类别定义一个patch
    # define one patch for evey class
    class_names_list = [
        'car', 'truck', 'construction_vehicle', 
        'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 
        'traffic_cone'
    ]
    patch_w = 100
    patch_h = 100
    class_patches_tensor = torch.rand(len(class_names_list),3, patch_h, patch_w).to(device)
    class_patches_tensor.requires_grad_()
    optimizer = torch.optim.Adam([class_patches_tensor], lr=optim_lr)


    time_test_flag = False

    epoch_max = 3
    patch_info_list_database = {}

    # epoch 0 2 4 6 ... for train
    # epoch 1 3 5 7 ... for eval
    for epoch_d in range(epoch_max*2+1):
        epoch = int(epoch_d/2)
        patch_is_training = (epoch_d % 2 == 0)

        if patch_is_training:
            print('=============================')
            print('======= epoch',epoch,' train start =========')
            print('=============================')
        else:
            print('=============================')
            print('======= epoch',epoch,'eval start =========')
            print('=============================')
            results = []

        prog_bar = mmcv.ProgressBar(len(dataset))
        last_time = time.time()
        for data_i, data_out in enumerate(data_loader):


            #### 1. data processing(customed)
            data_out = custom_data_preprocess(data_out)
            img_metas, img_path_list, img_org_np, img_processed, gt_labels_3d = custom_data_work(data_out)
            img_tensor_ncam = custom_img_read_from_img_org(img_org_np, device)
            last_time = time_counter(last_time, 'data load', time_test_flag)
            cam_num = len(img_path_list)

            #### 2. read patch info from file/database
            if not str(data_i) in patch_info_list_database:
                patch_info_list = []
                for cams_i in range(cam_num):
                    img_path = img_path_list[cams_i]
                    file_name_valid_list = split_path_string_to_multiname(img_path)[-3:]
                    file_name_valid_list.insert(0, '/data/zijian/mycode/BEV_Robust/TransFusion/patch_info_2d3d3dt_square_dir/all')
                    info_path = os.path.join(*file_name_valid_list)
                    info_path = info_path.replace('.jpg', '.pkl')
                    info_i = pickle.load(open(info_path, 'rb'))
                    patch_info_list.append(info_i)
                patch_info_list_database[str(data_i)] = patch_info_list
            else:
                patch_info_list = patch_info_list_database[str(data_i)]
            last_time = time_counter(last_time, 'read pkl', time_test_flag)


            #### 3. apply patch
            patched_img_tensor_ncam = img_tensor_ncam.clone()
            # to avoid no_gt
            has_gt_flag = gt_labels_3d.shape[0] != 0
            if has_gt_flag:
                if patch_is_training:
                    # for patch training
                    for cams_i in range(cam_num):
                        patch_info_in_cami = patch_info_list[cams_i]
                        patched_img_tensor_ncam[cams_i] = apply_patches_by_info(
                            info=patch_info_in_cami, 
                            image=patched_img_tensor_ncam[cams_i], 
                            patch_book=class_patches_tensor,
                            area_str=area_rate_str,
                            )
                else:
                    with torch.no_grad():
                        # for patch eval
                        for cams_i in range(cam_num):
                            patch_info_in_cami = patch_info_list[cams_i]
                            patched_img_tensor_ncam[cams_i] = apply_patches_by_info(
                                info=patch_info_in_cami, 
                                image=patched_img_tensor_ncam[cams_i], 
                                patch_book=class_patches_tensor,
                                area_str=area_rate_str,
                                )
            else: # 没有gt 图像不做改变 if no gt donot change images
                pass

            if not has_gt_flag and patch_is_training: 
                # 训练时，无gt的图直接跳过
                # when training, img with no gt will be skip
                # 测试时，正常测试，不跳过
                # when evaluating, img with no gt will still be evaluated
                continue


            # save for watch
            if patch_is_training and data_i % 100 == 0:
                save_image(patched_img_tensor_ncam, os.path.join(patch_save_dir, str(data_i)+'.png'))
            


            #### 4. resize norm pad
            image_ready = custom_differentiable_transform(
                    img_tensor_rgb_6chw_0to1=patched_img_tensor_ncam,
                    img_metas=img_metas,
                )
            last_time = time_counter(last_time, 'img rsnmpd', time_test_flag)

            if image_ready.isnan().sum()>0:
                print('nan in input image please check!')
            if data_i < 10:
                img_diff_print(img_processed, image_ready,'img_processed','image_ready')


            #### 5. update patch or evaluate



            if patch_is_training: # 在训练 更新patch
                data_give = custom_image_data_give(data_out, image_ready)
                result = model(return_loss=True, **data_give)
                last_time = time_counter(last_time, 'model forward', time_test_flag)
                
                loss = 0
                for key in result:
                    if 'loss' in key:
                        loss = loss + result[key]
                advloss = - loss

                # attack.step img
                optimizer.zero_grad()
                advloss.backward()
                optimizer.step()

                # attack.project img
                class_patches_tensor.data = torch.clamp(class_patches_tensor, 0, 1)
                last_time = time_counter(last_time, 'model backward', time_test_flag)
                print('attack step:', data_i, 
                        'model_loss:',round(float(loss),5),
                        )


            else:
                with torch.no_grad():
                    data_give = custom_image_data_give(data_out, image_ready)
                    data_give = custom_data_postprocess_eval(data_give)
                    result = model(return_loss=False, rescale=True, **data_give)
                    result = custom_result_postprocess(result)
                    results.extend(result)
                last_time = time_counter(last_time, 'model forward', time_test_flag)

            prog_bar.update()


        #### After one (train or val) epoch_d
        if not patch_is_training:
            print(dataset.evaluate(results,)) # eval_kwargs 在DETR3d里面，不是必须用到
            # class patch is evaluated during training. All evaluation scores are saved in nohup-log.

        ##################################
        # save
        ##################################
        if patch_is_training:
            print('=============================')
            print('======= epoch',epoch,'save =========')
            print('=============================')
            save_class_patches_path = os.path.join(patch_save_dir, 'epoch_'+str(epoch)+'class_patches.pkl')
            pickle.dump(class_patches_tensor.cpu(), open(save_class_patches_path, 'wb'))

    return results


