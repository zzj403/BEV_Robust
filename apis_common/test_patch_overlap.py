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
from extend_common.patch_apply import apply_patches_by_info_ovlp
from extend_common.path_string_split import split_path_string_to_multiname


def single_gpu_test(model, data_loader,
                    scattered_result_prefix=None,
                    area_rate_str=None,
                    optim_lr=None,
                    optim_step=None,
                    ):
    
    model.eval()
    device = model.src_device_obj
    
    scattered_result_dir = scattered_result_prefix + '_rate'+area_rate_str + '_lr' + str(optim_lr)+'_step' + str(optim_step)
    os.makedirs(scattered_result_dir, exist_ok=True)

    optim_lr = float(optim_lr)
    optim_step = int(optim_step)


    '''
        这里定义一下整体的流程
        for循环遍历
        为每一帧的每一个overlap的物体，定义一个patch的tensor，这个tensor会投影到两个图像上，
        所有的 overlap的物体的tensor投影完成之后，通过for循环，adam lr=0.1循环优化20step，
        当场检测这张攻击后的图，获得pkl文件，保存。
        如果没有overlap物体，则直接原图eval。

        为了保证2视角下，对抗优化的像素的一致性，保证两个视角下的patch视觉相似，patch的tensor尺寸设计为50x50
        
        总体上思路和 instance-level 的攻击类似
        在全部pkl保存完之后，使用 overlap专用的eval方法进行eval
    '''

    
    max_step = optim_step
    cam_num = 6
    results = []

    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    last_time = time.time()
    time_test_flag = False
    for data_i, data_out in enumerate(data_loader):
        
        results_name = str(data_i) + '.pkl'
        scattered_result_path = os.path.join(scattered_result_dir, results_name)
        
        if os.path.exists(scattered_result_path):
            print(scattered_result_path, 'exists! pass!')
            prog_bar.update()
            continue
        else:
            mmcv.dump(' ', scattered_result_path)


        #### 1. data processing(customed)
        data_out = custom_data_preprocess(data_out)
        img_metas, img_path_list, img_org_np, img_processed, gt_labels_3d = custom_data_work(data_out)
        img_tensor_ncam = custom_img_read_from_img_org(img_org_np, device)
        last_time = time_counter(last_time, 'data load', time_test_flag)
        cam_num = len(img_path_list)



        #### 2. read patch info from file
        patch_info_list = []
        for cams_i in range(cam_num):
            img_path = img_path_list[cams_i]
            file_name_valid_list = split_path_string_to_multiname(img_path)[-3:]
            file_name_valid_list.insert(0, '/data/zijian/mycode/BEV_Robust/TransFusion/patch_info_2d3d3dt_square_dir/all')
            info_path = os.path.join(*file_name_valid_list)
            info_path = info_path.replace('.jpg', '.pkl')
            info_i = pickle.load(open(info_path, 'rb'))




            patch_info_list.append(info_i)
        last_time = time_counter(last_time, 'read pkl', time_test_flag)




        #### 3. judge overlap object exists or not
        # 同时存在gt和ovlp物体，才进行攻击， 否则直接eval
        attack_flag = False
        has_gt_flag = gt_labels_3d.shape[0] != 0
        if has_gt_flag:
            ovlp_flag = patch_info_list[0]['objects_info']['is_overlap']
            ovlp_num = ovlp_flag.sum()
            ovlp_exist = (ovlp_num > 0)
            if ovlp_exist:
                attack_flag = True
    
        if attack_flag:
            patch_w = 50
            patch_h = 50
            ovlp_inst_patch_tensor = torch.rand(ovlp_num, 3, patch_h, patch_w).to(device)
            ovlp_inst_patch_tensor.requires_grad_()
            optimizer = torch.optim.Adam([ovlp_inst_patch_tensor], lr=optim_lr)



            for step in range(max_step):
                ####  apply patch
                patched_img_tensor_ncam = img_tensor_ncam.clone()
                for cams_i in range(cam_num):
                    patch_info_in_cami = patch_info_list[cams_i]
                    patched_img_tensor_ncam[cams_i] = apply_patches_by_info_ovlp(
                        info=patch_info_in_cami, 
                        image=patched_img_tensor_ncam[cams_i], 
                        patch_book=ovlp_inst_patch_tensor,
                        area_str=area_rate_str,
                        )
                
                ####  resize norm pad
                image_ready = custom_differentiable_transform(
                        img_tensor_rgb_6chw_0to1=patched_img_tensor_ncam,
                        img_metas=img_metas,
                    )
                last_time = time_counter(last_time, 'img rsnmpd', time_test_flag)
                if image_ready.isnan().sum()>0:
                    print('nan in input image please check!')
                if data_i < 3 and step < 10:
                    img_diff_print(img_processed, image_ready,'img_processed','image_ready')
                

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
                ovlp_inst_patch_tensor.data = torch.clamp(ovlp_inst_patch_tensor, 0, 1)
                last_time = time_counter(last_time, 'model backward', time_test_flag)
                print('attack step:', step, 
                        'model_loss:',round(float(loss),5),
                        )


            #########################
            ##### 攻击结束，最后粘贴patch，准备eval
            ####  apply patch
            patched_img_tensor_ncam = img_tensor_ncam.clone()
            for cams_i in range(cam_num):
                patch_info_in_cami = patch_info_list[cams_i]
                patched_img_tensor_ncam[cams_i] = apply_patches_by_info_ovlp(
                    info=patch_info_in_cami, 
                    image=patched_img_tensor_ncam[cams_i], 
                    patch_book=ovlp_inst_patch_tensor,
                    area_str=area_rate_str,
                    )
            if data_i < 30:
                save_image(patched_img_tensor_ncam, os.path.join(scattered_result_dir, str(data_i)+'.png'))
        else:
            # 同时存在gt和ovlp物体，才进行攻击， 否则直接eval
            patched_img_tensor_ncam = img_tensor_ncam.clone()



        ############################################    
        # 结尾 eval！
        with torch.no_grad():
            ############ resize norm pad
            image_ready = custom_differentiable_transform(
                    img_tensor_rgb_6chw_0to1=patched_img_tensor_ncam,
                    img_metas=img_metas,
                )
            last_time = time_counter(last_time, 'img rsnmpd', time_test_flag)
            if image_ready.isnan().sum()>0:
                print('nan in input image please check!')
            img_diff_print(img_processed, image_ready,'img_processed','image_ready')

            data_give = custom_image_data_give(data_out, image_ready)
            data_give = custom_data_postprocess_eval(data_give)
            result = model(return_loss=False, rescale=True, **data_give)
            result = custom_result_postprocess(result)
            results.extend(result)
            mmcv.dump(result, scattered_result_path)

        prog_bar.update()
    return results



