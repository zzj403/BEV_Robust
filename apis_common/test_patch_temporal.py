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
from extend_common.patch_apply import apply_patches_by_info_4side
from extend_common.path_string_split import split_path_string_to_multiname
from extend_common.get_scene_start_idx import get_scene_start_idx


def single_gpu_test(model, data_loader,
                    scattered_result_prefix=None, 
                    area_rate_str=None,
                    optim_lr=None,
                    optim_step=None,
                    index_min = None, 
                    index_max = None,
                    ):
    
    model.eval()
    dataset = data_loader.dataset
    device = model.src_device_obj
    
    scattered_result_dir = scattered_result_prefix +'_area'+area_rate_str+'_lr'+optim_lr+'_step' + optim_step
    os.makedirs(scattered_result_dir, exist_ok=True)

    optim_lr = float(optim_lr)
    optim_step = int(optim_step)

    scene_start_idx_list = get_scene_start_idx()
    max_epoch_local = optim_step
    
    patch_info_list_database = {}
    time_test_flag = False


    results = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    last_time = time.time()
    for data_i, data_out in enumerate(data_loader):
        if data_i < index_min:
            prog_bar.update()
            continue
        if data_i > index_max:
            break
        
        #### 1. data processing(customed)
        data_out = custom_data_preprocess(data_out)
        _, img_path_list, _, _, _ = custom_data_work(data_out)
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

        
        '''
            由于我们要一个场景（大概40帧左右），一起进行攻击
            所以我需要先遍历数据集，把这一个场景的数据先拿出来，统计里面instance的数量，构建一个 patch 库
            然后再在读取出的这一个场景的数据里做攻击

            如果是场景的第0帧
            则开始遍历当前场景，直到下一个第0帧的出现，这时候暂存下一个第0帧
            遍历场景时，存下所有的注释信息，
            并从之前存好的 patch info 中 获取 instance_token
        '''


        scene_start_here_flag = (data_i in scene_start_idx_list)
        
        go_to_training_flag = False
        
        if data_i == 0:
            # 第0帧
            # start new
            data_in_scene_list = []
            patch_info_in_scene_list = []
            data_i_list = []
            data_in_scene_list.append(data_out)
            patch_info_in_scene_list.append(patch_info_list)
            data_i_list.append(data_i)
        elif scene_start_here_flag and data_i > 0:
            # 之后的每一个首帧
            # 存一个连续场景的全部 data 和 patch_info
            # end old
            try:
                data_in_scene_list_full = data_in_scene_list
                patch_info_in_scene_list_full = patch_info_in_scene_list
                data_i_list_full = data_i_list
                go_to_training_flag = True
            except:
                print('start from data_i:', data_i)
            # start new
            data_in_scene_list = []
            patch_info_in_scene_list = []
            data_i_list = []
            data_in_scene_list.append(data_out)
            patch_info_in_scene_list.append(patch_info_list)
            data_i_list.append(data_i)
        elif data_i == len(dataset)-1:
            data_in_scene_list.append(data_out)
            patch_info_in_scene_list.append(patch_info_list)
            data_i_list.append(data_i)
            # 最后一帧
            # end old
            data_in_scene_list_full = data_in_scene_list
            patch_info_in_scene_list_full = patch_info_in_scene_list
            data_i_list_full = data_i_list
            go_to_training_flag = True
        else:
            data_in_scene_list.append(data_out)
            patch_info_in_scene_list.append(patch_info_list)
            data_i_list.append(data_i)
        prog_bar.update()
        
        if go_to_training_flag:
            # local dataset: data_in_scene_list_full
            # local dataset: patch_info_in_scene_list_full
            # local dataset: data_i_list_full
            scene_length = len(data_in_scene_list_full)
            
            ###### 1.构建patch库 Establish local-scene patchbook
            # 每个物体的4个面，都放patch，
            # patchtensor的形状, 由实际的patchsize确定，兼容正方形patch
            instance_token_list = []
            patch_4side_book_list = []
            for i_local in range(scene_length):
                # 1.把数据拿出来，处理数据
                data_local = data_in_scene_list_full[i_local]
                patch_info_local = patch_info_in_scene_list_full[i_local]
                _, _, _, _, gt_labels_3d = custom_data_work(data_local)
                # 2.判断有没有gt
                # 防止出现 no_gt
                has_gt_flag = (gt_labels_3d.shape[0] != 0) and (type(patch_info_local[0]) != str)
                if has_gt_flag:
                    scene_name = patch_info_local[0]['scene_info']['scene_name']
                    instance_tokens_i = patch_info_local[0]['objects_info']['instance_tokens']
                    for inst_tk_idx in range(len(instance_tokens_i)):
                        instance_token = instance_tokens_i[inst_tk_idx]
                        if not instance_token in instance_token_list:
                            # 添加patch 
                            # 根据最先出现的patch，标注的信息，添加4个patch
                            for j_cam_1frame in range(cam_num):
                                if patch_info_local[j_cam_1frame]['patch_visible_bigger'][inst_tk_idx]:
                                    # 如果可以被，当前的camera看到，则添加，否则不添加
                                    patch_3d_wh = patch_info_local[j_cam_1frame]['patch_3d_temporal']['patch_3d_wh'][inst_tk_idx]
                                    patch_3d_wh_use = patch_3d_wh[area_rate_str]

                                    patch_4side_ = []
                                    for j_side in range(4):
                                        patch_w_real, patch_h_real = patch_3d_wh_use[j_side]
                                        # 遵循每1m 100pix的密度
                                        patch_w_tensor = int(patch_w_real*100)
                                        patch_h_tensor = int(patch_h_real*100)
                                        patch_jside_ = torch.rand(3, patch_h_tensor, patch_w_tensor).to(device)
                                        patch_jside_.requires_grad_()
                                        patch_4side_.append(patch_jside_)

                                    instance_token_list.append(instance_token)
                                    patch_4side_book_list.extend(patch_4side_)

            # 为这些patch定义 优化器
            optimizer = torch.optim.Adam(patch_4side_book_list, lr=optim_lr)

            # 以后每一次取用，都需要，结合instance_token_list获取 token对应的index，再用

            
            for epoch_local in range(max_epoch_local):
                print('scene_name:', scene_name,'start epoch_local', epoch_local,'training')
                for i_local in range(scene_length):

                    ##############  把数据拿出来，处理数据 Take out the data and process the data
                    data_local = data_in_scene_list_full[i_local]
                    patch_info_local = patch_info_in_scene_list_full[i_local]
                    img_metas, img_path_list, img_org_np, img_processed, gt_labels_3d = custom_data_work(data_local)
                    img_tensor_ncam = custom_img_read_from_img_org(img_org_np, device)
                    last_time = time_counter(last_time, 'data process', time_test_flag)

                    ############## apply patch
                    patched_img_tensor_ncam = img_tensor_ncam.clone()
                    # in case of no_gt
                    has_gt_flag = (gt_labels_3d.shape[0] != 0) and (type(patch_info_local[0]) != str)
                    if has_gt_flag:
                        # apply patch
                        for cams_i in range(cam_num):
                            patch_info_in_cami = patch_info_local[cams_i]
                            patched_img_tensor_ncam[cams_i] = apply_patches_by_info_4side(
                                info=patch_info_in_cami, 
                                image=patched_img_tensor_ncam[cams_i], 
                                instance_token_book=instance_token_list,
                                patch_book_4side=patch_4side_book_list,
                                area_str=area_rate_str,
                                )
                            # patched_img_tensor_ncam[cams_i] = (patched_img_tensor_ncam[cams_i] + patch_4side_book_list[0].mean()/1000).clamp(0,1)
                    else: # no gt，图像不做改变，也不必优化patch
                        continue

                    last_time = time_counter(last_time, 'apply patch', time_test_flag)

                    ############ resize norm pad
                    image_ready = custom_differentiable_transform(
                            img_tensor_rgb_6chw_0to1=patched_img_tensor_ncam,
                            img_metas=img_metas,
                        )
                    last_time = time_counter(last_time, 'img rsnmpd', time_test_flag)


                    if image_ready.isnan().sum()>0:
                        print('nan in input image please check!')

                    data_i_actual = data_i_list_full[i_local]
                    if data_i_actual < 100 and epoch_local < 3 and i_local < 3:
                        img_diff_print(img_processed, image_ready,'img_processed','image_ready')


                    data_give = custom_image_data_give(data_local, image_ready)
                    result = model(return_loss=True, **data_give) # 经过model， data中的img会被修改为[6,3,H,W]
                    last_time = time_counter(last_time, 'model forward', time_test_flag)
                    loss = 0
                    for key in result:
                        if 'loss' in key:
                            loss = loss + result[key]
                    advloss = - loss
                    optimizer.zero_grad()
                    advloss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    last_time = time_counter(last_time, 'model backward', time_test_flag)

                    for _patch_i in range(len(patch_4side_book_list)):
                        patch_4side_book_list[_patch_i].data = torch.clamp(patch_4side_book_list[_patch_i], 0, 1)
                    last_time = time_counter(last_time, 'patch clamp', time_test_flag)
                    print('attack step:', i_local, 
                            'model_loss:',round(float(loss),5),
                            )
            #########################
            ##### 攻击结束，最后再遍历一遍，粘贴patch，eval
            print('scene_name:', scene_name,'start eval')
            prog_bar_local_eval = mmcv.ProgressBar(scene_length)
            with torch.no_grad():
                for i_local in range(scene_length):

                    #################  把数据拿出来，处理数据
                    data_local = data_in_scene_list_full[i_local]
                    patch_info_local = patch_info_in_scene_list_full[i_local]
                    img_metas, img_path_list, img_org_np, img_processed, gt_labels_3d = custom_data_work(data_local)
                    img_tensor_ncam = custom_img_read_from_img_org(img_org_np, device)

                    ################  安装patch
                    patched_img_tensor_ncam = img_tensor_ncam.clone()
                    # 防止出现 no_gt
                    has_gt_flag = (gt_labels_3d.shape[0] != 0) and (type(patch_info_local[0]) != str)
                    if has_gt_flag:
                        # apply patch
                        for cams_i in range(cam_num):
                            patch_info_in_cami = patch_info_local[cams_i]
                            patched_img_tensor_ncam[cams_i] = apply_patches_by_info_4side(
                                info=patch_info_in_cami, 
                                image=patched_img_tensor_ncam[cams_i], 
                                instance_token_book=instance_token_list,
                                patch_book_4side=patch_4side_book_list,
                                area_str=area_rate_str,
                                )
                    else: # 没有gt，图像不做改变，直接eval
                        pass

                    ############ resize norm pad
                    image_ready = custom_differentiable_transform(
                            img_tensor_rgb_6chw_0to1=patched_img_tensor_ncam,
                            img_metas=img_metas,
                        )
                    last_time = time_counter(last_time, 'img rsnmpd', time_test_flag)
                    if image_ready.isnan().sum()>0:
                        print('nan in input image please check!')
                    if i_local < 3:
                        img_diff_print(img_processed, image_ready,'img_processed','image_ready')

                    data_give = custom_image_data_give(data_local, image_ready)
                    data_give = custom_data_postprocess_eval(data_give)
                    result = model(return_loss=False, rescale=True, **data_give)
                    result = custom_result_postprocess(result)
                    results.extend(result)

                    data_i_actual = data_i_list_full[i_local]
                    scattered_result_path = os.path.join(scattered_result_dir, str(data_i_actual)+'.pkl')
                    mmcv.dump(result, scattered_result_path)
                    if data_i_actual < 100:
                        save_image(patched_img_tensor_ncam, os.path.join(scattered_result_dir, str(data_i_actual)+'.png'))
                    prog_bar_local_eval.update()
                print()
    return results



