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
from extend.custom_func import *
from extend_common.img_check import img_diff_print
from extend_common.time_counter import time_counter
from extend_common.path_string_split import split_path_string_to_multiname


def single_gpu_test(model, data_loader,
                    scattered_result_prefix=None, 
                    mask_code=None,
                    max_step=None
                    ):

    model.eval()
    results = []
    
    scattered_result_dir = scattered_result_prefix +'_area'+mask_code+'_step'+max_step
    max_step = int(max_step)
    
    os.makedirs(scattered_result_dir, exist_ok=True)
    device = model.src_device_obj
    
    time_test_flag = False
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    for data_i, data_out in enumerate(data_loader):
        
        results_name = str(data_i) + '.pkl'
        scattered_result_path = os.path.join(scattered_result_dir, results_name)
        
        if os.path.exists(scattered_result_path):
            print(scattered_result_path, 'exists! pass!')
            prog_bar.update()
            continue
        else:
            mmcv.dump(' ', scattered_result_path)
            
        
        last_time = time.time()

        # 1. data processing(customed)
        data_out = custom_data_preprocess(data_out)
        img_metas, img_path_list, img_org_np, img_processed, gt_labels_3d = custom_data_work(data_out)
        img_tensor_ncam = custom_img_read_from_img_org(img_org_np, device)
        last_time = time_counter(last_time, 'data load', time_test_flag)


        # 2. read mask
        # instance patch need read mask
        mask_tensor_list = []
        for img_path in img_path_list:
            file_name_valid_list = split_path_string_to_multiname(img_path)[-3:]
            file_name_valid_list.insert(0, '/data/zijian/mycode/BEV_Robust/TransFusion/patch_instance_mask_dir/'+mask_code)
            mask_path = os.path.join(*file_name_valid_list)
            mask_np = cv2.imread(mask_path)
            mask_tensor = torch.from_numpy(mask_np).permute(2,0,1)[[2,1,0]].float()/255.
            mask_tensor = (mask_tensor>0.5).float()
            mask_tensor_list.append(mask_tensor)
        mask_tensor_ncam = torch.stack(mask_tensor_list).to(device)
        
        


        orig_imgs_input = img_tensor_ncam.clone().detach()


        

        # to avoid no_gt
        if gt_labels_3d.shape[0] != 0:

            # random init noise layer
            noise_layer = torch.rand_like(orig_imgs_input)
            noise_layer.requires_grad_(True)
            optimizer = torch.optim.Adam([noise_layer], lr=0.1)


            for step in range(max_step):
                ############ apply patch
                imgs_noisy = torch.where(mask_tensor_ncam>0, noise_layer, orig_imgs_input)

                ############ resize norm pad
                image_ready = custom_differentiable_transform(
                        img_tensor_rgb_6chw_0to1=imgs_noisy,
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
                noise_layer.data = torch.clamp(noise_layer, 0, 1)
                last_time = time_counter(last_time, 'model backward', time_test_flag)
                
                try:
                    print('attack step:', step, 
                        'model_loss:', round(float(loss),3),
                        )
                except Exception as e:
                    print('print_func error:',e)
        else:
            # No gt bbox, directly test
            pass
            
        ############################################    
        # final eval！
        with torch.no_grad():
            ############ apply patch
            noise_layer = noise_layer.clone().detach()
            imgs_noisy = torch.where(mask_tensor_ncam>0, noise_layer, orig_imgs_input)

            ############ resize norm pad
            image_ready = custom_differentiable_transform(
                    img_tensor_rgb_6chw_0to1=imgs_noisy,
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

        last_time = time_counter(last_time, 'eval forward', time_test_flag)

        if 'pts_bbox' in result[0]:
            _scores = result[0]['pts_bbox']['scores_3d']
        elif 'img_bbox' in result[0]:
            _scores = result[0]['img_bbox']['scores_3d']
        else:
            _scores = result[0]['scores_3d']
        if len(_scores)>0:
            print('max conf:', round(float(_scores.max()),3))
        else:
            print('nothing detected')

        mmcv.dump(result, scattered_result_path)
        results.extend(result)
        prog_bar.update()

    return results
