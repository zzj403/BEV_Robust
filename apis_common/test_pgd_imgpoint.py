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


def single_gpu_test(model, data_loader,
                    scattered_result_prefix=None, 
                    img_eps255=None, 
                    point_eps_m=None, 
                    max_step=None,
                    ):

    model.eval()
    results = []
    dataset = data_loader.dataset
    
    scattered_result_dir = scattered_result_prefix +'_imgeps'+img_eps255 +'_pointeps'+point_eps_m +'_step'+max_step
    img_eps255 = float(img_eps255)
    point_eps_m = float(point_eps_m)
    max_step = int(max_step)
    
    os.makedirs(scattered_result_dir, exist_ok=True)
    device = model.src_device_obj
    
    time_test_flag = False
    prog_bar = mmcv.ProgressBar(len(dataset))
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
        img_metas, img_path_list, img_org_np, img_processed, gt_labels_3d, points_tensor = custom_data_work_point(data_out)
        img_tensor_ncam = custom_img_read_from_img_org(img_org_np, device)
        last_time = time_counter(last_time, 'data load', time_test_flag)


        # PGD all region attack, do not need patch info

        orig_imgs_input = img_tensor_ncam.clone().detach()
        orig_pts_input = points_tensor.clone().detach()

        
        img_eps01 = img_eps255 / 255.
        point_eps_m = point_eps_m
        img_lr = img_eps01 / (max_step-2)
        point_lr = point_eps_m / (max_step-2)

        # pgd random start
        delta = torch.rand_like(orig_imgs_input) * 2 * img_eps01 - img_eps01
        imgs_noisy = orig_imgs_input + delta
        imgs_noisy = torch.clamp(imgs_noisy, 0, 1)

        
        # pgd random start
        delta = torch.rand_like(orig_pts_input) * 2 * point_eps_m - point_eps_m
        pts_noisy = orig_pts_input + delta
        # points do not need clamp to [0,1], 
        # but we only change xyz-channel of points
        pts_noisy[:,3:] = orig_pts_input[:,3:]


        

        # to avoid no_gt
        if gt_labels_3d.shape[0] != 0:
            for step in range(max_step):
                imgs_noisy = imgs_noisy.clone().detach().requires_grad_(True)
                pts_noisy  = pts_noisy.clone().detach().requires_grad_(True)

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

                points_ready = pts_noisy

                data_give = custom_image_data_give_point(data_out, image_ready, points_ready)
                result = model(return_loss=True, **data_give)
                last_time = time_counter(last_time, 'model forward', time_test_flag)
                
                loss = 0
                for key in result:
                    if 'loss' in key:
                        loss = loss + result[key]
                advloss = - loss
                advloss.backward()
                
                # attack.step img
                imgs_noisy_grad = imgs_noisy.grad.detach()
                imgs_noisy = imgs_noisy - img_lr * imgs_noisy_grad.sign()
                # attack.project img
                diff = imgs_noisy - orig_imgs_input
                diff = torch.clamp(diff, -img_eps01, img_eps01)
                imgs_noisy = torch.clamp(diff + orig_imgs_input, 0, 1)


                # attack.step point
                pts_noisy_grad = pts_noisy.grad.detach()
                pts_noisy = pts_noisy - point_lr * pts_noisy_grad.sign()
                # attack.project point
                diff = pts_noisy - orig_pts_input
                diff = torch.clamp(diff, - point_eps_m, point_eps_m)
                pts_noisy = diff + orig_pts_input
                # the channel other than xyz cannot be changed
                pts_noisy[:,3:] = orig_pts_input[:,3:]

                
                last_time = time_counter(last_time, 'model backward', time_test_flag)

                try:
                    print('step:', step, 
                        'model_loss:', round(float(loss),3),
                        'img_eps:', round(float(img_eps01),5),
                        'img_lr:', round(float(img_lr),5),
                        'point_eps:', round(float(point_eps_m),5),
                        'point_lr:', round(float(point_lr),5),
                        )
                except Exception as e:
                    print('print_func error:',e)
        else:
            # No gt bbox, directly test
            pass
            
        ############################################    
        # final eval！
        with torch.no_grad():
            imgs_noisy = imgs_noisy.clone().detach()
            ############ resize norm pad
            image_ready = custom_differentiable_transform(
                    img_tensor_rgb_6chw_0to1=imgs_noisy,
                    img_metas=img_metas,
                )
            last_time = time_counter(last_time, 'img rsnmpd', time_test_flag)
            if image_ready.isnan().sum()>0:
                print('nan in input image please check!')
            img_diff_print(img_processed, image_ready,'img_processed','image_ready')

            points_ready = pts_noisy

            data_give = custom_image_data_give_point(data_out, image_ready, points_ready)
            data_give = custom_data_postprocess_eval(data_give)
            result = model(return_loss=False, rescale=True, **data_give)
            result = custom_result_postprocess(result)
            results.extend(result)
            mmcv.dump(result, scattered_result_path)

        last_time = time_counter(last_time, 'eval forward', time_test_flag)
        if 'pts_bbox' in result[0]:
            print('max conf:', round(float(result[0]['pts_bbox']['scores_3d'].max()),3))
        else:
            print('max conf:', round(float(result[0]['scores_3d'].max()),3))

        mmcv.dump(result, scattered_result_path)
        results.extend(result)
        prog_bar.update()

    return results
