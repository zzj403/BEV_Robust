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
                    eps255=None
                    ):

    model.eval()
    results = []
    dataset = data_loader.dataset
    
    scattered_result_dir = scattered_result_prefix + 'eps'+eps255
    eps255 = float(eps255)
    

    
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
        img_metas, img_path_list, img_org_np, img_processed, gt_labels_3d = custom_data_work(data_out)
        img_tensor_ncam = custom_img_read_from_img_org(img_org_np, device)
        last_time = time_counter(last_time, 'data load', time_test_flag)


        # FGSM all region attack, do not need patch info
        
        imgs_noisy = img_tensor_ncam.clone().detach()
        orig_imgs_input = img_tensor_ncam.clone().detach()
        
        eps01 = eps255 / 255.
        lr = eps01
        max_step = 1

        # to avoid no_gt
        if gt_labels_3d.shape[0] != 0:
            for step in range(max_step):
                imgs_noisy = imgs_noisy.clone().detach().requires_grad_(True)

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
                advloss.backward()
                
                # attack.step img
                imgs_noisy_grad = imgs_noisy.grad.detach()
                imgs_noisy = imgs_noisy - lr * imgs_noisy_grad.sign()
                
                # attack.project img
                diff = imgs_noisy - orig_imgs_input
                diff = torch.clamp(diff, - eps01, eps01)
                imgs_noisy = torch.clamp(diff + orig_imgs_input, 0, 1)
                
                last_time = time_counter(last_time, 'model backward', time_test_flag)
                try:
                    print('attack step:', step, 
                        'model_loss:',round(float(loss),5),
                        'fgsm_eps:', round(float(eps01),5),
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
