import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.utils import save_image


def proj_3d_to_2d(points_3d_batch, lidar2img, img_org_shape):
    points_4d_batch = torch.cat([points_3d_batch, torch.ones_like(points_3d_batch[:,0:1])], dim=-1)
    points_2d_batch = points_4d_batch @ lidar2img.t()
    points_2d_batch_depth = points_2d_batch[:,2]
    points_2d_batch_depth_positive = torch.clamp(points_2d_batch_depth, min=1e-5)
    points_2d_batch[:,0] /= points_2d_batch_depth_positive
    points_2d_batch[:,1] /= points_2d_batch_depth_positive
    on_the_image_depth = points_2d_batch_depth > 0
    coor_x, coor_y = points_2d_batch[:,0], points_2d_batch[:,1]
    h, w = img_org_shape
    on_the_image = (coor_x > 0) * (coor_x < w) * (coor_y > 0) * (coor_y < h)
    on_the_image = on_the_image * on_the_image_depth
    return points_2d_batch[:,:2], on_the_image



def safe_img_patch_apply(image, patch, x1, y1):
    assert len(image.shape) == len(patch.shape)
    assert image.shape[0] == patch.shape[0]

    try:
        w_patch = patch.shape[-1]
        h_patch = patch.shape[-2]
        x2 = x1 + w_patch
        y2 = y1 + h_patch
        w_img = image.shape[-1]
        h_img = image.shape[-2]

        x1_use = x1
        x2_use = x2
        y1_use = y1
        y2_use = y2
        patch_use = patch

        if x1 < 0:
            x1_use = 0
            patch_use = patch_use[:, :, -x1:]
        elif x1 > w_img-1:
            return image

        if x2 > w_img:
            x2_use = w_img
            patch_use = patch_use[:, :, :-(x2 - w_img)]
        elif x2 < 0:
            return image

        if y1 < 0:
            y1_use = 0
            patch_use = patch_use[:, -y1:, :]
        elif y1 > h_img - 1:
            return image

        if y2 > h_img:
            y2_use = h_img
            patch_use = patch_use[:, :-(y2 - h_img), :]
        elif y2 < 0:
            return image

        image[:, y1_use:y2_use, x1_use:x2_use] = patch_use
    except:
        return image
        print('safe_img_patch_apply failed, skip this patch')
    return image




def apply_patches_by_info(info, image, patch_book, area_str):
    local_classname_list = info['objects_info']['class_name']
    local_classidx_list = info['objects_info']['class_idx']

    # 重新排序，按照画面深度uvd的d排序，以保证从远到近粘贴
    _d = info['objects_info']['uvd_center'][:, 2]
    d_index = _d.sort(descending=True)[1]

    d_index_list = d_index.tolist()

    for i in d_index_list:
        if info['patch_visible_bigger'][i]: # 可见
            assert info['patch_2d']['patch_vertices'][i] != None
            use_patch_vertices_uv = info['patch_2d']['patch_vertices'][i][area_str][:,:2]
            use_patch_classname = local_classidx_list[i]
            use_patch = patch_book[use_patch_classname]
            patch_use = use_patch
            patch_h, patch_w = patch_use.shape[-2:]
            img_h, img_w = image.shape[-2:]

            if len(use_patch_vertices_uv.unique()) <= 4:
                # 低级抠图
                u1 = int(use_patch_vertices_uv[:,0].min())
                u2 = int(use_patch_vertices_uv[:,0].max())
                v1 = int(use_patch_vertices_uv[:,1].min())
                v2 = int(use_patch_vertices_uv[:,1].max())

                need_patch_h = v2 - v1
                need_patch_w = u2 - u1

                if need_patch_h > 0 and need_patch_w > 0:
                    patch_use = F.interpolate(patch_use[None], (need_patch_h, need_patch_w), mode='bilinear', align_corners=False)[0]
                    image = safe_img_patch_apply(
                        image=image, 
                        patch=patch_use, 
                        x1=u1, 
                        y1=v1
                    )
            else:
                # 高级变换

                patch_pad = F.pad(
                    patch_use, (0, img_w-patch_w, 0, img_h-patch_h),
                    )
                patch_mask_pad = torch.zeros_like(image)
                patch_mask_pad[:,:patch_h,:patch_w] = 1
                startpoints = [
                    (0, 0), 
                    (patch_w-1, 0),
                    (patch_w-1, patch_h-1),
                    (0, patch_h-1),
                ]
                endpoints = use_patch_vertices_uv
                patch_pad_trans = transforms.functional.perspective(
                                        patch_pad,
                                        startpoints=startpoints,
                                        endpoints=endpoints
                                        )
                patch_mask_pad_trans = transforms.functional.perspective(
                                        patch_mask_pad,
                                        startpoints=startpoints,
                                        endpoints=endpoints
                                        )
                image = torch.where(patch_mask_pad_trans > 0.5, patch_pad_trans, image)
    return image



def apply_patches_by_info_ovlp(info, image, patch_book, area_str):
    local_classname_list = info['objects_info']['class_name']
    local_classidx_list = info['objects_info']['class_idx']
    
    local_ovlp_flag = info['objects_info']['is_overlap']

    # 重新排序，按照画面深度uvd的d排序，以保证从远到近粘贴
    _d = info['objects_info']['uvd_center'][:, 2]
    d_index = _d.sort(descending=True)[1]
    
    # 需要判断，当前的obj是ovlp的第几个，粘贴对应的patch

    d_index_list = d_index.tolist()

    for i in d_index_list:
        if info['patch_visible_bigger'][i] and local_ovlp_flag[i]: # 可见 且 重叠
            assert info['patch_3d']['patch_vertices'][i] != None
            use_patch_vertices_uv = info['patch_3d']['patch_vertices'][i][area_str][:,:2]
            # 需要判断，当前的obj是ovlp的第几个，粘贴对应的patch
            use_patch_idx = local_ovlp_flag[:(i+1)].sum() - 1
            use_patch = patch_book[use_patch_idx]
            patch_use = use_patch
            patch_h, patch_w = patch_use.shape[-2:]
            img_h, img_w = image.shape[-2:]

            if len(use_patch_vertices_uv.unique()) <= 4:
                # 低级抠图
                u1 = int(use_patch_vertices_uv[:,0].min())
                u2 = int(use_patch_vertices_uv[:,0].max())
                v1 = int(use_patch_vertices_uv[:,1].min())
                v2 = int(use_patch_vertices_uv[:,1].max())

                need_patch_h = v2 - v1
                need_patch_w = u2 - u1

                if need_patch_h > 0 and need_patch_w > 0:
                    patch_use = F.interpolate(patch_use[None], (need_patch_h, need_patch_w), mode='bilinear', align_corners=False)[0]
                    image = safe_img_patch_apply(
                        image=image, 
                        patch=patch_use, 
                        x1=u1, 
                        y1=v1
                    )
            else:
                # 高级变换

                patch_pad = F.pad(
                    patch_use, (0, img_w-patch_w, 0, img_h-patch_h),
                    )
                patch_mask_pad = torch.zeros_like(image)
                patch_mask_pad[:,:patch_h,:patch_w] = 1
                startpoints = [
                    (0, 0), 
                    (patch_w-1, 0),
                    (patch_w-1, patch_h-1),
                    (0, patch_h-1),
                ]
                endpoints = use_patch_vertices_uv
                patch_pad_trans = transforms.functional.perspective(
                                        patch_pad,
                                        startpoints=startpoints,
                                        endpoints=endpoints
                                        )
                patch_mask_pad_trans = transforms.functional.perspective(
                                        patch_mask_pad,
                                        startpoints=startpoints,
                                        endpoints=endpoints
                                        )
                image = torch.where(patch_mask_pad_trans > 0.5, patch_pad_trans, image)
    return image




def apply_patches_by_info_4side(info, image, instance_token_book, patch_book_4side, area_str):

    # 重新排序，按照画面深度uvd的d排序，以保证从远到近粘贴
    _d = info['objects_info']['uvd_center'][:, 2]
    d_index = _d.sort(descending=True)[1]

    d_index_list = d_index.tolist()

    for i_obj in d_index_list:
        if info['patch_visible_bigger'][i_obj]: # 可见
            assert info['patch_3d_temporal']['patch_vertices'][i_obj] != None
            use_patch_vertices_uv_4side = info['patch_3d_temporal']['patch_vertices'][i_obj][area_str]
            use_patch_vertices_uv_side_visible = info['patch_3d_temporal']['boxside_visiblie'][i_obj]

            for side_j in range(4):
                try:
                    if use_patch_vertices_uv_side_visible[side_j]:

                        use_patch_instance_token = info['objects_info']['instance_tokens'][i_obj]
                        # 注意！ 每一个token对应4个patch，所以，
                        # 找到index之后要*4，
                        # 再加 0,1,2,3
                        patch_4_start_idx = instance_token_book.index(use_patch_instance_token) * 4
                        use_patch_vertices_uv = use_patch_vertices_uv_4side[side_j][:,:2].cuda()

                        use_patch = patch_book_4side[patch_4_start_idx + side_j]
                        patch_use = use_patch
                        patch_h, patch_w = torch.tensor(patch_use.shape[-2:]).cuda().long()
                        img_h, img_w = torch.tensor(image.shape[-2:]).cuda().long()

                        if len(use_patch_vertices_uv.unique()) <= 4:
                            # 低级抠图
                            u1 = (use_patch_vertices_uv[:,0].min()).long()
                            u2 = (use_patch_vertices_uv[:,0].max()).long()
                            v1 = (use_patch_vertices_uv[:,1].min()).long()
                            v2 = (use_patch_vertices_uv[:,1].max()).long()

                            need_patch_h = v2 - v1
                            need_patch_w = u2 - u1

                            if need_patch_h > 0 and need_patch_w > 0:
                                patch_use = F.interpolate(patch_use[None], (need_patch_h, need_patch_w), mode='bilinear', align_corners=False)[0]
                                image = safe_img_patch_apply(
                                    image=image, 
                                    patch=patch_use, 
                                    x1=u1, 
                                    y1=v1
                                )
                        else:
                            endpoints = use_patch_vertices_uv
                            # 判断 endpoints 是否都在图像中，如果是，使用节约显存版本
                            # 留一点边
                            edge_width = 2
                            u_in = (edge_width < endpoints[:,0]) * (endpoints[:,0] < img_w - edge_width - 1)
                            v_in = (edge_width < endpoints[:,1]) * (endpoints[:,1] < img_h - edge_width - 1)

                            endpoints_allinimage = u_in.all() * v_in.all()


                            need_old_version = ~endpoints_allinimage

                            if need_old_version == False:
                                # 高级变换 (节约显存版本)
                                # 思路： 只使用小图像进行变换，变换后抠图粘到大图中
                                u_min = endpoints[:,0].min()
                                u_max = endpoints[:,0].max()
                                v_min = endpoints[:,1].min()
                                v_max = endpoints[:,1].max()

                                u_min_int = (u_min.floor() - 1).long()
                                u_max_int = (u_max.ceil()  + 1).long()
                                v_min_int = (v_min.floor() - 1).long()
                                v_max_int = (v_max.ceil()  + 1).long()

                                small_region_w = u_max_int - u_min_int
                                small_region_h = v_max_int - v_min_int

                                # patch粘贴区域，可能比patch更大，也可能更小，我们取大值进行绘图

                                small_region_w_share = max(small_region_w, patch_w)
                                small_region_h_share = max(small_region_h, patch_h)
                                u_max_int_real = u_min_int + small_region_w_share
                                v_max_int_real = v_min_int + small_region_h_share
                                
                                patch_pad = patch_use.new_zeros(3, small_region_h_share, small_region_w_share)
                                patch_pad[:,:patch_h,:patch_w] = patch_use

                                # 注意，如果patch更大，贴回到原图的时候，上下左右边界就都不同了，要重新计算

                                # 注意，如果patch更大，可能patch左上角对齐粘贴处，尺寸就又超出去了，需要回到老方法
                                need_old_version = False
                                if u_max_int_real > img_w - edge_width - 1 or v_max_int_real > img_h - edge_width - 1:
                                    need_old_version = True


                            if need_old_version == False:
                                patch_mask_pad = torch.zeros_like(patch_pad)
                                patch_mask_pad[:,:patch_h,:patch_w] = 1
                                startpoints = torch.cuda.FloatTensor([
                                    (0, 0), 
                                    (patch_w-1, 0),
                                    (patch_w-1, patch_h-1),
                                    (0, patch_h-1),
                                ])

                                endpoints_small_region = torch.zeros_like(endpoints)
                                endpoints_small_region[:,0] = endpoints[:,0] - u_min_int
                                endpoints_small_region[:,1] = endpoints[:,1] - v_min_int

                                patch_pad_trans = transforms.functional.perspective(
                                                        patch_pad,
                                                        startpoints=startpoints.cpu(),
                                                        endpoints=endpoints_small_region.cpu()
                                                        ) # 输出居然有可能 = 1 + 1.19e-7 大于1了！
                                patch_mask_pad_trans = transforms.functional.perspective(
                                                        patch_mask_pad,
                                                        startpoints=startpoints.cpu(),
                                                        endpoints=endpoints_small_region.cpu()
                                                        )
                                small_region = torch.where(
                                    patch_mask_pad_trans > 0.5, 
                                    patch_pad_trans, 
                                    image[:, v_min_int:v_max_int_real, u_min_int:u_max_int_real]
                                    )

                                image[:, v_min_int:v_max_int_real, u_min_int:u_max_int_real] = small_region.clamp(0,1)
                            else:
                                # 高级变换 (原始版本)
                                patch_pad = F.pad(
                                    patch_use, (0, img_w-patch_w, 0, img_h-patch_h),
                                    )
                                patch_mask_pad = torch.zeros_like(image)
                                patch_mask_pad[:,:patch_h,:patch_w] = 1
                                startpoints = torch.cuda.FloatTensor([
                                    (0, 0), 
                                    (patch_w-1, 0),
                                    (patch_w-1, patch_h-1),
                                    (0, patch_h-1),
                                ])
                                endpoints = use_patch_vertices_uv
                                patch_pad_trans = transforms.functional.perspective(
                                                        patch_pad,
                                                        startpoints=startpoints.cpu(),
                                                        endpoints=endpoints.cpu()
                                                        )
                                patch_mask_pad_trans = transforms.functional.perspective(
                                                        patch_mask_pad,
                                                        startpoints=startpoints.cpu(),
                                                        endpoints=endpoints.cpu()
                                                        )
                                image = torch.where(patch_mask_pad_trans > 0.5, patch_pad_trans, image).clamp(0,1)
                except:
                    print('error happend, skip this side\'s patch')
    return image


