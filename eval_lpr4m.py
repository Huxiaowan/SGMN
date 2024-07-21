
import os
import sys
import math
import time
import json
import argparse
import numpy as np
try:
    import pickle5 as pkl
except:
    import pickle as pkl

from PIL import Image
from tqdm import tqdm
import multiprocessing
from collections import defaultdict

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn.functional as F

from modules.modeling_sgmn import CLIP4Clip
from modules.util_module import genPatchmask, IOU
from dataloaders.rawframe_util import RawVideoExtractor
image_resolution = 224
rawVideoExtractor = RawVideoExtractor(centercrop=True, size=(image_resolution, image_resolution))  # (12, 7)
rawImageExtractor = RawVideoExtractor(centercrop=True, size=(image_resolution, image_resolution))  # (12, 7)
 
 # Generate grid
patchsize = (32, 32)
h, w = patchsize
grid_h = image_resolution // h
grid_w = image_resolution // w
GRIDS = []
for i in range(grid_h):
    for j in range(grid_w):
        ltrb = [j*w, i*h, (j+1)*w, (i+1)*h]    
        GRIDS.append(ltrb)   

def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str,
                        default="./ckpts/ckpt_lpr4m_looseType/pytorch_model.bin.0",
                        help='none')
    parser.add_argument('--sim_header', type=str,
                        default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help='none')
    parser.add_argument('--cross_num_hidden_layers', type=int,
                        default=2,
                        help='none')
    parser.add_argument('--loose_type', action='store_true', help='none')
    parser.add_argument('--add_text', action='store_true', help='')
    parser.add_argument('--recons_feat', action='store_true', help='')
    parser.add_argument('--mo_fusion', action='store_true', help='')
    parser.add_argument('--ipd', action='store_true', help='')
    parser.add_argument('--embedding_sim',
                        action='store_true',
                        help='')
    parser.add_argument('--mode', type=str,
                        default="video",
                        choices=["video", "frame"],
                        help='none')
    parser.add_argument('--pretrained_clip_name', type=str)
    parser.add_argument('--backbone_name', type=str)
    args = parser.parse_args()
    return args

def _get_rawvideo(framepath_lst, choice_video_ids, rawVideoExtractor, max_frames=10, slice_framepos=2, frame_order=0):
    num_video = len(choice_video_ids)
    video_mask = np.zeros((num_video, max_frames), dtype=np.int64)
    max_video_length = [0] * num_video

    # Pair x L x T x 3 x H x W
    video = np.zeros((num_video, max_frames, 1, 3,
                      rawVideoExtractor.size_h, rawVideoExtractor.size_w), dtype=np.float64)

    for i, video_id in enumerate(choice_video_ids):
        raw_video_data = rawVideoExtractor.get_video_data(framepath_lst[i])
        raw_video_data = raw_video_data['video']

        if len(raw_video_data.shape) > 3:
            raw_video_data_clip = raw_video_data
            # L x T x 3 x H x W
            raw_video_slice = rawVideoExtractor.process_raw_data(raw_video_data_clip)
            if max_frames < raw_video_slice.shape[0]:
                if slice_framepos == 0:
                    video_slice = raw_video_slice[:max_frames, ...]
                elif slice_framepos == 1:
                    video_slice = raw_video_slice[-max_frames:, ...]
                else:
                    sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=max_frames, dtype=int)
                    video_slice = raw_video_slice[sample_indx, ...]
            else:
                video_slice = raw_video_slice

            video_slice = rawVideoExtractor.process_frame_order(video_slice, frame_order=frame_order)

            slice_len = video_slice.shape[0]
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
            if slice_len < 1:
                pass
            else:
                video[i][:slice_len, ...] = video_slice
        else:
            print("video path: {} error. video id: {}".format(video_path, video_id))

    for i, v_length in enumerate(max_video_length):
        video_mask[i][:v_length] = [1] * v_length

    video = torch.tensor(video)
    video_mask = torch.tensor(video_mask)
    return video, video_mask
    
def check_image_file(filename):
    return any([filename.endswith(extention) for extention in
                ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP']])

def most_freq(List):
    return max(set(List), key = List.count)

def mergeSubdict(data_root):
    dstdir = data_root
    liveid_lst = [t for t in os.listdir(dstdir) if t.isnumeric()]
    for liveid in liveid_lst:
        dict_dir = os.path.join(dstdir, liveid)
        dict_lst = [_t for _t in os.listdir(dict_dir) if _t.startswith("frame2item_") and "final" not in _t]
        fpath_lst = [os.path.join(dict_dir, _t) for _t in dict_lst]

        rst = dict()
        for _path in fpath_lst:
            with open(_path, "r") as f:
                data = json.load(f)
            for _key, _val in data.items():
                new_key = "_".join([liveid, _key])
                if new_key not in rst:
                    rst[new_key] = list()
                rst[new_key] += _val
        with open(os.path.join(dict_dir, "_frame2item.json"), "w") as f:
            json.dump(rst, f, indent=1)
    return 

def fmtGroundTruth(liveclip_root, liveframe_root):
    with open("/home/bad_to_gooditem.json", "r") as f:
        bad2good = json.load(f)

    live_ids = [t for t in os.listdir(liveclip_root) if t.isnumeric()]
    print(f"Total number of liveid: {len(live_ids)}")
    
    clip2gtitem = dict()
    for _liveid in tqdm(live_ids, desc="live forward"):
        live_path = os.path.join(liveclip_root, _liveid)
        clip_lst = os.listdir(live_path)
        clip_lst = [t.strip(".mp4") for t in clip_lst]
        item_lst = [t.split("_")[-1] for t in clip_lst]
        clip_lst = ["_".join(t.split("_")[:-1]) for t in clip_lst]
        for _clip, _item in zip(clip_lst, item_lst):
            if _clip not in clip2gtitem:
                clip2gtitem[_clip] = list()
            if _item in bad2good:
                _item = bad2good[_item]
            clip2gtitem[_clip].append(_item)
    clip2gtitem = {k:v for k,v in sorted(clip2gtitem.items(), key=lambda x:x[0])}
    print(f"Number of Clip: {len(clip2gtitem)}")
    

    clip_ids = os.listdir(liveframe_root)
    frame2gtitem = dict()
    for _clipid in tqdm(clip_ids, desc="clip forward"):
        clip_path = os.path.join(liveframe_root, _clipid)
        frame_lst = os.listdir(clip_path)
        frame_lst = [t.strip(".jpg") for t in frame_lst]
        item_lst = [t.split("_")[-2] for t in frame_lst]
        frame_lst = ["_".join(t.split("_")[:-2]+t.split("_")[-1:]) for t in frame_lst]
        for _frame, _item in zip(frame_lst, item_lst):
            if _frame not in frame2gtitem:
                frame2gtitem[_frame] = list()
            if _item in bad2good:
                _item = bad2good[_item]
            frame2gtitem[_frame].append(_item)
    frame2gtitem = {k:v for k,v in sorted(frame2gtitem.items(), key=lambda x:x[0])}
    print(f"Number of Frame: {len(frame2gtitem)}")

    # Write
    with open("/home/clip_to_gtitem.json", "w") as f:
        json.dump(clip2gtitem, f, indent=1)
    with open("/home/frame_to_gtitem.json", "w") as f:
        json.dump(frame2gtitem, f, indent=1)

    return

def init_model(sim_header, cross_num_hidden_layers, loose_type, device, ckpt_path=None, training=True, recons_feat=False, embedding_sim=True, add_text=False, pretrained_clip_name="eval", backbone_name="ViT-B/32", mo_fusion=False):
    # import pdb; pdb.set_trace()
    if ckpt_path is not None:
        model_state_dict = torch.load(ckpt_path, map_location=device)  # torch.device("cpu") torch.device("cuda:0")
    else:
        model_state_dict = None

    model = CLIP4Clip.from_pretrained(
        cross_model_name="cross-base", 
        max_words=32, max_frames=10, linear_patch="2d", 
        pretrained_clip_name=pretrained_clip_name, 
        loose_type=loose_type, 
        sim_header=sim_header, 
        cross_num_hidden_layers=cross_num_hidden_layers,
        state_dict=model_state_dict, 
        cache_dir=None, 
        type_vocab_size=2, 
        training=False, 
        recons_feat=recons_feat, 
        embedding_sim=embedding_sim, 
        add_text=add_text,
        backbone_name=backbone_name,
        mo_fusion=mo_fusion)
    
    model.load_state_dict(model_state_dict, strict=True)
    model.eval()
    del model_state_dict
    return model

def moveDir():
    import shutil
    dataroot = "/data/testset_frame_thrsec"
    dst_root = "/data/testset_frame"
    
    clip_ids = os.listdir(dataroot)
    for _clipid in tqdm(clip_ids, desc="clip forward"):
        dst_clipid = "_".join(_clipid.split("_")[:-1])
        dst_dir = os.path.join(dst_root, dst_clipid)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        clip_path = os.path.join(dataroot, _clipid)
        
        frame_lst = os.listdir(clip_path)
        dst_frame_lst = ["_".join(t.split("_")[:-2]+t.split("_")[-1:]) for t in frame_lst]
        for src_frame, dst_frame in zip(frame_lst, dst_frame_lst):
            src_path = os.path.join(clip_path, src_frame)
            dst_path = os.path.join(dst_dir, dst_frame)
            if os.path.isfile(dst_path):
                continue
            shutil.copy(src_path, dst_path)
    return

def calculate_acc(pred, gt, top_k=10, root_dir=None, write_dir=None, output_badcase=False, return_value=False):
    n_query = len(gt)
    hit_top = np.array([1, 3, 5, 10])
    hit_top_cnt = np.array([0, 0, 0, 0])
    # print('Cal results...')
    for q_index, (q_name, q_pred) in enumerate(pred.items()):
        #if (q_index + 1) % 1000 == 0:
        #    print('Processing {} queries'.format(q_index))
        if isinstance(q_pred[0], str):
            q_pred = [q_pred]
        tmp_top_hit = np.array([0, 0, 0, 0])
        for k in range(top_k):

            if k > len(q_pred)-1:
                cur_pred = q_pred[-1][0]
            else:
                cur_pred = q_pred[k][0]

            if cur_pred in gt[q_name]:
                if k < hit_top[0]:
                    tmp_top_hit[0:] = 1
                elif k < hit_top[1]:
                    tmp_top_hit[1:] = 1
                elif k < hit_top[2]:
                    tmp_top_hit[2:] = 1
                elif k < hit_top[3]:
                    tmp_top_hit[3:] = 1

        hit_top_cnt += tmp_top_hit
    
    # import pdb; pdb.set_trace()
    print('Top{} Hit: {:.2f}%'.format(hit_top[0], (100 * hit_top_cnt[0] / n_query)))
    print('Top{} Hit: {:.2f}%'.format(hit_top[1], float(100 * hit_top_cnt[1] / n_query)))
    print('Top{} Hit: {:.2f}%'.format(hit_top[2], float(100 * hit_top_cnt[2] / n_query)))
    print('Top{} Hit: {:.2f}%'.format(hit_top[3], float(100 * hit_top_cnt[3] / n_query)))
    
    r1 = float(hit_top_cnt[0] / n_query)
    r3 = float(hit_top_cnt[1] / n_query)
    r5 = float(hit_top_cnt[2] / n_query)
    r10 = float(hit_top_cnt[3] / n_query)
    if return_value==False:
        return
    else:        
        return r1, r3, r5, r10
    
    if not output_badcase:
        return

    # Record badcase
    dst_file = open(os.path.join(write_dir, "badcase_emb.txt"), "w")
    bbox_path = os.path.join(write_dir, "frame2bbox.json")
    with open(bbox_path, "r") as f:
        f2bbox = json.load(f)

    for q_name, q_pred in pred.items():
        is_hit = False
        top1_item, top1_score = q_pred[0]
        gt_item_lst = gt[q_name]
        if top1_item in gt_item_lst:  # top1 hit
            is_hit = True

        if is_hit:
            pass
        else:
            liveid, timestamp = q_name.split("_")
            frame_dir = os.path.join(root_dir, liveid, "frame")
            item_dir = os.path.join(root_dir, liveid, "shelf")
            frame_path = os.path.join(frame_dir, f"{timestamp}.jpg")
            try:
                frame_bbox = f2bbox[q_name]
            except Exception as e:
                print(e)
                frame_bbox = []
            top1_path = os.path.join(item_dir, f"{top1_item}_0.jpg")
            gt_path = os.path.join(item_dir, f"{gt_item_lst[0]}_0.jpg")
            dst_file.write(f"{frame_path}\t{frame_bbox}\t{top1_path}\t{top1_score}\t{gt_path}\n")
    dst_file.close()
    if return_value==False:
        return
    else:        
        return r1, r3, r5, r10

def topkPrediction(data_root, prefix="video_meanP"):
    #dst_root = os.path.join(data_root, prefix)
    dst_root = data_root
    clip_lst = [t for t in os.listdir(dst_root) if t.startswith("q2g_simmat_")]
    
    # import pdb; pdb.set_trace()
    # Merge 
    pred_path = os.path.join(dst_root, f"_{prefix}_q2topk.json")
    if not os.path.isfile(pred_path):
        rst = dict()
        for clip_name in tqdm(clip_lst):
            dstpath = os.path.join(dst_root, clip_name)
            
            try:
                with open(dstpath, "r") as f:
                    q2g = json.load(f)
            except json.decoder.JSONDecodeError:
                print(clip_name)
                # import pdb; pdb.set_trace()
                
                
            for _key, _val in q2g.items():
                _val = [[t[0].split("_")[0], t[1]] for t in _val]
                rst[_key] = _val
        print(f"Total number of clip: {len(rst)}")
       
        with open(pred_path, "w") as f:
            json.dump(rst, f, indent=1)

    # Calculate topk hit and acc
    with open(pred_path, "r") as f:
        pred = json.load(f)

    

    mode = prefix.split("_")[0]
    gt_name = "clip_to_gtitem.json" if mode=="video" else "frame_to_gtitem.json"
    with open(os.path.join("/home/inputs", gt_name), "r") as f:
        gt = json.load(f)
    
    gt = {k:v for k,v in gt.items() if k in pred}
    print(f"Total number of gt: {len(gt)}")
    
    calculate_acc(pred, gt, top_k=10, root_dir=None, write_dir=None, output_badcase=False)
    return

def topkPrediction_bycat(data_root, prefix="video_meanP"):
    #dst_root = os.path.join(data_root, prefix)
    dst_root = data_root
    # clip_lst = [t for t in os.listdir(dst_root) if t.startswith("q2g_simmat_")]

    pred_path = "./outputs_q2topk.json"
    
    

    # Calculate topk hit and acc
    with open(pred_path, "r") as f:
        pred = json.load(f)
    with open('/data/gt_dict.json', 'r') as f:
        cat1lev_2_clip_dict = json.load(f)
    
    mode = prefix.split("_")[0]
    gt_name = "clip_to_gtitem.json" if mode=="video" else "frame_to_gtitem.json"
    with open(os.path.join("./inputs", gt_name), "r") as f:
        gt = json.load(f)
    
    wf = open("topkPrediction_bycat.txt", 'w')
    for cat1name in cat1lev_2_clip_dict:
        clip_lst = cat1lev_2_clip_dict[cat1name]
        # 对于每个类目新搞一个gt
        
        cat_gt = {k:v for k,v in gt.items() if k in clip_lst}
        cat_pred = {k:v for k,v in pred.items() if k in clip_lst}
        # import pdb; pdb.set_trace()
        
        r1, r3, r5, r10 = calculate_acc(cat_pred, cat_gt, top_k=10, root_dir=None, write_dir=None, output_badcase=False, return_value=True)        
        print(f"{cat1name}\t{len(cat_gt)}\t{r1}\t{r3}\t{r5}\t{r10}")
        wf.write(f"{cat1name}\t{len(cat_gt)}\t{r1}\t{r3}\t{r5}\t{r10}\n")
        
    return

def clipWorker(subset, process_index, embedding_sim, recons_feat, add_text, ipd,
               sim_header, cross_num_hidden_layers, loose_type, ckpt_path, 
               q_dir, r_dir, out_root, mode, n_gpu, 
               num_frame=10, pretrained_clip_name="ViT-B/32", backbone_name="ViT-B/32", mo_fusion=False):
    
    # import pdb; pdb.set_trace()
    
    cat_text = False
    #cat_text = True
    model_name = ckpt_path.split("/")[1]
    sim_type = "emb" if embedding_sim else "decoder"
    
    with open("/home/eval_asr2emb_20086.json", "r") as f:
        asr2emb = json.load(f)
    with open("/home/eval_item2emb_6108.json", "r") as f:
        item2emb = json.load(f)
        
    with open("/home/_sf_testset_20086_frame2predbox_501656.json", "r") as f:
        f2box = json.load(f)

    BATCH_SIZE = 64  # Video batch size
    #BATCH_SIZE = 8  # Video batch size
    device_id = process_index % n_gpu
    device = torch.device(f"cuda:{device_id}")
#     model = init_model(sim_header, cross_num_hidden_layers, loose_type, device, ckpt_path=ckpt_path, training=False, recons_feat=recons_feat, embedding_sim=embedding_sim, add_text=add_text, )
    model = init_model(sim_header, cross_num_hidden_layers, loose_type, device, ckpt_path=ckpt_path, training=False, recons_feat=recons_feat, embedding_sim=embedding_sim, add_text=add_text, pretrained_clip_name=pretrained_clip_name, backbone_name=backbone_name, mo_fusion=mo_fusion)
    
    model = model.to(device)
    
    for clipid in tqdm(subset, desc=f"subprocess {process_index:02d}-th forward"):
        try:
            dpath = f"q2g_simmat_{clipid}.json"
            
            dpath = os.path.join(out_root, dpath)
            if os.path.exists(dpath):  
                # import pdb; pdb.set_trace()
                continue
            q2g = dict()
            if cat_text:  
                assert clipid in asr2emb, f"Clipid not in asr2emb: {clipid}"
                asr_emb = asr2emb[clipid]
                asr_emb = torch.tensor(asr_emb)
            elif add_text:  
                assert clipid in asr2emb, f"Clipid not in asr2emb: {clipid}"
                asr_emb = asr2emb[clipid]
                asr_emb = torch.tensor(asr_emb).to(device)
                asr_emb = asr_emb.unsqueeze(0)
            else:
                asr_emb = None
            liveid = clipid.split("_")[0]
            frame_path = os.path.join(q_dir, clipid)
            assert os.path.exists(frame_path), f"{frame_path} does not exists!"
    
            frame_lst = [t for t in os.listdir(frame_path) if t.endswith(".jpg")]
            frame_lst.sort()
    
            item_path = os.path.join(r_dir, liveid, "shelf")
            item_onshelf = [t for t in os.listdir(item_path) if t.endswith("_0.jpg")]  # only main image as gallery
            #item_onshelf = [t for t in item_onshelf if t.startswith("4253606665654")]  # for visualizing
            item_onshelf.sort()
    
            # Predict an item for video or frame?
            live_tictoc = list()
            if mode == "frame":
                for _index in range(len(frame_lst)):
                    if _index < (num_frame-1):
                        tic = 0
                        #toc = num_frame - 1  # including toc
                        toc = _index  # including toc
                    else:
                        tic = _index - num_frame + 1
                        toc = _index  # including toc
                    live_tictoc.append(frame_lst[tic:toc+1])
            else:
                assert mode=="video"
                if len(frame_lst) > num_frame:
                    sample_index = np.linspace(0, len(frame_lst), num=num_frame, endpoint=False, dtype=int)
                    live_tictoc.append([frame_lst[t] for t in sample_index])
                else:
                    live_tictoc.append(frame_lst)
                
            query2item = defaultdict(list)
            if mode == "frame":
                bs_photo = BATCH_SIZE
                batch_photo_lst = [live_tictoc[t:t+bs_photo] for t in range(0,len(live_tictoc),bs_photo)]
            else:
                batch_photo_lst = [live_tictoc]
                
            # Go through batch of live clips
            for _batch in batch_photo_lst:
                # Collecte batch photo
                photo_lst = list()
                photomask_lst = list()
                patchmask_lst = list()
                # Go through live clips
                for clip_framepath_lst in _batch:
                    frame_path_lst = [os.path.join(frame_path, t) for t in clip_framepath_lst]
                    video, video_mask = _get_rawvideo([frame_path_lst], ["random_clipid"], rawVideoExtractor)  # video_mask (1 max_frame)

                    # Opt1: Patch mask
                    if ipd:
                        box_lst = [f2box[t] for t in frame_path_lst]
                        box_lst = [[[t[0]*image_resolution,t[1]*image_resolution,t[2]*image_resolution,t[3]*image_resolution] for t in _box] for _box in box_lst]
                        patch_mask = [genPatchmask(t, GRIDS, iou_thresh=0.02) for t in box_lst]
                        if len(patch_mask)<video_mask.size(1):
                            num_pad = video_mask.size(1)-len(patch_mask)
                            patch_mask = patch_mask + [patch_mask[-1]]*num_pad
                        patch_mask = torch.tensor(patch_mask, dtype=torch.long)  # (num_frame 1+num_patch)
                        #assert patch_mask.size(0)==video_mask.size(1)
                        #patch_mask = patch_mask*video_mask.squeeze(0).unsqueeze(-1)

                    # Opt2: Patch mask
                    else:
                        patch_mask = video_mask.squeeze(0).unsqueeze(-1).repeat(1, 1+49)  # (num_frame 1+num_patch)

                    photo_lst.append(video)
                    photomask_lst.append(video_mask)
                    patchmask_lst.append(patch_mask)
                    
                video_batchsize = len(photo_lst)    
                video_data = torch.cat(photo_lst, dim=0).squeeze(2) # (bs max_frame 1 3 H W) to (bs max_frame 3 H W)
                video_data = video_data.view(-1, *video_data.shape[2:])  # (bs*max_frame 3 H W)
                video_mask = torch.cat(photomask_lst, dim=0)  # (bs max_frame)
                patch_mask = torch.stack(patchmask_lst, dim=0)  # (bs max_frame 1+num_patch)
                
                # Video inference
                with torch.no_grad():
                    video_data = video_data.to(device)
                    video_mask = video_mask.to(device)
                    patch_mask = patch_mask.to(device)

                    # Opt1
                    if ipd:
                        video_attn_mask = patch_mask.view(-1, patch_mask.size(-1)).contiguous()
                        video_attn_mask = video_attn_mask==0
                    # Opt2
                    else:
                        video_attn_mask = None
                    video_output, video_hidden = model.get_visual_output(video_data, video_batchsize, attn_mask=video_attn_mask, req_type="video")
                    video_hidden = video_hidden.view(video_batchsize, -1, *video_hidden.shape[-2:])

                # Collect batch of item on shelf
                #image_batchsize = video_batchsize if mode=="frame" else 128
                image_batchsize = 128 if embedding_sim else video_batchsize
                # image_batchsize = 32 if embedding_sim else video_batchsize
                
                item_batch_lst = [item_onshelf[t:t+image_batchsize] for t in range(0,len(item_onshelf),image_batchsize)]
                index_lst = list(range(len(item_batch_lst)))
    
                sim_lst = list()
                for item_lst in item_batch_lst:
                    item_path_lst = [os.path.join(item_path, t) for t in item_lst]
                    image_data = [rawImageExtractor.get_video_data([t])["video"] for t in item_path_lst]
                    image_data = torch.cat(image_data, dim=0)  # (bs 3 H W)
                    
                    # Add item title embedding
                    if cat_text:
                        __itemid_lst = [t.split("_")[0] for t in item_lst]
                        item_emb = [item2emb[t] for t in __itemid_lst]
                        item_emb = torch.tensor(item_emb)
                    elif add_text:
                        __itemid_lst = [t.split("_")[0] for t in item_lst]
                        item_emb = [item2emb[t] for t in __itemid_lst]
                        item_emb = torch.tensor(item_emb).to(device)
                    else:
                        item_emb = None
                    
                    # Image inference
                    with torch.no_grad():
                        image_data = image_data.to(device)
                        image_batchsize = image_data.size(0)
                        image_output, image_hidden = model.get_visual_output(image_data, image_batchsize, attn_mask=None, req_type="image")
                        
    
                        output = model.get_similarity_logits(
                            image_output, image_hidden, 
                            video_output, video_hidden, video_mask, patch_mask, 
                            asr_emb, item_emb,
                            shaped=True, loose_type=loose_type,  
                        )

                        if cat_text:
                            item_emb = F.normalize(item_emb, dim=1)
                            asr_emb = F.normalize(asr_emb.unsqueeze(0), dim=1)
                            i2v_simmat_txt = torch.matmul(item_emb, asr_emb.T)
                            i2v_simmat_txt = (i2v_simmat_txt + 1) / 2
                        
                        if embedding_sim:
                            i2v_simmat = output["contrastive_logit"]
                            if cat_text:
                                i2v_simmat += i2v_simmat_txt.to(i2v_simmat.device) * 0.3
                        else:
                            i2v_simmat = output["pairwise_logit"]
                        #v2i_simmat = i2v_simmat.T
                        v2i_simmat = i2v_simmat
                    sim_lst.append(v2i_simmat)
                    
                sim = torch.cat(sim_lst, dim=1)  # (bs_query, full_gallery)
                assert sim.size(1) == len(item_onshelf)
                
                if mode == "video":
                    assert video_batchsize==1 and sim.size(0)==1
                    q_name = "_".join(_batch[0][0].split("_")[:-1])
                    g_sim = [[k.strip(".jpg"), v.item()] for k,v in zip(item_onshelf, sim.squeeze(0))]
                    g_sim = [[k,v] for k,v in sorted(g_sim, key=lambda x:x[1], reverse=True)]
                    q2g.update({q_name: g_sim})
                else:
                    assert mode == "frame"
                    assert video_batchsize==sim.size(0)==len(_batch)
                    qname_lst = [t[-1].strip(".jpg") for t in _batch]
                    for q_name, _sim in zip(qname_lst, sim):
                        g_sim = [[k.strip(), v.item()] for k,v in zip(item_onshelf, _sim)]
                        g_sim = [[k,v] for k,v in sorted(g_sim, key=lambda x:x[1], reverse=True)]
                        q2g.update({q_name: g_sim})
                        
            with open(dpath, "w") as f:
                json.dump(q2g, f, indent=1)
        except Exception as e:
            # import pdb; pdb.set_trace()
            import traceback
            # traceback.print_exc()
            print(f"Meeting error, clip: {clipid}, {e}")
    return

def mpProcessClip(liveframe_root, gallery_root, out_root, mode, n_gpu,
         ckpt_path, sim_header, loose_type, cross_num_hidden_layers, embedding_sim, recons_feat, add_text, ipd, 
         num_frame, pretrained_clip_name, backbone_name, mo_fusion):
    clipid_lst = [t for t in os.listdir(liveframe_root)]
    print(f"Total number of clip: {len(clipid_lst)}")
    
    n_process = n_gpu*12 if mode=="video" else n_gpu*6

    n_per_process = math.ceil(len(clipid_lst)/n_process)
    input_split = [clipid_lst[i:i+n_per_process] for i in range(0,len(clipid_lst),n_per_process)]
    index_lst = list(range(len(input_split)))

    
    # Multiprocessing 
    manager = multiprocessing.Manager()
    cache_dict = manager.dict()
    
    jobs = []
    ctx = multiprocessing.get_context("spawn")
    for _index in index_lst:
        sub_batch = input_split[_index]
        p = ctx.Process(target=clipWorker, args=(sub_batch, _index, embedding_sim, recons_feat, add_text, ipd,
                                                 sim_header, cross_num_hidden_layers, loose_type, ckpt_path, 
                                                 liveframe_root, gallery_root, out_root, mode, n_gpu, num_frame, 
                                                 pretrained_clip_name, backbone_name, mo_fusion
                                                 ))
        p.start()
        jobs.append(p)
    for p in tqdm(jobs, desc="join subprocess"):
        p.join()
    return

if __name__ == "__main__":
    args = parsing_args()
    
    liveclip_root = "/lpr4m"
    liveframe_root = "/lpr4m/imgs"
    gallery_root = "/lpr4m/videosframe_testset"
    
    model_name = args.ckpt_path.split("/")[1]
    sim_type = "emb" if args.embedding_sim else "decoder"
    prefix = f"{args.mode}_{sim_type}_{model_name}"
    output_root = os.path.join("./outputs", prefix)
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        
    n_gpu = 8
    
    pretrained_clip_name = args.pretrained_clip_name
    backbone_name = args.backbone_name
    num_frame = 10
    print(pretrained_clip_name, backbone_name)
    

    mpProcessClip(liveframe_root, gallery_root, output_root, args.mode, n_gpu, args.ckpt_path, args.sim_header, args.loose_type, args.cross_num_hidden_layers, args.embedding_sim, args.recons_feat, args.add_text, args.ipd, num_frame, pretrained_clip_name, args.backbone_name, args.mo_fusion)
    topkPrediction(output_root, prefix)

   
