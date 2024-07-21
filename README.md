### Spatiotemporal Graph Learning for Multi-modal Livestreaming Product Retrieval


# Training Datasets
1. Prepare the Training Datasets

# MovingFashion Dataset
The MovingFashion Dataset can be download from https://github.com/humaticslab/seam-match-rcnn

# LPR4M Dataset 
The LPR4M Dataset can be download from https://github.com/adxcreative/RICE

# Dependency
2. Environment Configuration
* Create a virtual environment and install Pytorch.

python>=3.6
pytorchtorch>=1.7.1
torchvision
pytorchvideo
numpy
Pillow
timm
ftfy
tqdm
...

# Training
3. Run our Code
*  You can run the following command to train our SGMN

python -u -m torch.distributed.launch --nproc_per_node=8 train.py --do_train --num_thread_reader=0 --epochs=10 --batch_size=256 --n_display=1 --fp16 --features_path [data_path] --output_dir  [output_path] --lr 3e-4 --max_words 32 --max_frames 10 --batch_size_val 128 --datatype [dataset] --expand_msrvtt_sentences --feature_framerate 1 --coef_lr 1e-3 --freeze_layer_num 0 --slice_framepos 2 --linear_patch 2d --pretrained_clip_name ViT-B/32 --backbone_name ViT-B/32 --sim_header tightTransf --cross_num_hidden_layers 2 --add_text --embedding_sim --mo_fusion

# Evaluation
4. Run our Code
*  You can run the following command to evaluate our SGMN on LPR4M or on MF

python eval_mf.py --sim_header tightTransf --add_text --embedding_sim --mo_fusion --pretrained_clip_name ViT-B/32 --backbone_name ViT-B/32  --ckpt_path [model_path]



