
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import logging
import random
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from modules.util_module import PreTrainedModel, AllGather, CrossEn, ReconsWeight
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.module_tmc import Transformer as TransformerTMC

from modules.module_mofusion import CrossAttentionLayer

from modules.module_clip import CLIP, convert_weights

from modules.graph_loss import GraphLoss
from modules.graph_loss import TripletLoss_rank

logger = logging.getLogger(__name__)
allgather = AllGather.apply
 
try:
    local_rank = torch.distributed.get_rank()
except:
    local_rank = 0

def show_log(info):
    logger.warning(info)

def _update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log("Set {}.{}: {}.".format(target_name, target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def update_attr(target_name, target_config, target_attr_name, default_value=None):
    if default_value is not None:
        setattr(target_config, target_attr_name, default_value)
        show_log("Set {}.{}: {}.".format(target_name, target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]


class CrossEnDiag(nn.Module):
    def __init__(self,):
        super(CrossEnDiag, self).__init__()

    def forward(self, sim_matrix):
        #logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = F.softmax(sim_matrix, dim=1)
        lable = (~torch.eye(sim_matrix.size(0), sim_matrix.size(1), dtype=bool)).to(device=logpt.device).float()
        sim_loss = (lable-logpt)**2
        return sim_loss

class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, max_words, max_frames, loose_type, linear_patch, sim_header, pretrained_clip_name, cross_num_hidden_layers, state_dict=None, cache_dir=None, type_vocab_size=2, training=True, recons_feat=False, embedding_sim=True, add_text=False, backbone_name='ViT-B/32', mo_fusion=False):

        if state_dict is None:
            state_dict = {}
        
        # import pdb; pdb.set_trace()
        # pretrained_clip_name = "ViT-B/32" if pretrained_clip_name is None else pretrained_clip_name
        if training == True:
            if pretrained_clip_name is not None:
            # Load ViT params
                clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
                logging.info(f"Successully load state_dict: {pretrained_clip_name}")
                for key, val in clip_state_dict.items():
                    if key in ["context_length", "input_resolution", "vocab_size"] and (not training):
                        continue
                    new_key = "clip." + key
                    if new_key not in state_dict:
                        state_dict[new_key] = val.clone()
        else:
            assert state_dict is not None
            clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
            
     
        # Cross model, input is video and image, outputs similarity.
        # Useless when loose_type is False
        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None)

        # Init CLIP4Clip, random initialed parameters
        model = cls(cross_config, clip_state_dict, max_words, max_frames, loose_type, linear_patch, sim_header, cross_num_hidden_layers, recons_feat, embedding_sim, add_text, backbone_name, mo_fusion)

        ## ===> Initialization trick [HARD CODE]
        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight

        if model.sim_header == 'tightTransf':
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            if contain_cross is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        # cut from beginning
                        if num_layer < cross_num_hidden_layers:
                            state_dict["cross."+key] = val.clone()
                            continue

        if model.sim_header == "seqLSTM" or model.sim_header == "seqTransf":
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
        ## <=== End of initialization trick
        
        # Loading pretrained parameters
        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, local_rank)
        
        # Print model
        with open("./model.txt", "w") as f:
            print(model, file=f)
        return model

class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, max_words, max_frames, loose_type,
            linear_patch, sim_header, cross_num_hidden_layers, recons_feat, embedding_sim, add_text, backbone_name, mo_fusion):
        super(CLIP4Clip, self).__init__(cross_config)
        self.ignore_video_index = -1

        assert max_words + max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log("Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and loose_type:
            self.loose_type = True
            show_log("Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        # vit = "visual.proj" in clip_state_dict
        if backbone_name == "ViT-B/32" or backbone_name == "SWING3D" or backbone_name == "VIVIT" or backbone_name == "TIMESFORMER":
            vit = True
        else:
            vit = False
        # import pdb; pdb.set_trace()
        # assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log("\t embed_dim: {}".format(embed_dim))
        show_log("\t image_resolution: {}".format(image_resolution))
        show_log("\t vision_layers: {}".format(vision_layers))
        show_log("\t vision_width: {}".format(vision_width))
        show_log("\t vision_patch_size: {}".format(vision_patch_size))
        show_log("\t context_length: {}".format(context_length))
        show_log("\t vocab_size: {}".format(vocab_size))
        show_log("\t transformer_width: {}".format(transformer_width))
        show_log("\t transformer_heads: {}".format(transformer_heads))
        show_log("\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = linear_patch
        show_log("\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log("\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
            linear_patch=self.linear_patch,
            vit=vit,
            backbone_name=backbone_name,
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        self.sim_header = sim_header
        show_log("\t sim_header: {}".format(self.sim_header))

        if self.sim_header == "tightTransf": assert self.loose_type is False

        cross_config.max_position_embeddings = context_length  # TODO BUG
        #cross_config.max_position_embeddings = (max_frames+1)*((image_resolution//vision_patch_size)**2+1) # MAX cross attention token
        if self.loose_type is False:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", default_value=cross_num_hidden_layers)
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
        if self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
            self.transformerClip = TransformerClip(width=transformer_width, layers=cross_num_hidden_layers,
                                                   heads=transformer_heads, )
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)
            
        self.type_position_embeddings = nn.Embedding(2, cross_config.hidden_size)
        
        self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
        self.transformerClip = TransformerTMC(width=transformer_width, layers=cross_num_hidden_layers,
                                                   heads=transformer_heads, )
        
        self.frame2t_attention = TransformerTMC(width=transformer_width, layers=1, heads=transformer_heads, )
        
        self.sigmoid = torch.nn.Sigmoid()
        self.trans_layernorm = torch.nn.LayerNorm(512)

        self.loss_fct = CrossEn()
        
        self.loss_graph = GraphLoss()
        self.loss_triplet = TripletLoss_rank()
        
        self.logit_scale = 100.
        
        
        self.recons_feat = recons_feat
        self.embedding_sim = embedding_sim
        if recons_feat:
            # Feature Reconstruction 
            self.recons_weights = ReconsWeight(in_dim=8)
            self.loss_recons = nn.MSELoss()
        
        self.temporal_proj = 'sigmoid_selfA'
        
        self.mo_fusion = mo_fusion
        if self.mo_fusion: 
            self.fusion_layer = CrossAttentionLayer(512, 8, 1024, 0.1)
            
        self.cat2emb = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512))
        
        self.cat2emb_trans = CrossAttentionLayer(512, 8, 1024, 0.1)
        
        self.add_text = add_text
        if self.add_text:
            emb_dim = 512
            self.fc_asr = nn.Linear(emb_dim, emb_dim)
            self.fc_title = nn.Linear(emb_dim, emb_dim)
        # Random init

        self.apply(self.init_weights)

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)  # bs_pair = bs
        sequence_hidden = self.clip.encode_text(input_ids, return_hidden=False).float()  # (bs d)
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1)) # (bs 1 d)

        return sequence_hidden

    def get_visual_output(self, video, bs_pair, video_frame=-1, return_hidden=True, attn_mask=None, req_type="none"):
        visual_cls, visual_hidden = self.clip.encode_image(video, video_frame=video_frame, return_hidden=return_hidden, attn_mask=attn_mask, req_type=req_type)  # (bs*max_frame 512)
        visual_cls, visual_hidden = visual_cls.float(), visual_hidden.float()
        # import pdb; pdb.set_trace()
        visual_cls = visual_cls.view(bs_pair, -1, visual_cls.size(-1))  
        return visual_cls, visual_hidden
    
    
    def get_text_output(self, text, bs_pair, return_hidden=True):
        text_cls, text_hidden = self.clip.encode_text(text, return_hidden=return_hidden)  # (bs*max_frame 512)
        text_cls, text_hidden = text_cls.float(), text_hidden.float()
        # import pdb; pdb.set_trace()
        text_cls = text_cls.view(bs_pair, -1, text_cls.size(-1))  
        return text_cls, text_hidden


    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        sequence_output = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)
        visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)

        return sequence_output, visual_output
    
    def temporal_difference_block(self, visual_output, video_mask):
        """
        Args:
            visual_output: embedding
            video_mask: video mask
        Returns:
            visual_output: frame representation
            frame_position_embeddings: position embedding
            type_embedding: type embedding
            temporal_video_mask: attention mask
        """

        seq_length = visual_output.size(1) # 12

        # obtain the positional embedding
        position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device) # 12
        position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1) # batch_size * 12
        frame_position_embeddings = self.frame_position_embeddings(position_ids) # batch_size * 12 * 512

        # obtain the type embedding to indicate the frame token and difference-enhanced token
        video_ids = torch.ones_like(position_ids)
        videoDif_ids = torch.zeros_like(position_ids)
        video_type_embedding = self.type_position_embeddings(video_ids)
        videoDif_type_embedding = self.type_position_embeddings(videoDif_ids)


        # batch size * 11 * 512
        dif_visual_output = visual_output[:, 1: seq_length, :] - visual_output[:, 0: seq_length - 1, :]
        if self.temporal_proj == 'sigmoid_mlp':
            # adopt sigmoid to transform into [-1, 1]
            dif_visual_output = 2 * self.sigmoid(self.trans_layernorm(dif_visual_output @ self.frame2t_projection)) - 1

        elif self.temporal_proj == 'sigmoid_selfA':
            # batch_size * 11 * 512
            dif_visual_output = dif_visual_output + frame_position_embeddings[:, 1:seq_length, :]
            trans_video_mask = video_mask[:,1:seq_length]
            # batch_size * 1* 11
            extend_trans_video_mask = (1.0 - trans_video_mask.unsqueeze(1)) * -1000000.0
            # batch_size * 11 * 11
            extend_trans_video_mask = extend_trans_video_mask.expand(-1, trans_video_mask.size(1), -1)

            dif_visual_output = dif_visual_output.permute(1, 0, 2)  # NLD -> LND # 11 * batch_size * 512
            dif_visual_output = self.frame2t_attention(dif_visual_output, extend_trans_video_mask)
            dif_visual_output = dif_visual_output.permute(1, 0, 2)  # LND -> NLD # batch_size * 11 * 512

            dif_visual_output = 2 * self.sigmoid(self.trans_layernorm(dif_visual_output)) - 1

        # batch size * (12+11) * 512
        visual_middle = torch.cat((visual_output, dif_visual_output), 1)
        # batch size * (12+12) * 512
        frame_position_embeddings_middle = torch.cat((frame_position_embeddings, frame_position_embeddings), 1)
        temporal_video_mask_middle = torch.cat((video_mask, video_mask), 1)
        type_embedding_middle = torch.cat((video_type_embedding, videoDif_type_embedding), 1)

        # obtain the correct index to insert difference-enhanced token
        seq1_indices = torch.arange(start=0, end=seq_length, step=1, dtype=torch.long)
        seq2_indices = torch.arange(start=seq_length, end=2 * seq_length - 1, step=1, dtype=torch.long)
        seq_indices = torch.stack((seq1_indices[0], seq2_indices[0]))
        for i in range(1, seq_length - 1):
            seq_indices = torch.cat((seq_indices, seq1_indices[i].view(1), seq2_indices[i].view(1)))
        seq_indices = torch.cat((seq_indices, seq1_indices[seq_length - 1].view(1))).to(visual_middle.device)

        # insert difference-enhanced token between every adjacent frame token
        visual_output = visual_middle.index_select(1, seq_indices)
        frame_position_embeddings = frame_position_embeddings_middle.index_select(1, seq_indices)
        temporal_video_mask = temporal_video_mask_middle.index_select(1, seq_indices)
        type_embedding = type_embedding_middle.index_select(1, seq_indices)

        return visual_output, frame_position_embeddings, type_embedding, temporal_video_mask

    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        try:
            visual_output = visual_output * video_mask_un
        except Exception as e:
            print(visual_output.device, video_mask_un.device, next(self.parameters()).device)
            print(e)
            visual_output = visual_output * video_mask_un.to(visual_output.device)
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out

    
    
    def _temporal_difference_block_tdb(self, visual_output, video_mask, sim_header="meanP"):
         
        # obtain the basic embedding
        visual_output_original = visual_output # batch_size * 12 * 512

        # difference-enhanced token obtained by TDB
        visual_output, frame_position_embeddings, type_embedding, temporal_video_mask = self.temporal_difference_block(
            visual_output, video_mask)

        # obtain the output of transformer
        visual_output = visual_output + frame_position_embeddings + type_embedding # batch_size * 12 * 512
        extended_video_mask = (1.0 - temporal_video_mask.unsqueeze(1)) * -1000000.0 # batch_size * 1* 12
        extended_video_mask = extended_video_mask.expand(-1, temporal_video_mask.size(1), -1) # batch_size * 12 * 12
        
        visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND # 12 * batch_size * 512
        visual_output = self.transformerClip(visual_output, extended_video_mask) #12 * batch_size * 512
        visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD # batch_size * 12 * 512

        # select the output of frame token for final video representation
        frame_position_id = torch.arange(start=0, end=visual_output.size()[1], step=2, dtype=torch.long,
                                         device=visual_output.device)
        visual_output = visual_output[:, frame_position_id, :]
        visual_output = visual_output + visual_output_original.to(visual_output.device)
        #visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        
        #visual_output_graph = visual_output
        
        
        return visual_output
    
    
    def _loose_similarity_text_tdb(self, item_output, asr_output, sequence_output, visual_output, video_mask, sim_header="meanP"):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
        #item_output, asr_output = item_output.contiguous(), asr_output.contiguous()

        if self.training:
            visual_output = allgather(visual_output)
            video_mask = allgather(video_mask)
            sequence_output = allgather(sequence_output)
            item_output = allgather(item_output)
            asr_output = allgather(asr_output)
            torch.distributed.barrier()
            
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        
        visual_output_graph = visual_output
        sequence_output_graph = sequence_output
        
        
        visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequence_output = sequence_output.squeeze(1)
        #sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        
        
        item_output = item_output / item_output.norm(dim=-1, keepdim=True)
        asr_output = asr_output / asr_output.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale if self.training else 1.
        #logit_scale_exp = self.logit_scale.exp()
        #logit_scale = logit_scale_exp if self.training else 1.
        
        retrieve_logits = logit_scale * torch.matmul(visual_output, sequence_output.t())
        txt_logits  = logit_scale * torch.matmul(asr_output, item_output.t())
        
        #total_logits = torch.cat([retrieve_logits, txt_logits], dim=1)
        #total_logits = retrieve_logits+txt_logits*0.5
        
        return retrieve_logits, txt_logits, visual_output, sequence_output, visual_output_graph, sequence_output_graph
    
   
    
    def _text_similarity(self, item_emb, asr_emb):
        item_emb, asr_emb = item_emb.contiguous(), asr_emb.contiguous()

        if self.training:
            item_emb = allgather(item_emb)
            asr_emb = allgather(asr_emb)
            torch.distributed.barrier()

        item_emb = item_emb / item_emb.norm(dim=-1, keepdim=True)
        asr_emb = asr_emb / asr_emb.norm(dim=-1, keepdim=True)

        #logit_scale = self.clip.logit_scale.exp()
        logit_scale = self.logit_scale if self.training else 1.
        #txt_logits = logit_scale * torch.matmul(item_emb, asr_emb.t())
        txt_logits = logit_scale * (item_emb - asr_emb)**2
        return txt_logits
       

    def _cross_similarity(self, sequence_output, visual_output, video_mask, batch_simmat=None):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
        if self.training:
            sequence_output = allgather(sequence_output)
            visual_output = allgather(visual_output)
            video_mask = allgather(video_mask)
            torch.distributed.barrier()

        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()
        #num_sample = 4 if self.training else b_visual # {b_visual, num_sample}, number of positive + negative 
        num_sample = 4 if self.training else b_visual
        batch_simmat = batch_simmat.T

        if batch_simmat is not None:
            assert batch_simmat.size(0) == b_text and batch_simmat.size(1)==b_visual
            _, indexes = batch_simmat.sort(descending=True)
            indexes = indexes.detach().cpu().numpy().tolist()
            hard_index_lst = list()
            for hard_i, _index in enumerate(indexes):
                hard_index_lst.append([t for t in _index if t!=hard_i][:(num_sample-1)])
              

        retrieve_logits_list = []
        retrieve_attn_weights_list = []

        #step_size = b_text      # set smaller to reduce memory cost
        step_size = 2 # set smaller to reduce memory cost
        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        # due to clip text branch retrun the last hidden
        attention_mask = torch.ones(b_text, s_text)\
            .to(device=video_mask.device, dtype=video_mask.dtype)

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, num_sample, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)  # (bs*num_sample s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, num_sample, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            if not self.training:
                # Opt1: Use all video, num_sample=b_visual
                visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
                visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
                video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
                video_mask_r = video_mask_r.view(-1, s_visual)
            else:
                # Opt2: Random or Hard Negative sample
                _tic = i * step_size 
                _toc = (i+1) * step_size
                _toc = _toc if _toc < sequence_output.size(0) else sequence_output.size(0)
                visual_output_r = list()
                video_mask_r = list()
                for j in range(_tic, _toc):

                    # Hard sampling
                    index_neg = hard_index_lst[j]

                    pos = visual_output[j:j+1]  # (1 S d)
                    neg = visual_output[index_neg]  # (num_sample-1 S d)
                    pair = torch.cat([pos, neg], dim=0)  # (num_sample S d)
                    visual_output_r.append(pair)

                    pos_mask = video_mask[j:j+1]
                    neg_mask = video_mask[index_neg]  # (num_sample-1 S)
                    pair_mask = torch.cat([pos_mask, neg_mask], dim=0)
                    video_mask_r.append(pair_mask)

                visual_output_r = torch.stack(visual_output_r, dim=0)  # (bs num_sample S d)
                visual_output_r = visual_output_r.view(-1, s_visual, h_visual)  # (bs*num_sample S d), view=concat all rows
                video_mask_r = torch.stack(video_mask_r, dim=0)
                video_mask_r = video_mask_r.view(-1, s_visual)


            pooled_output, attn_weights = self.cross(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, num_sample)
            attn_weights = attn_weights.view(step_truth, num_sample, *attn_weights.shape[1:])[:,0, ...]  # only retain positive attn_weights

            retrieve_logits_list.append(retrieve_logits_row)
            retrieve_attn_weights_list.append(attn_weights)

        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        retrieve_weights = torch.cat(retrieve_attn_weights_list, dim=0)
        
        return retrieve_logits, retrieve_weights

    def get_similarity_logits(self, 
            sequence_output, sequence_hidden, 
            visual_output, visual_hidden, video_mask, patch_mask, 
            asr_emb=None, item_emb=None,
            shaped=False, loose_type=False):
        # sequence_output (bs 1 d)
        # sequence_hidden (bs 1+num_patch d)
        # visual_output (bs num_frame d)
        # visual_hidden to (bs num_frame 1+num_patch d)
        # video_mask (bs num_frame)
        # patch_mask (bs num_frame 1+num_patch)
        # asr_emb (bs d)
        # item_emb (bs d)
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        output = dict()
        # contrastive learning logits (bs bs)
        #cl_logits = self._loose_similarity(sequence_output, visual_output, video_mask, sim_header=self.sim_header)
        item_emb = self.fc_title(item_emb.to(sequence_output.dtype))  # (bs d) to (bs 1 d)
        asr_emb = self.fc_asr(asr_emb.to(visual_output.dtype))  # (bs d) to (bs 1 d)
        
        visual_output_tdb = self._temporal_difference_block_tdb(visual_output, video_mask, sim_header=self.sim_header)
        retrieve_logits, txt_logits, visual_output_emb, sequence_output_emb, visual_output_graph, sequence_output_graph = self._loose_similarity_text_tdb(item_emb, asr_emb, sequence_output, visual_output_tdb, video_mask, sim_header=self.sim_header)
        #
        output.update({"retrieve_logit": retrieve_logits})
        output.update({"text_logit": txt_logits})
        
        cl_logits = retrieve_logits + txt_logits.to(retrieve_logits.device) * 0.5

        output.update({"contrastive_logit": cl_logits})
        
        output.update({"visual_output_emb": visual_output_emb})
        output.update({"sequence_output_emb": sequence_output_emb})
        
        output.update({"visual_output_graph": visual_output_graph})
        output.update({"sequence_output_graph": sequence_output_graph})
        
        
        # import pdb; pdb.set_trace()
        if not loose_type: 
            assert self.sim_header in ["tightTransf"]
            _tic = time.time()
            visual_hidden = visual_hidden.view(visual_hidden.size(0), -1, visual_hidden.size(-1))  # (bs num_frame 1+num_patch d) to (bs num_frame*1+num_patch d)
            patch_mask = patch_mask.view(patch_mask.size(0), -1)  # (bs max_frame 1+num_patch) to (bs max_frame*1+num_patch)
            # Concat text modality
            if self.add_text:
                item_emb = item_emb.unsqueeze(1)
                asr_emb = asr_emb.unsqueeze(1)  # (bs d) to (bs 1 d)
                
                item_emb = item_emb / item_emb.norm(dim=-1, keepdim=True)
                asr_emb = asr_emb / asr_emb.norm(dim=-1, keepdim=True)
                
                sequence_hidden = sequence_hidden / sequence_hidden.norm(dim=-1, keepdim=True)
                visual_hidden = visual_hidden / visual_hidden.norm(dim=-1, keepdim=True)
                
                if self.mo_fusion:
                    # => [32, 50, 512]
                    sequence_hidden = self.fusion_layer(sequence_hidden, item_emb.to(sequence_hidden.dtype))
                    # => [32, 500, 512]
                    visual_hidden = self.fusion_layer(visual_hidden, asr_emb.to(visual_hidden.dtype)) # bs max_frame*(1+num_patch)+1 d
                    
                else:
                    sequence_hidden = torch.cat([sequence_hidden, item_emb], dim=1)  # bs (1+num_patch)+1 d
                    visual_hidden = torch.cat([visual_hidden, asr_emb], dim=1)  # bs max_frame*(1+num_patch)+1 d
                    text_mask = torch.ones([patch_mask.size(0), 1], dtype=patch_mask.dtype, device=patch_mask.device)
                    patch_mask = torch.cat([patch_mask, text_mask], dim=1)  # (bs max_frame*1+num_patch) to bs max_frame*(1+num_patch)+1
            
            mt_logits, attn_weights = self._cross_similarity(sequence_hidden, visual_hidden, patch_mask, batch_simmat=cl_logits)
            logit_scale = self.logit_scale if self.training else 1. 
            mt_logits = logit_scale * mt_logits
            _dur = time.time() - _tic
            #print(f"Cross similarity takes time: {_dur:.3f} sec")
            
            if self.add_text:
                attn_weights = attn_weights[:,:,:-1,:-1]  # (bs num_head N+1 M+1) to (bs num_head N M)
            output.update({"pairwise_logit": mt_logits, "attn_weights": attn_weights})
            

        return output
    
    
    def forward(self, video, video_mask, image, patch_mask=None, asr_emb=None, item_emb=None):
        """
        image: (bs 3 H W)
        video_mask: (bs 1 max_frame)
        patch_mask: (bs max_frame 1+num_patch)
        asr_emb: (bs d)
        item_emb: (bs d)
        """
        #input_ids = input_ids.view(-1, input_ids.shape[-1])  # (bs 1 max_words) to (bs max_words)
        #token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])  # same size as input_ids, all zeros
        #attention_mask = attention_mask.view(-1, attention_mask.shape[-1])  # same size as input_ids, 1 and 0
        video_mask = video_mask.view(-1, video_mask.shape[-1]) # (bs 1 max_frame) to (bs max_frame)
        bs_pair = video_mask.size(0)

        video = torch.as_tensor(video).float()  # (bs 1 max_frame 1 3 H W)
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts
        
        sequence_output, sequence_hidden = self.get_visual_output(image, bs_pair, attn_mask=None, req_type='image')
        
        if patch_mask is not None: 
            video_attn_mask = patch_mask.view(-1, patch_mask.size(-1)).contiguous()
            video_attn_mask = video_attn_mask==0
            #video_attn_mask = (1 - video_attn_mask).to(torch.uint8)
        else:
            video_attn_mask = patch_mask
        # visual_output (bs num_frame d)
        # visual_hidden (bs*num_frame 1+num_patch d) to (bs num_frame 1+num_patch d)

        visual_output, visual_hidden = self.get_visual_output(video, bs_pair, attn_mask=video_attn_mask, req_type='video')
        visual_hidden = visual_hidden.view(b, -1, *visual_hidden.shape[-2:])
        
        #asr_output, asr_hidden = self.get_text_output(asr_emb, bs_pair)
        #item_output, item_hidden = self.get_text_output(item_emb, bs_pair)
        asr_output=asr_emb
        item_output = item_emb
        
        if patch_mask is None:
            patch_mask = video_mask.unsqueeze(-1).repeat(1, 1, sequence_hidden.size(1))  # (bs num_frame 1+num_patch)

        if self.training:
            loss_dict = dict()
            output = self.get_similarity_logits(
                    sequence_output, sequence_hidden, 
                    visual_output, visual_hidden, video_mask, patch_mask,
                    asr_emb, item_emb,
                    shaped=True, loose_type=self.loose_type
                    )

            # Contrastive loss
            sim_matrix = output["retrieve_logit"]  # (bs bs)
            txt_matrix = output["text_logit"]  # (bs bs)
            
            visual_output_emb = output["visual_output_emb"]  # (bs 512)
            sequence_output_emb = output["sequence_output_emb"]  # (bs 512)
            
            visual_output_graph = output["visual_output_graph"]  # (bs 10 512)
            sequence_output_graph = output["sequence_output_graph"]  # (bs 1 512)
            
            txt_loss1 = self.loss_fct(txt_matrix)
            txt_loss2 = self.loss_fct(txt_matrix)
            txt_loss = (txt_loss1 + txt_loss2) / 2
            
            
            loss_dict.update({"txt_loss": txt_loss})

            
            base_loss, reg_loss, gnn_loss = self.loss_graph(visual_output_emb, sequence_output_emb, visual_output_graph, sequence_output_graph)
            
            loss_dict.update({"base_loss": base_loss})
            loss_dict.update({"reg_loss": reg_loss})
            loss_dict.update({"gnn_loss": gnn_loss})
            
            
            if not self.loose_type:
                # Pairwise loss
                sim_matrix = output["pairwise_logit"]  # (bs 2)
                pairwise_loss = self.loss_fct(sim_matrix)
                loss_dict.update({"pairwise_loss": pairwise_loss * 1.0 })
                
                # Reconstruction loss
                if self.recons_feat:
                    sequence_hidden = allgather(sequence_hidden)
                    visual_hidden = allgather(visual_hidden)
                    video_mask = allgather(video_mask)
                    if patch_mask is not None:
                        patch_mask = allgather(patch_mask)
                    torch.distributed.barrier()
                    if patch_mask is not None:
                        visual_hidden = visual_hidden * patch_mask.unsqueeze(-1)  # ((bs num_frame 1+num_patch d))
                    visual_hidden = visual_hidden[:, :, 1:, :].contiguous()  # ignore CLS token
                    y = visual_hidden.view(visual_hidden.size(0), -1, visual_hidden.size(-1))  # (bs num_frame 1+num_patch d) to (bs num_frame*1+num_patch d)
                    
                    num_frame = visual_hidden.size(1)
                    attn_w = output["attn_weights"]  # (bs num_heads N M)
                    attn_w = attn_w.view(attn_w.size(0), attn_w.size(1), attn_w.size(2), num_frame, -1) # (bs num_heads N num_frame N)
                    attn_w = attn_w[:, :, 1:, :, 1:].contiguous()
                    attn_w = attn_w.view(attn_w.size(0), attn_w.size(1), attn_w.size(2), -1)  # ignore CLS token
                    attn_w = self.recons_weights(attn_w).squeeze(1)  # (bs 1 N M) to (bs N M)

                    lambda_x = torch.bmm(attn_w, y.detach())
                    #lambda_x = torch.bmm(attn_w, y)
                    sequence_hidden = sequence_hidden[:, 1:, :].detach()  # ignore CLS token

                    diff_loss = self.loss_recons(lambda_x, sequence_hidden)
                    loss_dict.update({"diff_loss": diff_loss * 0.05})

                    
            return loss_dict
        else:
            return None
        
if __name__ == "__main__":
    PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE', Path.home() / '.pytorch_pretrained_bert'))
    cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')

    pretrained_clip_name = "ViT-B/32"
    model = CLIP4Clip.from_pretrained(cross_model_name="cross-base", 
            max_words=32, max_frames=12, loose_type=True, linear_patch="2d", sim_header="meanP", 
            pretrained_clip_name=pretrained_clip_name, cross_num_hidden_layers=4,
            state_dict=None, cache_dir=cache_dir, type_vocab_size=2)
    import pdb; pdb.set_trace()
    print(model)

