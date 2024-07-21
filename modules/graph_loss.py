import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from modules.mca import AdjacencyModel

def pos_neg_mask(labels):

    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) 
    neg_mask = labels.unsqueeze(0) != labels.unsqueeze(1)

    return pos_mask, neg_mask


def pos_neg_mask_xy(labels_col, labels_row):

    pos_mask = (labels_row.unsqueeze(0) == labels_col.unsqueeze(1)) 
    neg_mask = (labels_row.unsqueeze(0) != labels_col.unsqueeze(1))

    return pos_mask, neg_mask

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=0.2, max_violation=False):
        super(ContrastiveLoss, self).__init__()

        self.margin = margin
        self.max_violation = max_violation
        self.mask_repeat = 1

        self.false_hard = []

    def max_violation_on(self):
        self.max_violation = True
        # print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        # print('Use VSE0 objective.')

    def forward(self, im, s, img_ids=None):

        # compute image-sentence score matrix
        scores = get_sim(im, s)
        
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval, i->t
        cost_s = (self.margin + scores - d1).clamp(min=0)

        # compare every diagonal score to scores in its row
        # image retrieval t->i
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        if not self.mask_repeat:
            mask = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)
        else:
            img_ids = img_ids.cuda()
            mask = (img_ids.unsqueeze(1) == img_ids.unsqueeze(0))

        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s, idx_s = cost_s.max(1)
            cost_im, idx_im = cost_im.max(0)

        loss = cost_s.sum() + cost_im.sum()

        return loss


def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities


# Triplet loss + DistanceWeight Miner
# Sampling Matters in Deep Embedding Learning, ICCV, 2017
# more information refer to https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#distanceweightedminer
class TripletLoss(nn.Module):

    def __init__(self, margin=0.2, ):
        super().__init__()

        self.margin = margin
        
        self.cut_off = 0.5
        self.d = 512


        self.nonzero_loss_cutoff = 1.7
        
    def forward(self, im, s, img_ids):

        sim_mat = get_sim(im, s)
        img_ids = img_ids.cuda()

        if im.size(0) == s.size(0):
            pos_mask, neg_mask = pos_neg_mask(img_ids)
        else:
            pos_mask, neg_mask = pos_neg_mask_xy(torch.unique(img_ids), img_ids)

        loss_im = self.loss_forward(sim_mat, pos_mask, neg_mask)
        loss_s = self.loss_forward(sim_mat.t(), pos_mask.t(), neg_mask.t())

        loss = loss_im + loss_s

        return loss        

    def loss_forward(self, sim_mat, pos_mask, neg_mask): 

        pos_pair_idx = pos_mask.nonzero(as_tuple=False)
        anchor_idx = pos_pair_idx[:, 0]
        pos_idx = pos_pair_idx[:, 1]

        dist = (2 - 2 * sim_mat).sqrt()
        dist = dist.clamp(min=self.cut_off)

        log_weight = (2.0 - self.d) * dist.log() - ((self.d - 3.0) / 2.0) * (1.0 - 0.25 * (dist * dist)).log()
        inf_or_nan = torch.isinf(log_weight) | torch.isnan(log_weight)

        log_weight = log_weight * neg_mask  
        log_weight[inf_or_nan] = 0.      

        weight = (log_weight - log_weight.max(dim=1, keepdim=True)[0]).exp()
        weight = weight * (neg_mask * (dist < self.nonzero_loss_cutoff)).float() 
     
        weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-20)
        weight = weight[anchor_idx]

        # maybe not exist
        try:
            neg_idx = torch.multinomial(weight, 1).squeeze(1)   
        except Exception:
            return torch.zeros([], requires_grad=True, device=sim_mat.device) 


        s_ap = sim_mat[anchor_idx, pos_idx]
        s_an = sim_mat[anchor_idx, neg_idx]  

        loss = F.relu(self.margin + s_an - s_ap) 
        loss = loss.sum() 

        return loss


    
class dual_softmax_loss(nn.Module):
    
    def __init__(self,):
        super(dual_softmax_loss, self).__init__()
        
    def forward(self, sim_matrix, temp=1000):
        sim_matrix = sim_matrix * F.softmax(sim_matrix/temp, dim=0)*len(sim_matrix) #With an appropriate temperature parameter, the model achieves higher performance
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        loss = -logpt
        return loss
    
    
class TripletLoss_rank(nn.Module):

    def __init__(self, margin=0.2, ):
        super().__init__()

        self.margin = margin
        
        self.cut_off = 0.5
        self.d = 512


        self.nonzero_loss_cutoff = 1.7
        
    def forward(self, sim_mat):

        sim_mat = sim_mat
        img_ids = torch.arange(0, sim_mat.size(0))
        img_ids = img_ids.cuda()

        if sim_mat.size(0) == sim_mat.size(0):
            pos_mask, neg_mask = pos_neg_mask(img_ids)
        else:
            pos_mask, neg_mask = pos_neg_mask_xy(torch.unique(img_ids), img_ids)

        loss_im = self.loss_forward(sim_mat, pos_mask, neg_mask)
        loss_s = self.loss_forward(sim_mat.t(), pos_mask.t(), neg_mask.t())

        loss = loss_im + loss_s

        return loss        

    def loss_forward(self, sim_mat, pos_mask, neg_mask): 

        pos_pair_idx = pos_mask.nonzero(as_tuple=False)
        anchor_idx = pos_pair_idx[:, 0]
        pos_idx = pos_pair_idx[:, 1]

        dist = (2 - 2 * sim_mat).sqrt()
        dist = dist.clamp(min=self.cut_off)

        log_weight = (2.0 - self.d) * dist.log() - ((self.d - 3.0) / 2.0) * (1.0 - 0.25 * (dist * dist)).log()
        inf_or_nan = torch.isinf(log_weight) | torch.isnan(log_weight)

        log_weight = log_weight * neg_mask  
        log_weight[inf_or_nan] = 0.      

        weight = (log_weight - log_weight.max(dim=1, keepdim=True)[0]).exp()
        weight = weight * (neg_mask * (dist < self.nonzero_loss_cutoff)).float() 
     
        weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-20)
        weight = weight[anchor_idx]

        # maybe not exist
        try:
            neg_idx = torch.multinomial(weight, 1).squeeze(1)   
        except Exception:
            return torch.zeros([], requires_grad=True, device=sim_mat.device) 


        s_ap = sim_mat[anchor_idx, pos_idx]
        s_an = sim_mat[anchor_idx, neg_idx]  

        loss = F.relu(self.margin + s_an - s_ap) 
        loss = loss.sum() 

        return loss
    
    
def loss_select(loss_type='vse'):

    if loss_type == 'vse':
        # the default loss
        criterion = ContrastiveLoss(margin=0.2, max_violation=False)
    elif loss_type == 'trip':
        # Triplet loss with the distance-weight sampling
        criterion = TripletLoss()
    else:
        raise ValueError('Invalid loss {}'.format(loss_type))
    
    return criterion

# instance-level relation modeling
class GraphLoss(torch.nn.Module):
    def __init__(self, ):
        super().__init__()

        self.iter_count = 0
        self.embed_size = 512

        # Initialize the dml objective function for embeddings learning.
        self.base_loss = loss_select(loss_type='trip')
        self.gnn_loss = loss_select(loss_type='trip')

        # the fusion interaction mechanism
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_size, nhead=16, 
                                                    dim_feedforward=self.embed_size, dropout=0.1)          
        self.gnn = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # construct the cross-embedding graph
        self.adj_model = AdjacencyModel(hidden_size=self.embed_size, threshold=0.5, topk=10, detach=True)
            
# img_emb [256,512] cap_emb [256 512] img_emb_visual [256 10 512], cap_emb_sequence [256 1 512]
    def forward(self, img_emb, cap_emb, img_emb_visual, cap_emb_sequence):
        
        # get latent features and embeddings
        # include the pre-pooling and after-pooling features    
        img_ids = torch.arange(0, img_emb.size(0))
        
        img_len = torch.full(size=(img_emb.size(0), 1),fill_value=10)
        cap_len = torch.full(size=(img_emb.size(0), 1),fill_value=10)
        
        img_emb_notnorm = img_emb
        cap_emb_notnorm = cap_emb
        
        img_emb_pre_pool = img_emb_visual
        cap_emb_pre_pool = cap_emb_sequence.repeat(1, 10, 1)
        
        

        bs = img_emb_notnorm.shape[0]
        assert img_emb_notnorm.shape[0] == cap_emb_notnorm.shape[0]

        num_loss = 0

        # basic matching loss
        base_loss = self.base_loss(img_emb, cap_emb, img_ids)
        num_loss += 1

        if self.iter_count >= 0:

            # get the connection relation and the relevance relation
            mask_weight = 1.0
            batch_c, batch_r, reg_loss = self.adj_model(img_emb, cap_emb, 
                                                        img_regions=img_emb_pre_pool, 
                                                        cap_words=cap_emb_pre_pool, 
                                                        img_len=img_len,
                                                        cap_len=cap_len,)
            # connection relation
            connect_mask = torch.cat((torch.cat((batch_c['i2i'], batch_c['i2t']), dim=1), 
                                    torch.cat((batch_c['t2i'], batch_c['t2t']), dim=1)), dim=0)

            # relevance relation
            relation_mask = torch.cat((torch.cat((batch_r['i2i'], batch_r['i2t']), dim=1), 
                                    torch.cat((batch_r['t2i'], batch_r['t2t']), dim=1)), dim=0)

            mask = mask_weight * relation_mask.masked_fill_(~connect_mask, float('-inf'))                                                                

            # concat mbeddings, batch as the dim=1
            if 1:
                all_embs = torch.cat((img_emb, cap_emb), dim=0)
            else:
                all_embs = torch.cat((img_emb_notnorm, cap_emb_notnorm), dim=0)

            # get the instance-level relation modeling 
            all_embs_gnn = self.gnn(all_embs.unsqueeze(1), mask).squeeze(1)

            # get the enhanced embeddings
            img_emb_gnn, cap_emb_gnn = torch.split(all_embs_gnn, bs, dim=0)              

            # L2 normalization for the relation-enhanced embeddings
            img_emb_gnn = F.normalize(img_emb_gnn)
            cap_emb_gnn = F.normalize(cap_emb_gnn)
            
            # compute loss
            if 1:
                gnn_loss1 = self.gnn_loss(img_emb, cap_emb_gnn, img_ids)
                gnn_loss2 = self.gnn_loss(img_emb_gnn, cap_emb, img_ids)
                num_loss += 3
            else:
                gnn_loss1 = 0.
                gnn_loss2 = 0.
                num_loss += 1

            gnn_loss3 = self.gnn_loss(img_emb_gnn, cap_emb_gnn, img_ids)
            gnn_loss = gnn_loss1 + gnn_loss2 + gnn_loss3               
                
        else:
            gnn_loss = 0.
            reg_loss = 0.

        #loss = (base_loss + gnn_loss) 
        
        reg_loss = 10 * reg_loss

        self.iter_count += 1

        return base_loss, reg_loss, gnn_loss



if __name__ == '__main__':

    pass

