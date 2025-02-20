import math
import copy

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from utils.misc import inverse_sigmoid
from .ops.modules import MSDeformAttn

class TextGuidedAdapter(nn.Module):
    def __init__(self, d_model=256,
                 dropout=0.1,
                 n_levels=4,
                 n_heads=8,
                 n_points=4,):
        super().__init__()

        # img2text: Cross attention
        self.img2text_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.adapt_proj = MLP(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1)
        self.orig_proj = MLP(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1)

        self.tf_pow = 2.0
        self.tf_scale = nn.Parameter(torch.Tensor([1.0]))
        self.tf_sigma = nn.Parameter(torch.Tensor([0.5]))

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, return_indices=False, ceil_mode=False)

        # img2img: Multi-Scale Deformable Attention
        self.img2img_msdeform_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

        self.norm_text_cond_img = nn.LayerNorm(d_model)
        self.norm_img = nn.LayerNorm(d_model)

        # depth2text: Cross attention
        self.depth2textcross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        # depth2depth: Cross attention
        self.depth2depth_attn =  nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.norm_text_cond_depth = nn.LayerNorm(d_model)
        self.norm_depth = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, img_feat_src, masks, img_pos_embeds,
                reference_points, spatial_shapes,
                level_start_index,
                src_valid_ratios,
                word_feat, word_key_padding_mask,
                depth_pos_embed, mask_depth,im_name, instanceID, ann_id,
                word_pos=None):
        orig_multiscale_img_feat = img_feat_src
        orig_multiscale_masks = masks
        orig_multiscale_img_pos_embeds = img_pos_embeds

        # split four level multi-scale img_feat/masks/img_pos_embeds
        bs, sum, dim = img_feat_src.shape
        img_feat_src_list = img_feat_src.split([H_ * W_ for H_, W_ in spatial_shapes], dim=1) ### 按照原始拆分回原始的四层特征
        masks_list = masks.split([H_ * W_ for H_, W_ in spatial_shapes], dim=1)
        img_pos_embeds_list = img_pos_embeds.split([H_ * W_ for H_, W_ in spatial_shapes], dim=1)

        # For second level img_feat/masks/img_pos_embeds to compute score ### 取出第二层特征
        img_feat_src = img_feat_src_list[1]  ## bs 1920 256
        masks = masks_list[1]
        img_pos_embeds = img_pos_embeds_list[1]

                    
        q = self.with_pos_embed(img_feat_src, img_pos_embeds)
        k = self.with_pos_embed(word_feat, word_pos)
        imgfeat_adapt = self.img2text_attn(query=q.transpose(0, 1),
                                  key=k.transpose(0, 1),
                                  value=word_feat.transpose(0, 1),
                                  key_padding_mask=word_key_padding_mask)[0].transpose(0, 1)
        ##### 利用第二层特征去查询文本信息
        imgfeat_adapt_embed = self.adapt_proj(imgfeat_adapt)  # [bs, 1920, 256]
        imgfeat_orig_embed = self.orig_proj(img_feat_src)

        verify_score = (F.normalize(imgfeat_orig_embed, p=2, dim=-1) *
                        F.normalize(imgfeat_adapt_embed, p=2, dim=-1)).sum(dim=-1, keepdim=True) ### 求视觉特征和文本之间的相关性
        verify_score = self.tf_scale * \
                       torch.exp( - (1 - verify_score).pow(self.tf_pow) \
                        / (2 * self.tf_sigma**2))   # [12, 1920, 1]  ### 使用高斯函数进行建模 每一个特征之间的相关性

        # For score of map-16 to upsample and downsample
        verify_score_16 = verify_score.reshape(bs, spatial_shapes[1][0], spatial_shapes[1][1], 1).squeeze(-1)   ### w/16  正好是第二层特征
        verify_score_8 = self.upsample(verify_score_16.unsqueeze(1)).squeeze(1)  #### 上采样
        verify_score_32 = self.downsample(verify_score_16)  #### 下采样
        verify_score_64 = self.downsample(verify_score_32)  #### 下采样
        verify_score_list = [verify_score_8.flatten(1), verify_score_16.flatten(1),verify_score_32.flatten(1), verify_score_64.flatten(1)]
        verify_score = torch.cat(verify_score_list, dim=1).unsqueeze(-1)   ### 合并所有的相关性分数

        q = k = img_feat_src + imgfeat_adapt   # second image feature

        # concat multi-scale image feature
        src = torch.cat([img_feat_src_list[0],q ,img_feat_src_list[2],img_feat_src_list[3]], 1)

        reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
        #### 视觉引导的文本特征后续的MSDA
        text_cond_img_ctx = self.img2img_msdeform_attn(
                            self.with_pos_embed(src, orig_multiscale_img_pos_embeds),
                            reference_points_input, orig_multiscale_img_feat, spatial_shapes,
                                  level_start_index, orig_multiscale_masks)

        # adapted image feature
        adapt_img_feat = (self.norm_img(orig_multiscale_img_feat) + self.norm_text_cond_img(text_cond_img_ctx)) * verify_score
        ### 视觉引导的特征 乘以所有的score
        # text-guided depth encoder  
        depthfeat_adapt = self.depth2textcross_attn(  ### 来一个交叉注意力
            query=depth_pos_embed,
            key=self.with_pos_embed(word_feat, word_pos).transpose(0, 1),
            value=word_feat.transpose(0, 1), key_padding_mask=word_key_padding_mask)[0]

        q = k = depth_pos_embed + depthfeat_adapt   # depth feature of second image
        text_cond_depth = self.depth2depth_attn(query=q, key=k, value=depth_pos_embed, key_padding_mask=mask_depth)[0]### 再来个自注意力

        # adapted depth feature
        adapt_depth_feat = (self.norm_depth(depth_pos_embed.transpose(0, 1)) + self.norm_text_cond_depth(text_cond_depth.transpose(0, 1))) * verify_score_16.flatten(1).unsqueeze(-1)  ####
        adapt_depth_feat = adapt_depth_feat.transpose(0, 1)
        return torch.cat([orig_multiscale_img_feat, adapt_img_feat], dim=-1), torch.cat([depth_pos_embed, adapt_depth_feat], dim=-1)

class GatedFusion(nn.Module):
    def __init__(self, C):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * C, C),
            nn.Sigmoid()
        )
        self.regional_transform = nn.Linear(C, C)
        self.global_transform = nn.Linear(C, C)

    def forward(self, global_feat, regional_feat):
        bs, n, C = regional_feat.shape
        repeated_global = self.global_transform(global_feat).unsqueeze(1).repeat(1, n, 1)
        transformed_regional = self.regional_transform(regional_feat)
        concat_feat = torch.cat((transformed_regional, repeated_global), dim=-1)
        gate_values = self.gate(concat_feat)
        fused_feat = gate_values * transformed_regional + (1 - gate_values) * repeated_global
        return fused_feat
    

class Mono3DVGTransformer(nn.Module):
    def __init__(
            self,
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            return_intermediate_dec=False,
            num_feature_levels=4,
            dec_n_points=4,
            enc_n_points=4,
    ):

        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        hidden_dim = self.d_model

        encoder_layer = VisualEncoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points)
        self.encoder = VisualEncoder(encoder_layer, num_encoder_layers)

        self.TextGuidedAdapter = TextGuidedAdapter()

        decoder_layer = Mono3DVGDecoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points)
        self.decoder = Mono3DVGDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.reference_points = nn.Linear(d_model, 2)

        #### 定义两个全局可学习的token
        # CLS_visual = nn.Parameter()
        self.expander_dropout = 0.1
        self.cls_text = nn.Parameter(torch.randn(1,1,hidden_dim),requires_grad=True)
        self.global_text_att = nn.MultiheadAttention(hidden_dim, 8, dropout=self.expander_dropout)
        self.text_injection_att = nn.MultiheadAttention(hidden_dim, 8, dropout=self.expander_dropout)

        self.cls_visual = nn.Parameter(torch.randn(1,1,hidden_dim),requires_grad=True)
        self.global_visual_att= nn.MultiheadAttention(hidden_dim, 8, dropout=self.expander_dropout)
        self.global_visual_injection = GatedFusion(hidden_dim)

        # self.visual_injection_att = [nn.MultiheadAttention(hidden_dim, 8, dropout=self.expander_dropout) for i in range(4)]

        

        self._reset_parameters()
        ### 前后景 分隔 全连接层

        self.adapt_visual_classifier = nn.ModuleList([MLP(256, 256//2, 1, 3) for _ in range(4)])
        self.adapt_depth_classifier = nn.ModuleList([MLP(256, 256//2, 1, 3) for _ in range(1)])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            
            lr = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            tb = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            wh = torch.cat((lr, tb), -1)

            proposal = torch.cat((grid, wh), -1).view(N_, -1, 6)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None, depth_pos_embed=None,
                text_memory=None, text_mask=None,  im_name=None, instanceID=None, ann_id=None):
        assert query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):  ### 循环每一层的特征
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)### 记录每一层的特征
            src = src.flatten(2).transpose(1, 2)  ### 将宽*高进行拉直
            pos_embed = pos_embed.flatten(2).transpose(1, 2) ### 位置编码也是如此
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1) ### 位置编码加上层级编码
            ### 收集每一层的特征
            mask = mask.flatten(1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        ###使用cat进行拼接
        src_flatten = torch.cat(src_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs[0].device)###将shape转换为tensor
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)  ### 扩充到每个bs上

        # encoder  ##### 视觉信息编码  使用视觉特征作为query   文本信息作为key,value
        memory, text_memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios,text_memory, text_mask, lvl_pos_embed_flatten, mask_flatten)
        

        ### 提取文本全局特征
        cls_text = self.cls_text.expand(bs, 1, -1)
        global_text_feature  = self.global_text_att(
            query=cls_text.transpose(0, 1),
            key=text_memory.transpose(0, 1),
            value=text_memory.transpose(0, 1),
            key_padding_mask=text_mask
        )[0].transpose(0, 1).squeeze(1)

        ### 将全局信息融入到文本特征里去
        ## 全局信息作为第一个元素 进行一次自注意力可以吗
        all_text_feature = torch.cat([global_text_feature.unsqueeze(1),text_memory],dim=1)
        global_text_add_mask = torch.zeros(bs, 1).bool().to(global_text_feature.device)
        all_text_attention_mask = torch.cat([global_text_add_mask,text_mask],dim=1)

        # print('all_text_feature',all_text_feature.size())
        # print('all_text_attention_mask',all_text_attention_mask.size())
        all_text_feature = self.text_injection_att(
            query=all_text_feature.transpose(0, 1),
            key=all_text_feature.transpose(0, 1),
            value=all_text_feature.transpose(0, 1),
            key_padding_mask=all_text_attention_mask 
        )[0].transpose(0, 1)
        # print('all_text_feature',all_text_feature.size())
        text_memory = all_text_feature[:,1:,:]  ### 注入了文本信息的

        ### 提取视觉全局特征
        cls_visual = self.cls_visual.expand(bs, 1, -1)
        global_visual_feature  = self.global_visual_att(
            query=cls_visual.transpose(0, 1),
            key=memory.transpose(0, 1),
            value=memory.transpose(0, 1)
        )[0].transpose(0, 1).squeeze(1) 
        # print(memory.size())
        # print(global_visual_feature.size())
        memory = self.global_visual_injection(global_visual_feature,memory)
        # visual_align_memory = memory
        # prepare input for adapter and decoder
        bs, _, c = memory.shape
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)

        # Initial target query
        tgt = torch.zeros_like(query_embed)

        reference_points = self.reference_points(query_embed).sigmoid()
        init_reference_out = reference_points

        depth_pos_embed = depth_pos_embed.flatten(2).permute(2, 0, 1)
        mask_depth = masks[1].flatten(1)

        # prepare for adapter   ### 开始进行特征之间的交互
        img_feat_orig2adapt, depth_feat_orig2adapt = self.TextGuidedAdapter(memory, mask_flatten, lvl_pos_embed_flatten,
                reference_points, spatial_shapes,
                level_start_index,
                valid_ratios,
                text_memory, text_mask, depth_pos_embed, mask_depth, im_name, instanceID, ann_id)

        img_feat_srcs = img_feat_orig2adapt.chunk(2, dim=-1)
        memory_adapt_k = img_feat_srcs[1]
        depth_feat_srcs = depth_feat_orig2adapt.chunk(2, dim=-1)
        depth_adapt_k = depth_feat_srcs[1]

        # adapt_visual_src_list_classifier_one =None
        # adapt_depth_src_list_classifier_one = None
        # if self.training==True:
        
        #### 这里对原始视觉特征做拆分
        adapt_visual_src_list = memory_adapt_k.split([H_ * W_ for H_, W_ in spatial_shapes], dim=1) ### 按照原始拆分回原始的四层特征
        ##### 对每层特征进行分类
        adapt_visual_src_list_classifier = [self.adapt_visual_classifier[i](adapt_visual_src_list[i]) for i in range(len(adapt_visual_src_list))]
        ### 先高再宽  bs h w c
        adapt_visual_src_list_classifier_one = torch.cat(adapt_visual_src_list_classifier,dim=1)
    

        adapt_depth_src = depth_adapt_k.permute(1,0,2)
        adapt_depth_src_list_classifier_one = self.adapt_depth_classifier[0](adapt_depth_src)

        
        # decoder
        hs, inter_references, inter_references_dim = self.decoder(
            tgt,
            reference_points,
            memory_adapt_k,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            query_embed,
            mask_flatten,
            depth_pos_embed,
            depth_adapt_k,
            mask_depth,
            text_memory, text_mask,)

        inter_references_out = inter_references
        inter_references_out_dim = inter_references_dim

        return hs, init_reference_out, inter_references_out, inter_references_out_dim, None, None,adapt_visual_src_list_classifier_one, adapt_depth_src_list_classifier_one, global_text_feature,global_visual_feature


class VisualEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # Multi-Scale Deformable Attention
        self.msdeform_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index ,text_memory, text_mask, padding_mask=None):
        # Multi-Scale Deformable Attention
        src2 = self.msdeform_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask) ### 视觉特征的编码
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # cross attention
        src3 = self.cross_attn(query = self.with_pos_embed(src, pos).transpose(0, 1),
                               key = self.with_pos_embed(text_memory, torch.zeros_like(text_memory)).transpose(0, 1),
                               value= text_memory.transpose(0, 1)
                               )[0].transpose(0, 1)
        src = src + self.dropout4(src3)
        src = self.norm3(src)

        # ffn
        src = self.forward_ffn(src)
        return src, text_memory


class VisualEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios,text_memory, text_mask,
                pos=None, padding_mask=None):
        output = src
        text_output = text_memory
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)### 按照间隔进行拆分
        for _, layer in enumerate(self.layers):
            output, text_output = layer(output, pos, reference_points, spatial_shapes, level_start_index,text_output, text_mask, padding_mask)

        return output, text_output


class Mono3DVGDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # text cross attention
        self.cross_attn_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_text = nn.Dropout(dropout)
        self.norm_text = nn.LayerNorm(d_model)

        # depth cross attention
        self.cross_attn_depth = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_depth = nn.Dropout(dropout)
        self.norm_depth = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self,
                tgt,
                query_pos,
                reference_points,
                src,
                src_spatial_shapes,
                level_start_index,
                src_padding_mask,
                depth_pos_embed,
                depth_adapt_k,
                mask_depth,
                text_memory, text_mask):

        # Gather depth feats based on the text info
        tgt3 = self.cross_attn_depth(
                                    tgt.transpose(0, 1),
                                    depth_adapt_k,
                                     depth_pos_embed,
                                     key_padding_mask=mask_depth)[0].transpose(0, 1)
        tgt2 = self.dropout_depth(tgt3)
        tgt2 = self.norm_depth(tgt2)
        # Aggregate text info about the object
        tgt_text = self.cross_attn_text(self.with_pos_embed(tgt2, query_pos).transpose(0, 1),
                                        text_memory.transpose(0, 1),
                                        text_memory.transpose(0, 1),
                                        key_padding_mask=text_mask)[0].transpose(0, 1)
        tgt2 = self.dropout_text(tgt_text)
        tgt2 = self.norm_text(tgt2)
        # Gather visual feats based on the linguistic info
        tgt3 = self.cross_attn(self.with_pos_embed(tgt2, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt2 = tgt + self.dropout1(tgt3)
        tgt = self.norm1(tgt2)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class Mono3DVGDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.dim_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, depth_pos_embed=None, depth_adapt_k=None, mask_depth=None,text_memory=None, text_mask=None,):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        intermediate_reference_dims = []

        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 6:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            output = layer(output,
                           query_pos,
                           reference_points_input,
                           src,
                           src_spatial_shapes,
                           src_level_start_index,
                           src_padding_mask,
                           depth_pos_embed,
                           depth_adapt_k,
                           mask_depth,
                           text_memory, text_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 6:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.dim_embed is not None:
                reference_dims = self.dim_embed[lid](output)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_reference_dims.append(reference_dims)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), torch.stack(intermediate_reference_dims)

        return output, reference_points

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        if num_layers > 0:
            h = [hidden_dim] * (num_layers - 1)
            self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = []

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_mono3dvg_trans(cfg):
    return Mono3DVGTransformer(
        d_model=cfg['hidden_dim'],
        dropout=cfg['dropout'],
        activation="relu",
        nhead=cfg['nheads'],
        dim_feedforward=cfg['dim_feedforward'],
        num_encoder_layers=cfg['enc_layers'],
        num_decoder_layers=cfg['dec_layers'],
        return_intermediate_dec=cfg['return_intermediate_dec'],
        num_feature_levels=cfg['num_feature_levels'],
        dec_n_points=cfg['dec_n_points'],
        enc_n_points=cfg['enc_n_points'],
)
