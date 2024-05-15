# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from .vit_unit import Block
from .vit_unit import PatchEmbed
from .vit_unit import get_sinusoid_encoding_table
import torch.nn.functional as F

from timm.models.layers import trunc_normal_ as __call_trunc_normal_


class RandomMaskingGenerator(object):
    def __init__(self, input_size, mask_ratio, fix=False):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.fix = fix
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)
        self.num_unmask = self.num_patches - self.num_mask

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        if self.fix:
            # mask_ratio = 0.75
            mask = np.array([1, 1, 1, 0]*49)
            return mask.astype(bool)

        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask.astype(bool)


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class PretrainVisionTransformerEncoder(nn.Module):
    """
    Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,      # FFN hidde dim
                 qkv_bias=False,    # qkv net bias
                 qk_scale=None,     # for different len of head_dim, Q@K could be huge which is bad for softmax. specific or not
                 drop_rate=0.,      # attention net proj dropout rate
                 attn_drop_rate=0.,     # attention net Q@K(softmax) drop rate
                 drop_path_rate=0.,     # drop_path after attn and mlp
                 norm_layer=nn.LayerNorm,   # before attn(block and mlp(block and head
                 init_values=None,  # learnable para*attn or mlp init
                 head_out_dim=0,  # use classifier head or not
                 device=torch.device('cuda', index=0)
                 ):
        super().__init__()
        self.head_out_dim = head_out_dim
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim).to(device)
        self.num_patches = self.patch_embed.num_patches

        # sine-cosine positional embeddings
        self.pos_embed = get_sinusoid_encoding_table(self.num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, device=device)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim, eps=1e-6).to(device)
        self.head = nn.Linear(embed_dim, head_out_dim) if head_out_dim > 0 else nn.Identity()
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, mask):        # mask, set 1: do not input
        x = self.patch_embed(x)     # B patch_num embed_dim

        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()  # self.pos_embed: 1, num_patch, embed_dim

        B, _, C = x.shape
        x_vis = x[:, ~mask, :].reshape(B, -1, C)  # ~mask means visible

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask):
        x = self.forward_features(x, mask)  # B patch_num emb_dim
        x = self.head(x)
        return x


class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 patch_size=16,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,  # FFN hidde dim
                 qkv_bias=False,  # qkv net bias
                 qk_scale=None,  # for different len of head_dim, Q@K could be huge which is bad for softmax. specific or not
                 drop_rate=0.,  # attention net proj dropout rate
                 attn_drop_rate=0.,  # attention net Q@K(softmax) drop rate
                 drop_path_rate=0.,  # drop_path after attn and mlp
                 norm_layer=nn.LayerNorm,  # before attn(block and mlp(block and head
                 init_values=None,  # learnable para*attn or mlp init
                 head_out_dim=0,  # use classifier head or not
                 device=torch.device('cuda', index=0)
                 ):
        super().__init__()
        self.head_out_dim = head_out_dim
        assert head_out_dim == 3 * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, device=device)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim, eps=1e-6).to(device)
        self.head = nn.Linear(embed_dim, head_out_dim) if head_out_dim > 0 else nn.Identity()
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:, :]))  # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))  # [B, N, 3*16^2]

        return x


class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_head_out_dim=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_head_out_dim=768,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,  # FFN hidde dim
                 qkv_bias=False,  # qkv net bias
                 qk_scale=None, # for different len of head_dim, Q@K could be huge which is bad for softmax. specific or not
                 drop_rate=0.,  # attention net proj dropout rate
                 attn_drop_rate=0.,  # attention net Q@K(softmax) drop rate
                 drop_path_rate=0.,  # drop_path after attn and mlp
                 norm_layer=nn.LayerNorm,  # before attn(block and mlp(block and head
                 init_values=0,  # learnable para*attn or mlp init
                 device=torch.device('cuda', index=0)
                 ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            head_out_dim=encoder_head_out_dim,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            device=device
            )

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            head_out_dim=decoder_head_out_dim,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            device=device
            )

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim)).to(device)
        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)

        # init parameter
        trunc_normal_(self.mask_token, std=.02)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask):
        x_vis = self.encoder(x, mask)  # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[:, ~mask, :].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[:, mask, :].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)
        # notice: if N_mask==0, the shape of x is [B, N_mask, 3 * 16 * 16]
        x = self.decoder(x_full, pos_emd_mask.shape[1])  # [B, N_mask, 3 * 16 * 16]

        return x


class PretrainVisionTransformerE2D(nn.Module):
    def __init__(self, num_class=2, head=True):
        super().__init__()
        self.in_feature_dim = 768
        self.attention_net_dim = 768
        self.attention_V = nn.Sequential(
            nn.Linear(self.in_feature_dim, self.attention_net_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.in_feature_dim, self.attention_net_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(self.attention_net_dim, 1)
        self.head = nn.Linear(768, num_class, bias=False) if head else nn.Identity()

    def forward(self, x):
        # x: B patch_num emb_dim
        U = self.attention_U(x)     # U: B patch_num emb_dim
        V = self.attention_V(x)     # V: B patch_num emb_dim
        A = self.attention_weights(U * V)   # A: B patch_num 1
        A = F.softmax(A, dim=1)     # A: B patch_num 1
        attention_feature = x.permute(0, 2, 1) @ A   # B emb_dim patch_num @ B patch_num 1 -> B emb_dim 1
        attention_feature = attention_feature.squeeze()     # B emb_dim
        class_out = self.head(attention_feature)    # B emb_dim --> B num_class

        return attention_feature, A.squeeze(), class_out


class PretrainVisionTransformerAttention(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_head_out_dim=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_head_out_dim=768,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,  # FFN hidde dim
                 qkv_bias=False,  # qkv net bias
                 qk_scale=None, # for different len of head_dim, Q@K could be huge which is bad for softmax. specific or not
                 drop_rate=0.,  # attention net proj dropout rate
                 attn_drop_rate=0.,  # attention net Q@K(softmax) drop rate
                 drop_path_rate=0.,  # drop_path after attn and mlp
                 norm_layer=nn.LayerNorm,  # before attn(block and mlp(block and head
                 init_values=0,  # learnable para*attn or mlp init
                 device=torch.device('cuda', index=0)
                 ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            head_out_dim=encoder_head_out_dim,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            device=device
            )

        # self.decoder = PretrainVisionTransformerDecoder(
        #     patch_size=patch_size,
        #     head_out_dim=decoder_head_out_dim,
        #     embed_dim=decoder_embed_dim,
        #     depth=decoder_depth,
        #     num_heads=decoder_num_heads,
        #     mlp_ratio=mlp_ratio,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     drop_rate=drop_rate,
        #     attn_drop_rate=attn_drop_rate,
        #     drop_path_rate=drop_path_rate,
        #     norm_layer=norm_layer,
        #     init_values=init_values,
        #     device=device
        #     )

        self.attention_net = PretrainVisionTransformerE2D(head=False).to(device)
        # self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False).to(device)
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim)).to(device)
        # self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)

        # init parameter
        # trunc_normal_(self.mask_token, std=.02)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask):
        # mask = np.array([False] * 14 * 14)
        # x_encoder = self.encoder(x, mask)  # B patch_num emb_dim
        # x_attention, A, _ = self.attention_net(x_encoder)  # x_attention: B emb_dim   A: B patch_num
        # x_attention = x_attention.unsqueeze(dim=1)    # x_attention: B 1 beb_dim
        # x_attention = self.encoder_to_decoder(x_attention)  # x_attention: B 1 beb_dim
        # B, _, emb_dim = x_attention.shape
        #
        # expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x_attention).to(x.device).clone().detach()
        # mask = np.array([True] * 14 * 14)
        # mask[0] = False
        # pos_emd_vis = expand_pos_embed[:, ~mask, :].reshape(B, -1, emb_dim)
        # pos_emd_mask = expand_pos_embed[:, mask, :].reshape(B, -1, emb_dim)
        #
        # x_full = torch.cat([x_attention + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)
        # x = self.decoder(x_full, 14*14)  # B patch_num emb_dim
        #
        # return x, A

        # =======for feature extract==========
        mask = np.array([False] * 14 * 14)
        x_encoder = self.encoder(x, mask)  # B patch_num emb_dim
        x_attention, A, _ = self.attention_net(x_encoder)  # x_attention: B emb_dim   A: B patch_num
        # x_attention = x_attention.unsqueeze(dim=1)    # x_attention: B 1 beb_dim
        # x_attention = self.encoder_to_decoder(x_attention)  # x_attention: B 1 beb_dim
        # B, _, emb_dim = x_attention.shape
        #
        # expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x_attention).to(x.device).clone().detach()
        # mask = np.array([True] * 14 * 14)
        # mask[0] = False
        # pos_emd_vis = expand_pos_embed[:, ~mask, :].reshape(B, -1, emb_dim)
        # pos_emd_mask = expand_pos_embed[:, mask, :].reshape(B, -1, emb_dim)
        #
        # x_full = torch.cat([x_attention + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)
        # x = self.decoder(x_full, 14*14)  # B patch_num emb_dim

        return x_attention, A, _


class MyEncoder(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_head_out_dim=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 mlp_ratio=4.,  # FFN hidde dim
                 qkv_bias=False,  # qkv net bias
                 qk_scale=None,
                 # for different len of head_dim, Q@K could be huge which is bad for softmax. specific or not
                 drop_rate=0.,  # attention net proj dropout rate
                 attn_drop_rate=0.,  # attention net Q@K(softmax) drop rate
                 drop_path_rate=0.,  # drop_path after attn and mlp
                 norm_layer=nn.LayerNorm,  # before attn(block and mlp(block and head
                 init_values=0,  # learnable para*attn or mlp init
                 device=torch.device('cuda', index=0)
                 ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            head_out_dim=encoder_head_out_dim,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            device=device
        )

        self.attention_net = PretrainVisionTransformerE2D(num_class=2, head=True).to(device)

    def forward(self, x, mask=None):
        x = self.encoder(x, np.array([False] * 14 * 14))  # B patch_num emb_dim
        attention_feature, A, class_out = self.attention_net(x)  # class_out: B 2  // A: B patch_num // attention_feature: B beb_dim

        return attention_feature, A, class_out


