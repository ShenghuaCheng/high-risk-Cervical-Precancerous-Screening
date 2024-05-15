# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from .vit_unit import Block
import torch.nn.functional as F


class AttentionNet(nn.Module):
    """the same as .mae_simple.PretrainVisionTransformerE2D"""

    def __init__(self, dim, num_class=2, head=True):
        super().__init__()
        self.in_feature_dim = dim
        self.attention_net_dim = dim
        self.attention_V = nn.Sequential(
            nn.Linear(self.in_feature_dim, self.attention_net_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.in_feature_dim, self.attention_net_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(self.attention_net_dim, 1)
        self.head = nn.Linear(dim, num_class, bias=False) if head else nn.Identity()

    def forward(self, x):
        # x: B patch_num emb_dim
        U = self.attention_U(x)  # U: B patch_num emb_dim
        V = self.attention_V(x)  # V: B patch_num emb_dim
        A = self.attention_weights(U * V)  # A: B patch_num 1
        A = F.softmax(A, dim=1)  # A: B patch_num 1
        attention_feature = x.permute(0, 2, 1) @ A  # B emb_dim patch_num @ B patch_num 1 -> B emb_dim 1
        attention_feature = attention_feature.squeeze()  # B emb_dim
        class_out = self.head(attention_feature)  # B emb_dim --> B num_class
        return attention_feature, A.squeeze(), class_out


class FeatureEncoder(nn.Module):
    def __init__(self,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,  # FFN hidde dim
                 qkv_bias=False,  # qkv net bias
                 qk_scale=None,
                 # for different len of head_dim, Q@K could be huge which is bad for softmax. specific or not
                 drop_rate=0.,  # attention net proj dropout rate
                 attn_drop_rate=0.,  # attention net Q@K(softmax) drop rate
                 drop_path_rate=0.,  # drop_path after attn and mlp
                 norm_layer=nn.LayerNorm,  # before attn(block and mlp(block and head
                 init_values=0,  # learnable para*attn or mlp init
                 device=torch.device('cuda', index=0)):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, device=device)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # B token_num embed_dim -> B token_num embed_dim
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


class AttentionFeatureClassifier(nn.Module):
    def __init__(self, pre_head, encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12, mlp_ratio=4, qkv_bias=True,
                 device=torch.device('cuda', index=0)):
        super().__init__()
        self.dim = 768
        if pre_head is None:
            self.pre_head = nn.Identity()
        else:
            self.pre_head = pre_head
        self.s2s = FeatureEncoder(
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,  # FFN hidde dim
            qkv_bias=qkv_bias,  # qkv net bias
            qk_scale=None,  # for different len of head_dim, Q@K could be huge which is bad for softmax. specific or not
            drop_rate=0.,  # attention net proj dropout rate
            attn_drop_rate=0.,  # attention net Q@K(softmax) drop rate
            drop_path_rate=0.,  # drop_path after attn and mlp
            norm_layer=nn.LayerNorm,  # before attn(block and mlp(block and head
            init_values=0,  # learnable para*attn or mlp init
            device=device
        )
        self.attention_net = AttentionNet(dim=self.dim, num_class=2, head=True)
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

    def forward(self, x):
        # B token_num embed_dim -> B token_num embed_dim
        x = self.pre_head(x)
        x = self.s2s(x)
        # -> B emb_dim // B token_num // B class_num
        attention_feature, A, class_out = self.attention_net(x)
        return attention_feature, A, class_out


class AttentionFeatureClassifierLinear(nn.Module):
    def __init__(self, pre_head, encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12, mlp_ratio=4, qkv_bias=True,
                 device=torch.device('cuda', index=0)):
        super().__init__()
        self.dim = 768
        self.linear = nn.Linear(1024, 768)
        if pre_head is None:
            self.pre_head = nn.Identity()
        else:
            self.pre_head = pre_head
        self.s2s = FeatureEncoder(
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,  # FFN hidde dim
            qkv_bias=qkv_bias,  # qkv net bias
            qk_scale=None,  # for different len of head_dim, Q@K could be huge which is bad for softmax. specific or not
            drop_rate=0.,  # attention net proj dropout rate
            attn_drop_rate=0.,  # attention net Q@K(softmax) drop rate
            drop_path_rate=0.,  # drop_path after attn and mlp
            norm_layer=nn.LayerNorm,  # before attn(block and mlp(block and head
            init_values=0,  # learnable para*attn or mlp init
            device=device
        )
        self.attention_net = AttentionNet(dim=self.dim, num_class=2, head=True)
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

    def forward(self, x):
        # B token_num embed_dim -> B token_num embed_dim
        x = self.linear(x)
        x = self.pre_head(x)
        x = self.s2s(x)
        # -> B emb_dim // B token_num // B class_num
        attention_feature, A, class_out = self.attention_net(x)
        return attention_feature, A, class_out
