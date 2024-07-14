import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import yaml
from clip import clip
from clip.model import QuickGELU,LayerNorm
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from operator import mul
from functools import reduce
import math
_tokenizer = _Tokenizer()

class Attention(nn.Module):
    '''
    A generalized attention module with more flexibility.
    '''

    def __init__(
            self, q_in_dim: int, k_in_dim: int, v_in_dim: int,
            qk_proj_dim: int, v_proj_dim: int, num_heads: int,
            out_dim: int
    ):
        super().__init__()

        self.q_proj = nn.Linear(q_in_dim, qk_proj_dim)
        self.k_proj = nn.Linear(k_in_dim, qk_proj_dim)
        self.v_proj = nn.Linear(v_in_dim, v_proj_dim)
        self.out_proj = nn.Linear(v_proj_dim, out_dim)

        self.num_heads = num_heads
        assert qk_proj_dim % num_heads == 0 and v_proj_dim % num_heads == 0

        self._initialize_weights()

    def _initialize_weights(self):
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
        N = q.size(0);
        assert k.size(0) == N and v.size(0) == N
        Lq, Lkv = q.size(1), k.size(1);
        assert v.size(1) == Lkv

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)

        H = self.num_heads
        Cqk, Cv = q.size(-1) // H, v.size(-1) // H

        q = q.view(N, Lq, H, Cqk)
        k = k.view(N, Lkv, H, Cqk)
        v = v.view(N, Lkv, H, Cv)

        aff = torch.einsum('nqhc,nkhc->nqkh', q / (Cqk ** 0.5), k)
        aff = aff.softmax(dim=-2)
        mix = torch.einsum('nqlh,nlhc->nqhc', aff, v)

        out = self.out_proj(mix.flatten(-2))

        return out

class InsertEventPrompt(nn.Module):
    def __init__(
            self, cfg
    ):
        super().__init__()
        if cfg['MODEL']['BACKBONE']['Name'] == "ViT-L-14-336px":
            patch_size = (24, 24)
        else:
            patch_size = (16, 16)

        if cfg['MODEL']['BACKBONE']['Name'] == "ViT-L-14-336px" or cfg['MODEL']['BACKBONE']['Name'] == "ViT-L-14":
            feature_dim = 1024
            num_heads = 16
        else:
            feature_dim = 768
            num_heads = 12

        self.num_frames = cfg['Dataset']['Num_frame']
        self.use_event_modality_prompts = cfg['MODEL']['EventEncoder']['use_event_modality_prompts']
        self.num_event_modality_prompts = cfg['MODEL']['EventEncoder']['num_event_modality_prompts']
        if self.use_event_modality_prompts:
            self.event_modality_prompts = nn.Parameter(torch.zeros(self.num_event_modality_prompts, feature_dim))
            self._initialize_event_modality_prompts(patch_size, feature_dim)

        self.use_cross_frame_prompts = cfg['MODEL']['EventEncoder']['use_cross_frame_prompts']
        self.use_intra_frame_prompts = cfg['MODEL']['EventEncoder']['use_intra_frame_prompts']

        # for both cross_frame_prompts and intra_frame_prompts we need the cls_proj layer and the num_frames
        if self.use_cross_frame_prompts or self.use_intra_frame_prompts:
            self.cls_proj = nn.Linear(feature_dim, feature_dim)

        # for cross_frame_prompts we need a layer norm and attention
        if self.use_cross_frame_prompts:
            self.cross_frame_prompts_ln = LayerNorm(feature_dim)
            self.cross_frame_prompts_attn_layer = Attention(
                q_in_dim=feature_dim, k_in_dim=feature_dim, v_in_dim=feature_dim,
                qk_proj_dim=feature_dim, v_proj_dim=feature_dim, num_heads=num_heads, out_dim=feature_dim)

        # for intra_frame_prompts, init learnable tokens
        if self.use_intra_frame_prompts:
            self.intra_frame_prompts = nn.Parameter(torch.zeros(1, self.num_frames, feature_dim))
            self._initialize_intra_frame_prompts(patch_size, feature_dim)

    def _initialize_intra_frame_prompts(self, patch_size, prompt_dim):
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.intra_frame_prompts.data, -val, val)

    def _initialize_event_modality_prompts(self, patch_size, prompt_dim):
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.event_modality_prompts.data, -val, val)

    def forward(self, x, B, T):
        device = x.device
        shape = 0
        if self.use_intra_frame_prompts:
            shape = self.intra_frame_prompts.shape[1]
        if self.use_event_modality_prompts:
            event_modality_prompts = self.event_modality_prompts.expand(B * T, -1, -1).to(device)
            # add global_prompts after the cls token while in front of the original token.
            x = torch.cat((x[:, :1, :], event_modality_prompts, x[:, 1:, :]), dim=1)
        if self.use_cross_frame_prompts or self.use_intra_frame_prompts:
            BT, N, C = x.shape
            T = self.num_frames
            B = BT // T

            cls_token = x[:, 0, :].view(B, T, C)
            cls_token_proj = self.cls_proj(cls_token)  # B, T, C

        # then apply ln and attn if cross_frame_prompts being used
        if self.use_cross_frame_prompts:
            cross_frame_prompts_norm = self.cross_frame_prompts_ln(cls_token_proj).to(device)
            cross_frame_prompts_attn = cls_token_proj + self.cross_frame_prompts_attn_layer(
                cross_frame_prompts_norm, cross_frame_prompts_norm,
                cross_frame_prompts_norm)
            cross_frame_prompts_attn_reshape = cross_frame_prompts_attn.view(BT, 1, C)
            x = torch.cat([x, cross_frame_prompts_attn_reshape], dim=1)

        # then if local prompts are being used
        if self.use_intra_frame_prompts:
            intra_frame_prompts = self.intra_frame_prompts.expand(B, -1, -1).to(device)
            # If train time frames and
            # test time frames are not equal
            if T != self.num_frames:
                token_multiplier = T // self.num_frames
                intra_frame_prompts = intra_frame_prompts.repeat(1, token_multiplier, 1)

            # use additive conditioning
            intra_frame_prompts = intra_frame_prompts + cls_token_proj  # B, T, C

            # repeat across frames
            intra_frame_prompts_i = intra_frame_prompts.repeat_interleave(repeats=T, dim=0)
            x = torch.cat((x[:, :1, :], intra_frame_prompts_i, x[:, 1:, :]), dim=1)
        return x, shape

class EventEncoder(nn.Module):

    def __init__(
            self, cfg, clip_model
    ):
        super().__init__()
        if cfg['MODEL']['BACKBONE']['Name'] == "ViT-L-14-336px":
            input_size = (336, 336)
            patch_size = (24, 24)
        else:
            input_size = (224, 224)
            patch_size = (16, 16)

        if cfg['MODEL']['BACKBONE']['Name'] == "ViT-L-14-336px" or cfg['MODEL']['BACKBONE']['Name'] == "ViT-L-14":
            feature_dim = 1024
            num_layers = 24
        else:
            feature_dim = 768
            num_layers = 12

        self.visual = clip_model.visual
        self.feature_dim = feature_dim
        self.num_frames = cfg['Dataset']['Num_frame']
        self.representation = cfg['Dataset']['Representation']
        self.num_patches = np.prod([x // y for x, y in zip(input_size, patch_size)]) + 1

        self.cls_token = nn.Parameter(torch.zeros([feature_dim]))
        self.pos_embed = nn.Parameter(torch.zeros([self.num_patches, feature_dim]))  # zero initialization for pos_embed
        self.time_embed = nn.Parameter(
            torch.zeros([self.num_frames, feature_dim]))  # zero initialization for time_embed
        self.visual.proj = clip_model.visual.proj

        self.use_temporal_encoding = cfg['MODEL']['EventEncoder']['use_temporal_encoding']
        self.use_event_modality_prompts = cfg['MODEL']['EventEncoder']['use_event_modality_prompts']
        self.num_event_modality_prompts = cfg['MODEL']['EventEncoder']['num_event_modality_prompts']

        self.InsertEventPrompt = nn.ModuleList()
        for i in range(num_layers):
            self.InsertEventPrompt.append(InsertEventPrompt(cfg))

        self.use_cross_frame_prompts = cfg['MODEL']['EventEncoder']['use_cross_frame_prompts']
        self.use_intra_frame_prompts = cfg['MODEL']['EventEncoder']['use_intra_frame_prompts']

        self._initialize_weights()

        # for low_level_feature projection
        self.output_dim = 512
        scale = feature_dim ** -0.5
        self.low_level_idx = cfg['MODEL']['EventEncoder']['Low_level_feature_idx']
        self.proj_low_level = []
        self.ln_low_level = nn.ModuleList()
        for i in range(len(self.low_level_idx)):
            self.ln_low_level.append(LayerNorm(feature_dim))
            self.proj_low_level.append(nn.Parameter(scale * torch.randn(feature_dim, self.output_dim)))

    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.time_embed, std=0.02)

    def temporal_encoding(self, x, T, B):
        ## Time Embeddings
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)

        ## Resizing time embeddings in case they don't match
        if T != self.time_embed.size(0):
            time_embed = self.time_embed.unsqueeze(0).transpose(1, 2)
            new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
            new_time_embed = new_time_embed.transpose(1, 2).squeeze(0)
            x = x + new_time_embed
        else:
            x = x + self.time_embed

        x = rearrange(x, '(b n) t m -> (b t) n m', b=B, t=T)
        return x

    def resize_to_resolution(self, x, output_resolution=(224, 224)):

        x = F.interpolate(x, size=output_resolution)

        return x

    def forward(self, x: torch.Tensor, real_num_frame):

        B, C, T, H, W = x.size()
        device = x.device
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)  # BT, C, H, W
        # x transform into batches
        x = self.visual.conv1(x)  # shape = [BT, width, H, W]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # shape = [BT, HW, width]
        # add cls token
        x = torch.cat(
            [self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                   device=x.device), x], dim=1)  # BT, HW+1, width
        # add pos_embed
        x = x + self.visual.positional_embedding.to(x.dtype)
        # add temporal_embed
        if self.use_temporal_encoding:
            x = self.temporal_encoding(x, T, B)
        # layer normalization
        x = self.visual.ln_pre(x)

        event_low_level_feature = []
        for i, blk in enumerate(self.visual.transformer.resblocks):
            # the global prompts are inserted between every transformer block,
            # by concatenating with the x output from the last transformer block.
            x, shape = self.InsertEventPrompt[i](x, B, T)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = blk(x)
            x = x.permute(1, 0, 2)  # NLD -> LND

            # extract the output x without the intra_frame_prompts
            if self.use_intra_frame_prompts:
                x = torch.cat((x[:, :1, :], x[:, shape + 1:, :]), dim=1)
            # extract the output x without the cross_frame_prompts
            if self.use_cross_frame_prompts:
                x = x[:, :-1, :]
            # extract the output x without the event_modality_prompts
            if self.use_event_modality_prompts:
                x = torch.cat((x[:, :1, :], x[:, self.num_event_modality_prompts + 1:, :]), dim=1)

            if i + 1 in self.low_level_idx:
                event_low_level_feature.append(x)  # LND -> NLD

        # extract cls token for the final event embedding
        cls_x = self.visual.ln_post(x[:, 0, :])
        cls_x = cls_x @ self.visual.proj  # b, t, e

        # average cls tokens from all frames
        cls_x = rearrange(cls_x, '(b t) e -> b t e', b=B, t=T).mean(dim=1)  # b, e

        cls_low_level_x = []
        if len(self.low_level_idx) != 0:
            for i in range(len(event_low_level_feature)):
                cls_low_level_x_i = self.ln_low_level[i](event_low_level_feature[i][:, 0, :])
                cls_low_level_x_i = cls_low_level_x_i @ self.proj_low_level[i].to(device)  # b, t, e
                # average cls tokens from all frames
                cls_low_level_x_i = rearrange(cls_low_level_x_i, '(b t) e -> b t e', b=B, t=T).mean(dim=1)  # b, e
                cls_low_level_x.append(cls_low_level_x_i)
            return cls_x, cls_low_level_x

        else:
            return cls_x

def read_yaml(yaml_path):
    with open(yaml_path, encoding="utf-8", mode="r") as f:
        result = yaml.load(stream=f, Loader=yaml.FullLoader)
        return result

def load_clip_to_cpu(cfg):
    backbone_name = cfg['MODEL']['BACKBONE']['Name']
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root=cfg['MODEL']['BACKBONE']['PRE_trained_model'])

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model

