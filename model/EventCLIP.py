import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
import yaml
from .TextEncoder import SpecificTextualPrompt, TextEncoder
from .EventEncoder import EventEncoder
from .ImageEncoder import ImageEncoder

def load_clip_to_cpu(cfg):
    backbone_name = cfg['MODEL']['BACKBONE']['Name']
    url = clip._MODELS[backbone_name]
    model_path = cfg['MODEL']['BACKBONE']['PRE_trained_model']

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model

class EventCLIP(nn.Module):
    def __init__(self, cfg, clip_model_im, clip_model_ev):
        super().__init__()
        self.use_init_ctx = cfg['MODEL']['TextEncoder']['init_ctx']
        self.use_leranable_ctx = cfg['MODEL']['TextEncoder']['leranable_ctx']
        if self.use_init_ctx:
            self.prompt_init = SpecificTextualPrompt(cfg, clip_model_im,
                                   init_ctx = cfg['MODEL']['TextEncoder']['init_ctx'])
        if self.use_leranable_ctx:
            self.prompt_leranable = SpecificTextualPrompt(cfg, clip_model_im,
                                        leranable_ctx = cfg['MODEL']['TextEncoder']['leranable_ctx'])
        self.image_encoder = ImageEncoder(cfg, clip_model_im)
        self.text_encoder = TextEncoder(clip_model_im)
        self.event_encoder = EventEncoder(cfg, clip_model_ev)
        self.logit_scale = clip_model_im.logit_scale.exp()
        self.dtype = clip_model_im.dtype
        self.low_level_idx = cfg['MODEL']['EventEncoder']['Low_level_feature_idx']
        self.use_image_bias_textual_prompts = cfg['MODEL']['TextEncoder']['use_image_bias_textual_prompts']
        self.use_event_bias_textual_prompts = cfg['MODEL']['TextEncoder']['use_event_bias_textual_prompts']

    def forward(self, events, image, class_idxs, real_num_frame):
        image_features, low_image_features_list = self.image_encoder(image)  # b,dim_i]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # b,dim_i [2,512]

        event_features, low_event_features_list = self.event_encoder(events,real_num_frame)
        event_features = event_features / event_features.norm(dim=-1, keepdim=True)  # b,dim_e [2,512]

        text_features_e = self.genenrate_text_prompt(event_features, class_idxs, self.use_event_bias_textual_prompts)
        text_features_im = self.genenrate_text_prompt(image_features, class_idxs, self.use_image_bias_textual_prompts)
        return image_features, event_features, text_features_e, text_features_im

    def genenrate_text_prompt(self, f, class_idxs, use_bias_prompts):
        device = f.device
        text_features_init, text_features_leranable = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
        if self.use_init_ctx:
            Text_Prompts_init, tokenized_prompts_init = self.prompt_init(f, class_idxs, use_bias_prompts)
            Text_Prompts_init = Text_Prompts_init[0, :, :, :]  # n_cls, n_tkn, ctx_dim
            text_features_init = self.text_encoder(Text_Prompts_init, tokenized_prompts_init)
            text_features_init = text_features_init / text_features_init.norm(dim=-1, keepdim=True)  # b,dim_t_e
            return text_features_init
        if self.use_leranable_ctx:
            Text_Prompts_leranable, tokenized_prompts_leranable = self.prompt_leranable(f, class_idxs, use_bias_prompts)
            Text_Prompts_leranable = Text_Prompts_leranable[0, :, :, :]  # n_cls, n_tkn, ctx_dim
            text_features_leranable = self.text_encoder(Text_Prompts_leranable, tokenized_prompts_leranable)
            text_features_leranable = text_features_leranable / text_features_leranable.norm(dim=-1, keepdim=True)  # b,dim_t_e
            return text_features_leranable
        if self.use_leranable_ctx and self.use_init_ctx:
            return (text_features_init + text_features_leranable) / 2.0

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

def read_yaml(yaml_path):
    with open(yaml_path, encoding="utf-8", mode="r") as f:
        result = yaml.load(stream=f, Loader=yaml.FullLoader)
        return result

def load_clip_to_cpu(cfg):
    backbone_name = cfg['MODEL']['BACKBONE']['Name']
    model_path = cfg['MODEL']['BACKBONE']['PRE_trained_model']

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model

