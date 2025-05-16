from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch import nn
from model.clip import clip
from utils.layers import GraphConvolution, DistanceAdj
from utils.tools import get_batch_mask, get_prompt_text
from PIL import Image

class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor, attn_mask=None):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        attn_mask = attn_mask.to(device=x.device) if attn_mask is not None else self.attn_mask.to(device=x.device)
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=attn_mask)[0]

    def forward(self, x):
        x, padding_mask, attn_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask, attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask, attn_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIPVAD(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 device):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device

        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)

        self.clipmodel, self.clip_preprocessor = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(self.attn_window, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0

        return mask

    def build_random_attention_mask(self, attn_window, visual_length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(visual_length, visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(visual_length / attn_window)):
            if (i + 1) * attn_window < visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: visual_length, i * attn_window: visual_length] = 0

        return mask


    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1)) # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def encode_video(self, images, padding_mask, lengths):
        images = images.to(torch.float)
        visual_length = images.size(1)

        # Generate one window of position IDs
        position_ids = torch.arange(self.attn_window, device=self.device)

        # Repeat enough times to cover the sequence length
        repeats = (visual_length + self.attn_window - 1) // self.attn_window  # Ceiling division
        position_ids = position_ids.expand(images.shape[0], -1).repeat(1, repeats)

        # Truncate to match the exact length
        position_ids = position_ids[:, :visual_length]
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        attn_mask = self.build_random_attention_mask(self.attn_window, visual_length)
        x, _, _ = self.temporal((images, None, attn_mask))
        x = x.permute(1, 0, 2)

        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)
        x = self.linear(x)

        return x

    def encode_textprompt(self, text):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)

        return text_features

    def forward(self, visual, padding_mask, text, lengths):
        visual_features = self.encode_video(visual, padding_mask, lengths)
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))

        text_features_ori = self.encode_textprompt(text)

        text_features = text_features_ori
        logits_attn = logits1.permute(0, 2, 1)
        visual_attn = logits_attn @ visual_features
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])
        text_features = text_features_ori.unsqueeze(0)
        text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)

        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features_norm.permute(0, 2, 1)
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07

        return text_features_ori, logits1, logits2


class CLIPVADInference(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 device):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device

        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)

        ######## SHOULD BE SIMILAR TO ViT-B/16 preprocessor ########
        def center_crop_max_square(pil_img):
            w, h = pil_img.size
            crop_size = min(w, h)
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
            return pil_img.crop((left, top, left + crop_size, top + crop_size))

        self.clip_preprocessor = transforms.Compose([
                                transforms.Lambda(lambda img: Image.fromarray(img)),  # convert NumPy to PIL
                                transforms.Lambda(lambda img: center_crop_max_square(img)),
                                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                                transforms.Lambda(lambda img: img.convert("RGB")),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                                    std=(0.26862954, 0.26130258, 0.27577711)),
                            ])
        ###########################################################


        self.frame_position_embeddings = nn.Embedding(self.attn_window, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.initialize_parameters()

        label_map = dict({'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson', 'Assault': 'Assault', 'Burglary': 'Burglary', 'Explosion': 'Explosion', 'Fighting': 'Fighting', 'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery', 'Shooting': 'Shooting', 'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing', 'Vandalism': 'Vandalism'})
        self.prompt_text = get_prompt_text(label_map)
        self.text_features_ori = None


    def load_clip_model(self, model_path="ViT-B/16"):
        self.clipmodel, _ = clip.load("ViT-B/16", self.device)

        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False
        

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0

        return mask

    def build_random_attention_mask(self, attn_window, visual_length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(visual_length, visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(visual_length / attn_window)):
            if (i + 1) * attn_window < visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: visual_length, i * attn_window: visual_length] = 0

        return mask


    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1)) # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def encode_textprompt(self, text):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)

        return text_features

    
    def encode_local_window(self, images):
        images = images.to(torch.float)
        visual_length = images.size(1)
        

        position_ids = torch.arange(self.attn_window, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        position_ids = position_ids.repeat(1, images.size(1)//self.attn_window)

        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        attn_mask = self.build_random_attention_mask(self.attn_window, visual_length)
        x, _, _ = self.temporal((images, None, attn_mask))
        x = x.permute(1, 0, 2)

        return x

    def encode_global_window(self, x, class_logit=False):
        adj = self.adj4(x, seq_len=None)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)
        visual_features = self.linear(x)

        logits1 = self.classifier(visual_features + self.mlp2(visual_features))

        if class_logit:
            if self.text_features_ori is None:
                self.text_features_ori = self.encode_textprompt(self.prompt_text)

            text_features_ori = self.text_features_ori
            text_features = text_features_ori
            logits_attn = logits1.permute(0, 2, 1)
            visual_attn = logits_attn @ visual_features
            visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
            visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])
            text_features = text_features_ori.unsqueeze(0)
            text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
            text_features = text_features + visual_attn
            text_features = text_features + self.mlp1(text_features)

            visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features_norm = text_features_norm.permute(0, 2, 1)
            logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07

            return  logits1, logits2

        else:
            return logits1, None


    def compute_clip_emb(self, frame):
        preprocessed_img = self.clip_preprocessor(frame).unsqueeze(0)
        feature = self.clipmodel.encode_image(preprocessed_img.to(self.device))
        
        return feature                    

def center_crop(frames, crop_size):
    N, H, W, C = frames.shape
    crop_h, crop_w = crop_size, crop_size

    # Ensure crop fits within frame size
    if H < crop_h or W < crop_w:
        raise ValueError(f"Frame too small for crop: {(H, W)} vs (224, 224)")

    # Sample random top-left corner
    top = np.random.randint(0, H - crop_h + 1)
    left = np.random.randint(0, W - crop_w + 1)

    # Crop all frames at once (vectorized)
    cropped_frames = frames[:, top:top + crop_h, left:left + crop_w, :] 

    return cropped_frames

def load_model(checkpoint_path, device):
    data = torch.load(checkpoint_path, weights_only=False)
    model = CLIPVADInference(data["classes_num"], data["embed_dim"], \
                            data["visual_length"], data["visual_width"], \
                            data["visual_head"], data["visual_layers"], \
                            data["attn_window"], data["prompt_prefix"],\
                            data["prompt_postfix"], device)
    
    visual_length = data["visual_length"]
    attn_window = data["attn_window"]

    # state_dict = {k.replace("module.", "", 1): v for k, v in data["state_dict"].items() if "clipmodel" not in k}
    # data["state_dict"] = state_dict
    # torch.save(data, "pretrained/best_auc_no_clip.pt")
    
    model.load_state_dict(data["state_dict"], strict=True)
    model.to(device)
    
    print("Successfully load VadCLIP model")
    print(f"classes_num = {data['classes_num']}")
    print(f"embed_dim = {data['embed_dim']}")
    print(f"visual_length = {data['visual_length']}")
    print(f"visual_width = {data['visual_width']}")
    print(f"visual_head = {data['visual_head']}")
    print(f"visual_layers = {data['visual_layers']}")
    print(f"attn_window = {data['attn_window']}")
    print(f"prompt_prefix = {data['prompt_prefix']}")
    print(f"prompt_postfix = {data['prompt_postfix']}")

    return model