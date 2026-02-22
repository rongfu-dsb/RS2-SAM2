import torch
import torch.nn.functional as F
from torch import nn

import os
# import math
# from util import box_ops
# from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
#                        nested_tensor_from_videos_list,
#                        accuracy, get_world_size, interpolate,
#                        is_dist_avail_and_initialized, inverse_sigmoid)
#
# from .position_encoding import PositionEmbeddingSine1D
# from .backbone import build_backbone
# from .deformable_transformer import build_deforamble_transformer
# from .segmentation import CrossModalFPNDecoder, VisionLanguageFusionModule
# from .matcher import build_matcher
# from .criterion import SetCriterion
# from .postprocessors import build_postprocessors
from .evf_sam import EvfSamModel
from .configuration_evf import EvfConfig
# from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizerFast, AutoConfig
from transformers import XLMRobertaTokenizer, AutoTokenizer
# import transformers
import copy
from einops import rearrange, repeat



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)


class ReferFormer(nn.Module):
    """ This is the ReferFormer module that performs referring video object detection """

    def __init__(self, evf_model, tokenizer,inference=True,precision='fp32'):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         ReferFormer can detect in a video. For ytvos, we recommend 5 queries for each frame.
            num_frames:  number of clip frames
            mask_dim: dynamic conv inter layer channel number.
            dim_feedforward: vision-language fusion module ffn channel number.
            dynamic_mask_channels: the mask feature output channel number.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.tokenizer= tokenizer
        self.evf_model = evf_model
        self.inference = inference
        self.precision = precision
    def forward(self, imgs, img_evfs, targets, captions):
        """?The forward expects a NestedTensor, which consists of:
               - samples.tensors: image sequences, of shape [num_frames x 3 x H x W]
               - samples.mask: a binary mask of shape [num_frames x H x W], containing 1 on padded pixels
               - captions: list[str]
               - targets:  list[dict]

            It returns a dict with the following elements:
               - "pred_masks": Shape = [batch_size x num_queries x out_h x out_w]

               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        # Backbone
        # if not isinstance(samples, NestedTensor):
        #     samples = nested_tensor_from_videos_list(samples)
        # if not isinstance(img_evfs_sap, NestedTensor):
        #     img_evfs_sap = nested_tensor_from_videos_list(img_evfs_sap)



        # imgs = samples.tensors

        # img_evfs = img_evfs_sap.tensors
        # B, T, C,H,W = imgs.shape
        B,C,H,W = imgs.shape
        # t=T
        # batch_size = B
        # imgs = rearrange(imgs, 'b t c h w -> (b t) c h w')

        # img_evfs = rearrange(img_evfs, 'b t c h w ->(b t) c h w')
        # if 'valid_indices' in targets[0]:
        #     valid_indices = torch.tensor([i * t + target['valid_indices'] for i, target in enumerate(targets)]).to(
        #         imgs.device)
        #     imgs = imgs.index_select(0, valid_indices)
        #     img_evfs = img_evfs.index_select(0, valid_indices)
        #     # samples.mask = samples.mask.index_select(0, valid_indices)
        #     # t: num_frames -> 1
        #     T = 1

        if self.precision == "fp16":
            imgs = imgs.half()
            img_evfs = img_evfs.half()
        elif self.precision == "bf16":
            imgs = imgs.bfloat16()
            img_evfs = img_evfs.bfloat16()
        else:
            imgs = imgs.float()
            img_evfs = img_evfs.float()

        # gt_mask = targets['masks']
        #
        # shape_label = torch.ones(targets['masks'].shape[-1], targets['masks'].shape[2])*255
        # orig_size = [tuple(target['orig_size'].tolist())  for target in targets for _ in range(T)]
        # resize = [tuple(target['size'].tolist()) for target in targets for _ in range(T)]
        input_ids = [
            self.tokenizer(prompt, return_tensors="pt").input_ids[0]
            for prompt in captions
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_masks = input_ids.ne(self.tokenizer.pad_token_id)
        if self.inference == False:
            truncate_len = self.tokenizer.model_max_length
            if input_ids.shape[1] > truncate_len:
                input_ids = input_ids[:, :truncate_len]
                attention_masks = attention_masks[:, :truncate_len]
        # repeat prompt
        # input_ids = repeat(input_ids, 'b l->(b t) l',t=T)
        # attention_masks = repeat(attention_masks, 'b l->(b t) l',t=T)

        offset = [0]
        cnt = 0
        for prompt in captions:
            cnt += 1
            offset.append(cnt)
            # for _ in range(T):

        offset = torch.LongTensor(offset)
        device = imgs.device
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        offset = offset.to(device)
        # pred_mask = self.evf_model(imgs, img_evfs, input_ids, attention_masks,offset,orig_size,resize,self.inference)
        pred_mask,text = self.evf_model(imgs, img_evfs,input_ids, attention_masks, offset,
                                   self.inference)
        # pred_mask = rearrange(pred_mask,'(b t) q h w-> b t q h w', b=B,t=T)
        # sen_feat = rearrange(sen_feat, '(b t) h w -> b t h w', b=B,t=T)
        out = torch.cat([1-pred_mask,pred_mask],dim=1)

        # out['pred_masks'] = pred_mask
        # out['sen_feat'] = sen_feat
        # if self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_seg_masks)
        return out,text

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b, "pred_masks": c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_seg_masks[:-1])]

    def forward_text(self, captions, device):
        if isinstance(captions[0], str):
            tokenized = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt").to(device)
            encoded_text = self.text_encoder(**tokenized)
            # encoded_text.last_hidden_state: [batch_size, length, 768]
            # encoded_text.pooler_output: [batch_size, 768]
            text_attention_mask = tokenized.attention_mask.ne(1).bool()
            # text_attention_mask: [batch_size, length]

            text_features = encoded_text.last_hidden_state
            text_features = self.resizer(text_features)
            text_masks = text_attention_mask
            text_features = NestedTensor(text_features, text_masks)  # NestedTensor

            # text_sentence_features = encoded_text.pooler_output
            # text_sentence_features = self.resizer(text_sentence_features)
        else:
            raise ValueError("Please mask sure the caption is a list of string")
        return text_features
        # return text_features, text_sentence_features

    def dynamic_mask_with_coords(self, mask_features, mask_head_params, reference_points, targets):
        """
        Add the relative coordinates to the mask_features channel dimension,
        and perform dynamic mask conv.

        Args:
            mask_features: [batch_size, time, c, h, w]
            mask_head_params: [batch_size, time * num_queries_per_frame, num_params]
            reference_points: [batch_size, time * num_queries_per_frame, 2], cxcy
            targets (list[dict]): length is batch size
                we need the key 'size' for computing location.
        Return:
            outputs_seg_mask: [batch_size, time * num_queries_per_frame, h, w]
        """
        device = mask_features.device
        b, t, c, h, w = mask_features.shape
        # this is the total query number in all frames
        _, num_queries = reference_points.shape[:2]
        q = num_queries // t  # num_queries_per_frame

        # prepare reference points in image size (the size is input size to the model)
        new_reference_points = []
        for i in range(b):
            img_h, img_w = targets[i]['size']
            scale_f = torch.stack([img_w, img_h], dim=0)
            tmp_reference_points = reference_points[i] * scale_f[None, :]
            new_reference_points.append(tmp_reference_points)
        new_reference_points = torch.stack(new_reference_points, dim=0)
        # [batch_size, time * num_queries_per_frame, 2], in image size
        reference_points = new_reference_points

        # prepare the mask features
        if self.rel_coord:
            reference_points = rearrange(reference_points, 'b (t q) n -> b t q n', t=t, q=q)
            locations = compute_locations(h, w, device=device, stride=self.mask_feat_stride)
            relative_coords = reference_points.reshape(b, t, q, 1, 1, 2) - \
                              locations.reshape(1, 1, 1, h, w, 2)  # [batch_size, time, num_queries_per_frame, h, w, 2]
            relative_coords = relative_coords.permute(0, 1, 2, 5, 3,
                                                      4)  # [batch_size, time, num_queries_per_frame, 2, h, w]

            # concat features
            mask_features = repeat(mask_features, 'b t c h w -> b t q c h w',
                                   q=q)  # [batch_size, time, num_queries_per_frame, c, h, w]
            mask_features = torch.cat([mask_features, relative_coords], dim=3)
        else:
            mask_features = repeat(mask_features, 'b t c h w -> b t q c h w',
                                   q=q)  # [batch_size, time, num_queries_per_frame, c, h, w]
        mask_features = mask_features.reshape(1, -1, h, w)

        # parse dynamic params
        mask_head_params = mask_head_params.flatten(0, 1)
        weights, biases = parse_dynamic_params(
            mask_head_params, self.dynamic_mask_channels,
            self.weight_nums, self.bias_nums
        )

        # dynamic mask conv
        mask_logits = self.mask_heads_forward(mask_features, weights, biases, mask_head_params.shape[0])
        mask_logits = mask_logits.reshape(-1, 1, h, w)

        # upsample predicted masks
        assert self.mask_feat_stride >= self.mask_out_stride
        assert self.mask_feat_stride % self.mask_out_stride == 0

        mask_logits = aligned_bilinear(mask_logits, int(self.mask_feat_stride / self.mask_out_stride))
        mask_logits = mask_logits.reshape(b, num_queries, mask_logits.shape[-2], mask_logits.shape[-1])

        return mask_logits  # [batch_size, time * num_queries_per_frame, h, w]

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits

def beit_transforms():
    image_size = 224
    return T.Compose([T.ToTensor(),
            T.Resize((image_size, image_size), interpolation=3),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                      ])

def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


def compute_locations(h, w, device, stride=1):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device)

    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class RobertaPoolout(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def build(args):
    # if args.binary:
    #     num_classes = 1
    # else:
    #     if args.dataset_file == 'ytvos':
    #         num_classes = 65
    #     elif args.dataset_file == 'davis':
    #         num_classes = 78
    #     elif args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb':
    #         num_classes = 1
    #     else:
    #         num_classes = 91  # for coco
    # device = torch.device(args.device)
    # tokenizer = XLMRobertaTokenizer.from_pretrained(
    #     "/home/users/fu.rong/RVOS-evf/weights/beit3_large_patch16_224.pth",
    #     padding_side="right",
    #     use_fast=False,
    # )
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     args.version,
    #     padding_side="right",
    #     use_fast=False,
    # )
    if args.inference or args.get_pretrain:
        tokenizer = AutoTokenizer.from_pretrained(
            args.version,
            padding_side="right",
            use_fast=False,
        )
    else:
        tokenizer = XLMRobertaTokenizer(args.version)
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    if args.get_pretrain:
        evf_model_args = {
            # "vision_pretrained": args.vision_pretrained,
            # "encoder_pretrained": args.encoder_pretrained,
            # "dice_loss_weight": args.dice_loss_weight,
            # "bce_loss_weight": args.bce_loss_weight,
            "train_mask_decoder": args.train_mask_decoder,
            "train_prompt_encoder": args.train_prompt_encoder
            # "sam_scale": args.sam_scale,
            # "mm_extractor_scale": args.mm_extractor_scale
        }
    else:
        evf_model_args = {
            "vision_pretrained": args.vision_pretrained,
            "encoder_pretrained": args.encoder_pretrained,
            # "dice_loss_weight": args.dice_loss_weight,
            # "bce_loss_weight": args.bce_loss_weight,
            "train_mask_decoder": args.train_mask_decoder,
            "train_prompt_encoder": args.train_prompt_encoder
            # "sam_scale": args.sam_scale,
            # "mm_extractor_scale": args.mm_extractor_scale
        }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half


    # evf_model = EvfSamModel.from_pretrained(
    #     args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **evf_model_args
    # )
    if args.inference:
        evf_model = EvfSamModel.from_pretrained(args.version,torch_dtype=torch_dtype)
    else:
        if args.get_pretrain:
            # evf_config = EvfConfig.from_pretrained(args.version, hidden_size=args.hidden_size, torch_dtype=torch_dtype,
            #                                        low_cpu_mem_usage=True, **evf_model_args)
            # evf_model = EvfSamModel(evf_config)
            evf_model = EvfSamModel.from_pretrained(args.version, torch_dtype=torch_dtype)
            # if args.train_mask_decoder:
            #     evf_model.visual_model.mask_decoder.train()
            # evf_model.text_hidden_fcs.train()
            # evf_model.initialize_pretrain_evf_modules(evf_model.config)
            evf_model.config.eos_token_id = tokenizer.eos_token_id
            evf_model.config.bos_token_id = tokenizer.bos_token_id
            evf_model.config.pad_token_id = tokenizer.pad_token_id
        else:
            evf_config = EvfConfig(hidden_size=args.hidden_size, torch_dtype=torch_dtype)
            evf_model = EvfSamModel(evf_config, **evf_model_args)
            # evf_model.initialize_evf_modules(evf_model.config)
            evf_model.config.eos_token_id = tokenizer.eos_token_id
            evf_model.config.bos_token_id = tokenizer.bos_token_id
            evf_model.config.pad_token_id = tokenizer.pad_token_id



    # evf_model.gradient_checkpointing_enable()

    # evf_model.get_model().initialize_vision_modules(evf_model.get_model().config)
    # if not args.eval_only:
    #     evf_model.get_model().initialize_evf_modules(evf_model.get_model().config)
    #
    # conversation_lib.default_conversation = conversation_lib.conv_templates[
    #     args.conv_type
    # ]
    # evf_model.print_trainable_parameters()
    # evf_model.resize_token_embeddings(len(tokenizer))
    # evf_model.initialize_evf_modules(evf_model.config)
    for n, p in evf_model.named_parameters():
        if any(
                [
                    x in n
                    for x in ["mm_extractor", "prompt_encoder", "mask_decoder", "text_hidden_fcs"]
                ]
        ):
            # print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True
    # for parameter in tokenizer.parameters():
    #     parameter.requires_grad_(False)

    model = ReferFormer(
        evf_model,
        tokenizer,
        inference= args.inference,
        precision= args.precision,
    )
    # matcher = build_matcher(args)
    # weight_dict = {}
    # # weight_dict['loss_ce'] = args.cls_loss_coef
    # weight_dict['loss_bbox'] = args.bbox_loss_coef
    # weight_dict['loss_giou'] = args.giou_loss_coef
    # if args.masks:  # always true
    #     weight_dict['loss_mask'] = args.mask_loss_coef
    #     weight_dict['loss_dice'] = args.dice_loss_coef
    # weight_dict['loss_mask'] = args.mask_loss_coef
    # weight_dict['loss_dice'] = args.dice_loss_coef
    # weight_dict['loss_sim'] = args.sim_loss_coef
    # # TODO this is a hack
    # if args.aux_loss:
    #     aux_weight_dict = {}
    #     for i in range(args.dec_layers - 1):
    #         aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    #     weight_dict.update(aux_weight_dict)

    # losses = ['labels', 'boxes']
    # if args.masks:
    #     losses += ['masks']
    # losses =['masks','sim']
    # criterion = SetCriterion(
    #     num_classes,
    #     matcher=matcher,
    #     weight_dict=weight_dict,
    #     eos_coef=args.eos_coef,
    #     losses=losses,
    #     focal_alpha=args.focal_alpha)
    # criterion.to(device)

    # postprocessors, this is used for coco pretrain but not for rvos
    # postprocessors = build_postprocessors(args, args.dataset_file)
    # return model, criterion, postprocessors
    return model


