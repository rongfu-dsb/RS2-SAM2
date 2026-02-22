from typing import Optional, List
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoConfig, AutoModelForCausalLM
# from .segment_anything import build_sam_vit_h
from .unilm.beit3.modeling_utils import BEiT3Wrapper, _get_base_config, _get_large_config
from .configuration_evf import EvfConfig
from models.sam2.build_sam import  build_sam2_video_predictor
# from .sam2.sam2_video_predictor import SAM2VideoPredictor
import os
from einops import rearrange, repeat
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        scale=1000,  # 100000.0,
        eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class VisionLanguageFusionModule(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt * tgt2
        return tgt


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class EvfSamModel(PreTrainedModel):
    config_class = EvfConfig

    def __init__(
            self,
            config,
            **kwargs
    ):
        super(EvfSamModel, self).__init__(config)

        self.config = config
        print(config)
        print(kwargs)
        self.vision_pretrained = kwargs.get("vision_pretrained", None)
        self.encoder_pretrained = kwargs.get("encoder_pretrained", None)
        # self.dice_loss_weight = kwargs.get("dice_loss_weight", None)
        # self.bce_loss_weight = kwargs.get("bce_loss_weight", None)
        self.train_mask_decoder = kwargs.get("train_mask_decoder", False)
        self.train_prompt_encoder = kwargs.get("train_prompt_encoder", False)
        self.initialize_evf_modules(config)
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    def initialize_evf_modules(self, config):
        # SAM
        # if config.sam_scale == "huge":
        #     self.visual_model = build_sam_vit_h(self.vision_pretrained)
        # else:
        #     raise NotImplementedError
        # vision_yaml= "/horizon-bucvket/saturn_v_dev/01_users/fu.rong/RVOS_evf/weights/sam2-hiera-large/sam2_hiera_l.yaml"
        self.visual_model = build_sam2_video_predictor("sam2_hiera_l.yaml", self.vision_pretrained, mode="train")
        # print(self.vision_pretrained)
        # self.visual_model = SAM2VideoPredictor.from_pretrained(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = True
        for param in self.visual_model.image_encoder.parameters():
            param.requires_grad = False
        for blk in self.visual_model.image_encoder.trunk.blocks:
            for param in blk.MLP_Adapter.parameters():
                param.requires_grad = True
            for param in blk.Space_Adapter.parameters():
                param.requires_grad = True

        # if self.train_mask_decoder:
        #     self.visual_model.mask_decoder.train()
        #     for param in self.visual_model.sam_mask_decoder.parameters():
        #         param.requires_grad = True
        # if self.train_prompt_encoder:
        #     self.visual_model.sam_prompt_encoder.no_mask_embed.requires_grad_(True)

        # beit-3
        if self.config.mm_extractor_scale == "base":
            beit_config = _get_base_config()
        elif self.config.mm_extractor_scale == "large":
            beit_config = _get_large_config()
        else:
            raise AttributeError(f"model config should contain key 'mm_extractor_scale', with value 'base' or 'large'.")

        self.mm_extractor = BEiT3Wrapper(beit_config)
        if self.encoder_pretrained is not None:
            beit_state_dict = torch.load(self.encoder_pretrained)["model"]
            self.mm_extractor.load_state_dict(
                beit_state_dict,
                strict=False
            )
        for param in self.mm_extractor.parameters():
            param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        assert in_dim == beit_config.encoder_embed_dim, \
            f"projection layer dim {in_dim} mismatch with mm_extractor dim {beit_config.encoder_embed_dim}"
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        self.text_feat_linear = nn.Linear(1024, 256)
        # self.visual_feat_linear = nn.Linear(1024, 256)
        for param in self.text_feat_linear.parameters():
            param.requires_grad = True
        self.mask_linear = nn.Linear(1024, 1024)
        self.mask_product = nn.Linear(1024, 1)
        self.num_vm_layer = 1
        self.vmtoken = nn.ModuleList()
        self.vmtoken = VisionLanguageFusionModule(d_model=1024, nhead=8)

        self.meo_linear = nn.Linear(1024, 256)
        self.text2one_linear = nn.Linear(256, 1)

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        """
        Perform PostProcessing on output masks.
        """
        masks = masks.float()
        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks

    def forward(
            self,
            images: torch.FloatTensor,
            images_evf: torch.FloatTensor,
            input_ids: torch.LongTensor,
            attention_masks: torch.LongTensor,
            offset: torch.LongTensor,
            # orig_size_list: List[tuple],
            # resize_list: List[tuple],
            inference: bool = False,
            **kwargs,
    ):
        # image_embeddings = self.get_visual_embs(images)
        # _features = self._image_encoder(images)
        # image_embeddings, high_res_features = _features["image_embed"], _features["high_res_feats"]
        batch_size = images.shape[0]
        # assert batch_size == len(offset) - 1




        images_evf_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            images_evf_i = (
                images_evf[i]
                .unsqueeze(0)
                .expand(end_i - start_i, -1, -1, -1)
                .contiguous()
            )
            images_evf_list.append(images_evf_i)
        images_evf = torch.cat(images_evf_list, dim=0)

        # multimask_output = False

        output = self.mm_extractor.beit3(
            visual_tokens=images_evf,
            textual_tokens=input_ids,
            text_padding_position=~attention_masks
        )
        _, len_all, _ = output["encoder_out"].shape
        _, text_len = input_ids.shape
        visual_len = len_all - text_len - 1
        visual_feat = output["encoder_out"][:, 1:visual_len + 1, ...]
        b, _, _ = visual_feat.shape
        feat = output["encoder_out"][:, :1, ...]
        text_feat = output["encoder_out"][:, len_all - text_len:len_all, ...]
        mask_token = output["encoder_out"][:, :1, ...]
        visual_feat = rearrange(visual_feat, 'b hw c -> hw b c')
        mask_token = rearrange(mask_token, 'b m c -> m b c')
        for i in range(self.num_vm_layer):

            mask_token = self.vmtoken(tgt=mask_token,
                                      memory=visual_feat,
                                      memory_key_padding_mask=None,
                                      pos=None,
                                      query_pos=None)

        mask_token = rearrange(mask_token, 'm b c -> b m c')
        visual_feat = rearrange(visual_feat, 'hw b c -> b hw c')
        mask_token = self.mask_linear(mask_token).unsqueeze(1)  # b,1,1,c

        visual_feat = rearrange(visual_feat, 'b (h w) c -> b h w c', h=14, w=14)
        mask_pred = visual_feat * mask_token
        mask_prompt = torch.sigmoid(self.mask_product(visual_feat * mask_token))  # b h w 1
        mask_prompt = rearrange(mask_prompt, 'b h w c -> b c h w')

        feat = self.text_hidden_fcs[0](feat)
        text_feat = self.text_feat_linear(text_feat)
        type = images.dtype
        # if batch_size==1:
        #     for param in self.visual_model.parameters():
        #         param.data = param.data.to(torch.float32)
        #     images = images.to(torch.float32)
        #     feat = feat.to(torch.float32)
        #     #finetune
        #     nf, _, image_h, image_w = images.shape
        #     inference_state = self.visual_model.init_state(images, image_h, image_w)
        #     for idx in range(nf):
        #         frame_idx, object_ids, masks = self.visual_model.add_new_text(inference_state=inference_state,
        #                                                                       frame_idx=idx,
        #                                                                       obj_id=1, text=feat[idx:idx+1, :, :])
        #
        #
        #     pred_masks = []
        #     for out_frame_idx, out_obj_ids, out_mask in self.visual_model.propagate_in_video(inference_state):
        #         pred_masks.append(out_mask)
        #
        #     pred_masks = torch.stack(pred_masks, dim=0).squeeze(1)
        #
        #     pred_masks = pred_masks.to(type)
            # for param in self.visual_model.parameters():
            #     param.data = param.data.to(torch.float16)
        for param in self.visual_model.parameters():
            param.data = param.data.to(torch.float32)
        #images = images.to(torch.float32)
        # feat = feat.to(torch.float32)
        backbone_out = self.visual_model.forward_image(images,text_feat)
        _, image_embeddings, _, _ = self.visual_model._prepare_backbone_features(backbone_out)
        image_embeddings = [_.to(type) for _ in image_embeddings]
        if self.visual_model.directly_add_no_mem_embed:
            image_embeddings[-1] = image_embeddings[-1] + self.visual_model.no_mem_embed

        feats = [
                    feat_img.permute(1, 2, 0).view(batch_size, -1, *feat_size)
                    for feat_img, feat_size in zip(image_embeddings[::-1], self._bb_feat_sizes[::-1])
                ][::-1]
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        if mask_prompt.shape[-2:] != self.visual_model.sam_prompt_encoder.mask_input_size:
            mask_prompt = F.interpolate(
                mask_prompt.float(),
                size=self.visual_model.sam_prompt_encoder.mask_input_size,
                align_corners=False,
                mode="bilinear",
                antialias=True,  # use antialias for downsampling
            )
            mask_prompt.to(type)
        else:
            mask_prompt = mask_prompt
        # pretrain
        (
            sparse_embeddings,
            dense_embeddings,
        ) = self.visual_model.sam_prompt_encoder(
            points=None,
            boxes=None,
            masks=mask_prompt,
            text_embeds=feat,
        )
        sparse_embeddings = sparse_embeddings.to(feat.dtype)
        high_res_features = [
            feat_level
            for feat_level in _features["high_res_feats"]
        ]
        low_res_masks, iou_predictions, _, _ = self.visual_model.sam_mask_decoder(
            image_embeddings=_features["image_embed"],
            image_pe=self.visual_model.sam_prompt_encoder.get_dense_pe(),
            text_feat=text_feat,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        # pred_masks = self.postprocess_masks(
        #     low_res_masks,
        #     orig_hw=resize_list[0],
        # )
        pred_masks = self.postprocess_masks(
            low_res_masks,
            orig_hw=(1024,1024),
        )
        pred_masks = pred_masks.float()
        sen_feat = text_feat[:, 0:1, :]
        sen_feat = self.text2one_linear(sen_feat)
        _, _, mask_h, mask_w = pred_masks.shape
        sen_feat = sen_feat.float()
        sen_feat = repeat(sen_feat, 'b n o -> b (n h) (o w)', h=mask_h, w=mask_w)


        # pred_masks = self.visual_model.postprocess_masks(
        #     low_res_masks,
        #     input_size=resize_list[0],
        #     original_size=resize_list[0],
        # )
        # for i in range(len(feat)):
        #
        #     # low_res_masks, iou_predictions = self.visual_model.mask_decoder(
        #     #     image_embeddings=image_embeddings[i].unsqueeze(0),
        #     #     image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
        #     #     sparse_prompt_embeddings=sparse_embeddings,
        #     #     dense_prompt_embeddings=dense_embeddings,
        #     #     multimask_output=multimask_output,
        #     # )
        #
        #     # if multimask_output:
        #     #     sorted_ids = torch.argsort(iou_predictions, dim=-1, descending=True)
        #     #     low_res_masks = torch.take_along_dim(low_res_masks, sorted_ids[..., None, None], dim=1)[:, :1]
        #
        #     pred_mask = self.visual_model.postprocess_masks(
        #         low_res_masks,
        #         input_size=resize_list[i],
        #         original_size=resize_list[i],
        #     )
        #     pred_masks.append(pred_mask[:, 0])

        return pred_masks,sen_feat

        # gt_masks = masks_list
        #
        # if inference:
        #     return {
        #         "pred_masks": pred_masks,
        #         "gt_masks": gt_masks,
        #     }

        # mask_bce_loss = 0
        # mask_dice_loss = 0
        # num_masks = 0
        # for batch_idx in range(len(pred_masks)):
        #     gt_mask = gt_masks[batch_idx]
        #     pred_mask = pred_masks[batch_idx]
        #
        #     assert (
        #             gt_mask.shape[0] == pred_mask.shape[0]
        #     ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
        #         gt_mask.shape, pred_mask.shape
        #     )
        #     mask_bce_loss += (
        #             sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
        #             * gt_mask.shape[0]
        #     )
        #     mask_dice_loss += (
        #             dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
        #             * gt_mask.shape[0]
        #     )
        #     num_masks += gt_mask.shape[0]
        #
        # mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        # mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        # mask_loss = mask_bce_loss + mask_dice_loss
        #
        # loss = mask_loss

        # return {
        #     "loss": loss,
        #     "mask_bce_loss": mask_bce_loss,
        #     "mask_dice_loss": mask_dice_loss,
        #     "mask_loss": mask_loss,
        # }
    def _image_encoder(self, input_image):
        backbone_out = self.sam2_model.forward_image(input_image)
        _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.sam2_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed
        bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        feats = [
            feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        return _features

    def inference(
            self,
            images,
            images_evf,
            input_ids,
            resize_list,
            original_size_list,
            multimask_output=False,
    ):
        with torch.no_grad():
            image_embeddings = self.visual_model.image_encoder(images)
        multimask_output = multimask_output

        output = self.mm_extractor.beit3(visual_tokens=images_evf, textual_tokens=input_ids,
                                         text_padding_position=torch.zeros_like(input_ids))

        feat = output["encoder_out"][:, :1, ...]
        feat = self.text_hidden_fcs[0](feat)
        (
            sparse_embeddings,
            dense_embeddings,
        ) = self.visual_model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text_embeds=feat,
        )
        sparse_embeddings = sparse_embeddings.to(feat.dtype)
        low_res_masks, iou_predictions = self.visual_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        if multimask_output:
            sorted_ids = torch.argsort(iou_predictions, dim=-1, descending=True)
            low_res_masks = torch.take_along_dim(low_res_masks, sorted_ids[..., None, None], dim=1)[:, :1]

        pred_mask = self.visual_model.postprocess_masks(
            low_res_masks,
            input_size=resize_list[0],
            original_size=original_size_list[0],
        )

        return pred_mask[:, 0]


AutoConfig.register("evf", EvfConfig)
AutoModelForCausalLM.register(EvfConfig, EvfSamModel)