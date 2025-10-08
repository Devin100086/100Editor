import math

from diffusers.utils import logging
import torch
import torch.nn.functional as F
from einops import rearrange
from flash_attn import flash_attn_func
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from copy import deepcopy   
import torch
import torch.nn.functional as F


logger = logging.get_logger(__name__)

def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False

def predict_attention(query, key, ks_q=128, ks_k=64, pool_mode="avg"):
    """
    Args:
        query: (B, nh, Tq, C)
        key: (B, nh, Tk, C)

    Return:
        pooled_prob: (B, nh, Tq, Tk)
    """
    assert pool_mode in ["max", "avg"], f"{pool_mode=}"

    pooling_fn = {
        "max": F.max_pool1d,
        "avg": F.avg_pool1d,
    }[pool_mode]

    assert query.ndim == 4, f"{query.shape=}"
    assert key.ndim == 4, f"{key.shape=}"

    B, nh, Tq, C = query.shape
    _, _, Tk, _ = key.shape

    # Query Pooling
    query = rearrange(query, "B nh Tq C -> (B nh) C Tq")
    pooled_query = pooling_fn(query, kernel_size=ks_q, ceil_mode=True)
    pooled_query = rearrange(pooled_query, "(B nh) C Tq -> B nh Tq C", B=B, nh=nh)

    # Key Pooling
    key = rearrange(key, "B nh Tk C -> (B nh) C Tk")
    pooled_key = pooling_fn(key, kernel_size=ks_k, ceil_mode=True)
    pooled_key = rearrange(pooled_key, "(B nh) C Tk -> B nh Tk C", B=B, nh=nh)

    # Dot Product
    scale = 1 / math.sqrt(C)
    pooled_score = pooled_query @ pooled_key.transpose(-1, -2) * scale  # (B, nh, Tq, Tk)
    pooled_prob = F.softmax(pooled_score, dim=-1)

    return pooled_prob

def register_sparge_attention(model):
    def sa_forward(self):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None):
            residual = hidden_states

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            args = ()
            query = self.to_q(hidden_states, *args)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states, *args)
            value = self.to_v(encoder_hidden_states, *args)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // self.heads

            query = query.view(batch_size, -1, self.heads, head_dim)

            key = key.view(batch_size, -1, self.heads, head_dim)
            value = value.view(batch_size, -1, self.heads, head_dim)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            # hidden_states = F.scaled_dot_product_attention(
            #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            # )

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = flash_attn_func(
                query, key, value, 
                dropout_p=0.0,
                softmax_scale=self.scale,
                causal=False 
            )

            hidden_states = hidden_states.reshape(batch_size, -1, self.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = self.to_out[0](hidden_states, *args)
            # dropout
            hidden_states = self.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states

        return forward

    for _, module in model.unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.attn1.forward = sa_forward(module.attn1)
            module.attn2.forward = sa_forward(module.attn2)


def warpped_feature(sample, step):
    """
    sample: batch_size*dim*h*w, uncond: 0 - batch_size//2, cond: batch_size//2 - batch_size
    step: timestep span
    """
    bs, dim, h, w = sample.shape
    uncond_fea, cond_fea1, cond_fea2 = sample.chunk(3)
    uncond_fea = uncond_fea.repeat(step,1,1,1) # (step * bs//3) * dim * h *w
    cond_fea1 = cond_fea1.repeat(step,1,1,1) # (step * bs//3) * dim * h *w
    cond_fea2 = cond_fea2.repeat(step,1,1,1) # (step * bs//3) * dim * h *w
    return torch.cat([uncond_fea, cond_fea1, cond_fea2])

def warpped_skip_feature(block_samples, step):
    down_block_res_samples = []
    for sample in block_samples:
        sample_expand = warpped_feature(sample, step)
        down_block_res_samples.append(sample_expand)
    return tuple(down_block_res_samples)

def warpped_text_emb(text_emb, step):
    """
    text_emb: batch_size*77*768, uncond: 0 - batch_size//2, cond: batch_size//2 - batch_size
    step: timestep span
    """
    bs, token_len, dim = text_emb.shape
    pos_fea, neg_fea1, neg_fea2 = text_emb.chunk(3)
    pos_fea = pos_fea.repeat(step,1,1) # (step * bs//3) * 77 *768
    neg_fea1 = neg_fea1.repeat(step,1,1) # (step * bs//3) * 77 * 768
    neg_fea2 = neg_fea2.repeat(step,1,1) # (step * bs//3) * 77 * 768
    return torch.cat([pos_fea, neg_fea1, neg_fea2]) # (step*bs) * 77 *768

def warpped_timestep(timesteps, bs):
    """
    timestpes: list, such as [981, 961, 941]
    """
    semi_bs = bs//3
    ts = []
    for timestep in timesteps:
        timestep = timestep[None]
        texp = timestep.expand(semi_bs)
        ts.append(texp)
    timesteps = torch.cat(ts)
    return timesteps.repeat(3,1).reshape(-1)

def register_faster_forward(model, mod = '50ls'):
    def faster_forward(self):
        def forward(
                sample: torch.FloatTensor,
                timestep: Union[torch.Tensor, float, int],
                encoder_hidden_states: torch.Tensor,
                class_labels: Optional[torch.Tensor] = None,
                timestep_cond: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
                mid_block_additional_residual: Optional[torch.Tensor] = None,
                return_dict: bool = True,
            ) -> Union[UNet2DConditionOutput, Tuple]:
                r"""
                Args:
                    sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
                    timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
                    encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
                    return_dict (`bool`, *optional*, defaults to `True`):
                        Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.
                    cross_attention_kwargs (`dict`, *optional*):
                        A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                        `self.processor` in
                        [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

                Returns:
                    [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                    [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
                    returning a tuple, the first element is the sample tensor.
                """
                # By default samples have to be AT least a multiple of the overall upsampling factor.
                # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
                # However, the upsampling interpolation output size can be forced to fit any upsampling size
                # on the fly if necessary.
                default_overall_up_factor = 2**self.num_upsamplers

                # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
                forward_upsample_size = False
                upsample_size = None

                if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
                    logger.info("Forward upsample size to force interpolation output size.")
                    # print("Forward upsample size to force interpolation output size.")
                    forward_upsample_size = True

                # prepare attention_mask
                if attention_mask is not None:
                    attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
                    attention_mask = attention_mask.unsqueeze(1)

                # 0. center input if necessary
                if self.config.center_input_sample:
                    sample = 2 * sample - 1.0

                # 1. time
                if isinstance(timestep, list):
                    timesteps = timestep[0]
                    step = len(timestep)
                else:
                    timesteps = timestep
                    step = 1
                if not torch.is_tensor(timesteps) and (not isinstance(timesteps,list)):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = sample.device.type == "mps"
                    if isinstance(timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
                elif (not isinstance(timesteps,list)) and len(timesteps.shape) == 0:
                    timesteps = timesteps[None].to(sample.device)
                
                if (not isinstance(timesteps,list)) and len(timesteps.shape) == 1:
                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timesteps = timesteps.expand(sample.shape[0])
                elif isinstance(timesteps, list):
                    #timesteps list, such as [981,961,941]
                    timesteps = warpped_timestep(timesteps, sample.shape[0]).to(sample.device)
                t_emb = self.time_proj(timesteps)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # but time_embedding might actually be running in fp16. so we need to cast here.
                # there might be better ways to encapsulate this.
                t_emb = t_emb.to(dtype=self.dtype)

                emb = self.time_embedding(t_emb, timestep_cond)

                if self.class_embedding is not None:
                    if class_labels is None:
                        raise ValueError("class_labels should be provided when num_class_embeds > 0")

                    if self.config.class_embed_type == "timestep":
                        class_labels = self.time_proj(class_labels)

                        # `Timesteps` does not contain any weights and will always return f32 tensors
                        # there might be better ways to encapsulate this.
                        class_labels = class_labels.to(dtype=sample.dtype)

                    class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)

                    if self.config.class_embeddings_concat:
                        emb = torch.cat([emb, class_emb], dim=-1)
                    else:
                        emb = emb + class_emb

                if self.config.addition_embed_type == "text":
                    aug_emb = self.add_embedding(encoder_hidden_states)
                    emb = emb + aug_emb

                if self.time_embed_act is not None:
                    emb = self.time_embed_act(emb)

                if self.encoder_hid_proj is not None:
                    encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

                #===============
                order = self.order #timestep, start by 0
                #===============
                ipow = int(np.sqrt(9 + 8*order))
                cond = order in [0, 1, 2, 3, 5, 10, 15, 25, 35]
                if isinstance(mod, int):
                    cond = order % mod == 0
                elif mod == "pro":
                    cond = ipow * ipow == (9 + 8 * order)
                elif mod == "50ls":
                    cond = order in [0, 1, 2, 3, 5, 10, 15] #40 #[0,1,2,3, 5, 10, 15] #[0, 1, 2, 3, 5, 10, 15, 25, 35, 40]
                if cond:
                    # print('current timestep:', order)
                    # 2. pre-process
                    sample = self.conv_in(sample)

                    # 3. down
                    down_block_res_samples = (sample,)
                    for downsample_block in self.down_blocks:
                        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                            sample, res_samples = downsample_block(
                                hidden_states=sample,
                                temb=emb,
                                encoder_hidden_states=encoder_hidden_states,
                                attention_mask=attention_mask,
                                cross_attention_kwargs=cross_attention_kwargs,
                            )
                        else:
                            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

                        down_block_res_samples += res_samples

                    if down_block_additional_residuals is not None:
                        new_down_block_res_samples = ()

                        for down_block_res_sample, down_block_additional_residual in zip(
                            down_block_res_samples, down_block_additional_residuals
                        ):
                            down_block_res_sample = down_block_res_sample + down_block_additional_residual
                            new_down_block_res_samples += (down_block_res_sample,)

                        down_block_res_samples = new_down_block_res_samples

                    # 4. mid
                    if self.mid_block is not None:
                        sample = self.mid_block(
                            sample,
                            emb,
                            encoder_hidden_states=encoder_hidden_states,
                            attention_mask=attention_mask,
                            cross_attention_kwargs=cross_attention_kwargs,
                        )

                    if mid_block_additional_residual is not None:
                        sample = sample + mid_block_additional_residual

                    #----------------------save feature-------------------------
                    # setattr(self, 'skip_feature', (tmp_sample.clone() for tmp_sample in down_block_res_samples))
                    setattr(self, 'skip_feature', deepcopy(down_block_res_samples))
                    setattr(self, 'toup_feature', sample.detach().clone())
                    #-----------------------save feature------------------------



                    #-------------------expand feature for parallel---------------
                    if isinstance(timestep, list):
                        #timesteps list, such as [981,961,941]
                        timesteps = warpped_timestep(timestep, sample.shape[0]).to(sample.device)
                        t_emb = self.time_proj(timesteps)

                        # `Timesteps` does not contain any weights and will always return f32 tensors
                        # but time_embedding might actually be running in fp16. so we need to cast here.
                        # there might be better ways to encapsulate this.
                        t_emb = t_emb.to(dtype=self.dtype)

                        emb = self.time_embedding(t_emb, timestep_cond)

                    down_block_res_samples = warpped_skip_feature(down_block_res_samples, step)
                    sample = warpped_feature(sample, step)
                    encoder_hidden_states = warpped_text_emb(encoder_hidden_states, step)
                    #-------------------expand feature for parallel---------------
                    
                else:
                    down_block_res_samples = self.skip_feature
                    sample = self.toup_feature

                    #-------------------expand feature for parallel---------------
                    down_block_res_samples = warpped_skip_feature(down_block_res_samples, step)
                    sample = warpped_feature(sample, step)
                    encoder_hidden_states = warpped_text_emb(encoder_hidden_states, step)
                    #-------------------expand feature for parallel---------------

                # 5. up
                for i, upsample_block in enumerate(self.up_blocks):
                    is_final_block = i == len(self.up_blocks) - 1

                    res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                    down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

                    # if we have not reached the final block and need to forward the
                    # upsample size, we do it here
                    if not is_final_block and forward_upsample_size:
                        upsample_size = down_block_res_samples[-1].shape[2:]

                    if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                        sample = upsample_block(
                            hidden_states=sample,
                            temb=emb,
                            res_hidden_states_tuple=res_samples,
                            encoder_hidden_states=encoder_hidden_states,
                            cross_attention_kwargs=cross_attention_kwargs,
                            upsample_size=upsample_size,
                            attention_mask=attention_mask,
                        )
                    else:
                        sample = upsample_block(
                            hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                        )

                # 6. post-process
                if self.conv_norm_out:
                    sample = self.conv_norm_out(sample)
                    sample = self.conv_act(sample)
                sample = self.conv_out(sample)

                if not return_dict:
                    return (sample,)

                return UNet2DConditionOutput(sample=sample)
        return forward
    if model.__class__.__name__ == 'UNet2DConditionModel':
        model.forward = faster_forward(model)