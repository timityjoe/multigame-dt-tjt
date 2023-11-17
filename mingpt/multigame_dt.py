from typing import Mapping, Optional, Tuple

import numpy as np
import scipy
import torch
import torch.nn as nn
from torch import Tensor
import cv2

from mingpt.multigame_dt_utils import (
    accuracy,
    autoregressive_generate,
    cross_entropy,
    decode_return,
    encode_return,
    encode_reward,
    sample_from_logits,
    variance_scaling_,
)

from mingpt.visualize_attention import visualize_attn_np, attention_patches_mean, attention_layers_mean, visualize_attn_heatmap, visualize_attn_tensor, visualize_attn_tensor_detach
from mingpt.visualize_attention2 import visualize_predict

from mingpt.transformer_dt import Transformer

from loguru import logger
# logger.remove()
# logger.add(sys.stdout, level="INFO")
# logger.add(sys.stdout, level="SUCCESS")
# logger.add(sys.stdout, level="WARNING")


#------------------------------------------------
class MultiGameDecisionTransformer(nn.Module):
    _np_attn_container = None
    _np_attn_mean = None
    _np_rgb_img = None

    def __init__(
        self,
        img_size: Tuple[int],
        patch_size: Tuple[int],
        num_actions: int,
        num_rewards: int,
        return_range: Tuple[int],
        d_model: int,
        num_layers: int,
        dropout_rate: float,
        predict_reward: bool,
        single_return_token: bool,
        conv_dim: int,
    ):
        super().__init__()

        logger.info(f"MultiGameDecisionTransformer:: 0) Start") 
        logger.info(f"      img_size:{img_size} patch_size:{patch_size} num_actions:{num_actions} num_rewards:{num_rewards} return_range:{return_range}")
        logger.info(f"      d_model:{d_model} num_layers:{num_layers} dropout_rate:{dropout_rate} conv_dim:{conv_dim}")

        # Expected by the transformer model.
        if d_model % 64 != 0:
            raise ValueError(f"Model size {d_model} must be divisible by 64")

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_actions = num_actions
        self.num_rewards = num_rewards
        self.num_returns = return_range[1] - return_range[0]
        self.return_range = return_range
        self.d_model = d_model
        self.predict_reward = predict_reward
        self.conv_dim = conv_dim
        self.single_return_token = single_return_token
        self.spatial_tokens = True

        transformer_num_heads = self.d_model // 64

        logger.info(f"MultiGameDecisionTransformer:: 1) Create transformer - transformer_num_heads:{transformer_num_heads}") 
        self.transformer = Transformer(
            embed_dim=self.d_model,
            num_heads=self.d_model // 64,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )

        patch_height, patch_width = self.patch_size[0], self.patch_size[1]
        # If img_size=(84, 84), patch_size=(14, 14), then P = 84 / 14 = 6.

        logger.info(f"MultiGameDecisionTransformer:: 2) Image embedding") 
        self.image_emb = nn.Conv2d(
            in_channels=1,
            out_channels=self.d_model,
            kernel_size=(patch_height, patch_width),
            stride=(patch_height, patch_width),
            padding="valid",
        )  # image_emb is now [BT x D x P x P].
        logger.info(f"    type(self.image_emb):{type(self.image_emb)}")

        patch_grid = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        num_patches = patch_grid[0] * patch_grid[1]

        logger.info(f"MultiGameDecisionTransformer:: 3) num_patches:{num_patches}, patch_grid[0]:{patch_grid[0]}, patch_grid[1]:{patch_grid[1]}") 
        # MultiGameDecisionTransformer:: 3) num_patches:36, patch_grid[0]:6, patch_grid[1]:6

        self.image_pos_enc = nn.Parameter(torch.randn(1, 1, num_patches, self.d_model))

        self.ret_emb = nn.Embedding(self.num_returns, self.d_model)
        self.act_emb = nn.Embedding(self.num_actions, self.d_model)
        if self.predict_reward:
            self.rew_emb = nn.Embedding(self.num_rewards, self.d_model)

        num_steps = 4
        num_obs_tokens = num_patches if self.spatial_tokens else 1
        if self.predict_reward:
            tokens_per_step = num_obs_tokens + 3
        else:
            tokens_per_step = num_obs_tokens + 2
        self.positional_embedding = nn.Parameter(torch.randn(tokens_per_step * num_steps, self.d_model))

        self.ret_linear = nn.Linear(self.d_model, self.num_returns)
        self.act_linear = nn.Linear(self.d_model, self.num_actions)
        logger.info(f"ret_linear num_returns:{self.num_returns} ")  # 120
        logger.info(f"act_linear num_actions:{self.num_actions} ")  # 18

        if self.predict_reward:
            self.rew_linear = nn.Linear(self.d_model, self.num_rewards)

        # Mod by Tim: Init the array
        # self._np_attn_container = np.empty((156, 156, 3, transformer_num_heads)) 
        self._np_attn_container = np.empty((transformer_num_heads, 156, 156, 3))    
        self._np_attn_mean = np.empty((156, 156, 3)) 

    #------------------------------------------
    # Mod by Tim: Retrieve attention maps
    # There are 10 layers, each with 20 attention heads (represented by a singular mean)
    # So perform another mean across the 10 layers
    def get_attention_map(self):
        layers = self.transformer.layers
        # logger.info(f"get_attention_map:: len(layers):{len(layers)}, type:{type(layers)}")

        for i, layer in enumerate(layers):
            if layer.attn._np_attn_mean is None:
                # logger.info(f"  Layer:{i} Not initialized; type(_np_attn_mean):{type(layer.attn._np_attn_mean)}, returning...")
                continue
            self._np_attn_container[i] = layer.attn._np_attn_mean
    
    def get_last_selfattention(self, x):
        # x = self.prepare_tokens(x)
        # for i, blk in enumerate(self.blocks):
        for i, blk in enumerate(self.transformer.layers):
            if i < len(self.transformer.layers) - 1:
                # x = blk(x)
                pass
            else:
                # return attention of the last block
                # return blk(x, return_attention=True)
                value =  self.transformer.layers[i].attn
                # value = blk._attn
                logger.info(f"get_last_selfattention() - type:{type(value)}")
                logger.info(f"get_last_selfattention() - shape:{value.shape}")
                return value

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(
                math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(
            w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)     

    
    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        print(f"prepare_tokens - x:{x.shape}, B:{B}, nc:{nc}, w:{w}, h:{h}")

        # x = self.patch_embed(x)  # patch linear embedding
        # x = self.image_emb(x)  # patch linear embedding
        # print(f"    (1)x:{x.shape}")

        # add the [CLS] token to the embed patch tokens
        # cls_tokens = self.cls_token.expand(B, -1, -1)
        # print(f"    cls_tokens.shape:{cls_tokens.shape}")
        # x = torch.cat((cls_tokens, x), dim=1)
        # print(f"    (2)x:{x.shape}")

        # add positional encoding to each token
        # x = x + self.interpolate_pos_encoding(x, w, h)
        # print(f"    (3)x:{x.shape}")

        # value = self.pos_drop(x)
        # print(f"    self.pos_drop(x).shape:{value.shape}")
        return x
            

    def update_game_image(self, img):
        # logger.info(f"update_game_image() - img.shape:{img.shape}")  # img.shape:(210, 160, 3)
        self._np_rgb_img = img
        self._np_rgb_img = self._np_rgb_img * 255.
        self._np_rgb_img = cv2.applyColorMap(self._np_rgb_img.astype(np.uint8), cv2.COLORMAP_INFERNO )
        img = self._np_rgb_img
        # logger.info(f"type:{type(self._np_rgb_img)}")

        # *With attention map
        # Adjusted in Transformer.forward(), L86
        render_layer_id = self.transformer._render_layer_id
        # render_attn_head_id = self.transformer._render_attn_head_id
        np_heatmap = self.transformer.layers[render_layer_id].attn._np_attn_heatmap
        img = cv2.addWeighted(self._np_rgb_img, 0.5, np_heatmap, 0.5, 0)

        cv2.imshow(f"img", img)
        cv2.waitKey(1) 
        # cv2.destroyAllWindows()


    #------------------------------------------

    def reset_parameters(self):
        nn.init.trunc_normal_(self.image_emb.weight, std=0.02)
        nn.init.zeros_(self.image_emb.bias)
        nn.init.normal_(self.image_pos_enc, std=0.02)

        nn.init.trunc_normal_(self.ret_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.act_emb.weight, std=0.02)
        if self.predict_reward:
            nn.init.trunc_normal_(self.rew_emb.weight, std=0.02)

        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        variance_scaling_(self.ret_linear.weight)
        nn.init.zeros_(self.ret_linear.bias)
        variance_scaling_(self.act_linear.weight)
        nn.init.zeros_(self.act_linear.bias)
        if self.predict_reward:
            variance_scaling_(self.rew_linear.weight)
            nn.init.zeros_(self.rew_linear.bias)

    def _image_embedding(self, image: Tensor):
        r"""Embed [B x T x C x W x H] images to tokens [B x T x output_dim] tokens.

        Args:
            image: [B x T x C x W x H] image to embed.

        Returns:
            Image embedding of shape [B x T x output_dim] or [B x T x _ x output_dim].
        """
        assert len(image.shape) == 5
        # logger.info(f"_image_embedding() - image:{image.shape}")  # image:torch.Size([1, 4, 1, 84, 84])

        image_dims = image.shape[-3:]
        batch_dims = image.shape[:2]
        # logger.info(f"image_dims:{image_dims} batch_dims:{batch_dims}") 
        # image_dims:torch.Size([1, 84, 84]) 
        # batch_dims:torch.Size([1, 4])

        # Mod by Tim: Visualize the incoming images patches (84 x 84). These seem to be the 
        # B = Batch of num_envs
        # T = 4 per env
        # C - Color channel (1)
        # np_image = image.cpu().numpy()
        # np_image = np.mean(np_image, axis=0)
        # np_image = np.mean(np_image, axis=0)
        # visualize_attn_tensor(torch.from_numpy(np_image))

        # Reshape to [BT x C x H x W].
        # logger.info(f"(1)image.shape:{image.shape}")   # image.shape:torch.Size([1, 4, 1, 84, 84])
        image = torch.reshape(image, (-1,) + image_dims) 
        # logger.info(f"(2)image.shape:{image.shape}")   # image.shape:torch.Size([4, 1, 84, 84])

        # np_image = image.cpu().numpy()
        # np_image = np.mean(np_image, axis=0)
        # visualize_attn_tensor(torch.from_numpy(np_image))

        # Perform any-image specific processing.
        image = image.to(dtype=torch.float32) / 255.0
        # logger.info(f"(3)image.shape:{image.shape}")    # image.shape:torch.Size([4, 1, 84, 84])

        image_emb = self.image_emb(image)  # [BT x D x P x P] 
        # logger.info(f"(4)image_emb.shape:{image_emb.shape}")    # image_emb.shape:torch.Size([4, 1280, 6, 6])

        # haiku.Conv2D is channel-last, so permute before reshape below for consistency
        # [BT x P x P x D] 
        image_emb = image_emb.permute(0, 2, 3, 1)  
        # See L411, nn.Conv2d()
        # logger.info(f"(5)image_emb.shape:{image_emb.shape}")    # image_emb.shape:torch.Size([4, 6, 6, 1280])
        

        # Reshape to [B x T x P*P x D].
        image_emb = torch.reshape(image_emb, batch_dims + (-1, self.d_model))
        # logger.info(f"(6)image_emb.shape:{image_emb.shape}")    # image_emb.shape:torch.Size([1, 4, 36, 1280])

        image_emb = image_emb + self.image_pos_enc  
        # logger.info(f"(7)image_emb.shape:{image_emb.shape}")    # image_emb:torch.Size([1, 4, 36, 1280])

        return image_emb

    def _embed_inputs(self, obs: Tensor, ret: Tensor, act: Tensor, rew: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Embed only prefix_frames first observations.
        # obs are [B x T x C x H x W].
        # logger.info(f"_embed_inputs()")
        # logger.info(f"  (1)obs.shape:{obs.shape}")  # [B x T x C x W x H] - obs.shape:torch.Size([1, 4, 1, 84, 84])
        obs_emb = self._image_embedding(obs)
        # logger.info(f"  (2)obs_emb.shape:{obs_emb.shape}")  # [B x T x P*P x D] - obs_emb.shape:torch.Size([1, 4, 36, 1280])

        # Embed returns and actions
        # Encode returns.
        ret = encode_return(ret, self.return_range)
        rew = encode_reward(rew)
        ret_emb = self.ret_emb(ret)
        act_emb = self.act_emb(act)
        if self.predict_reward:
            rew_emb = self.rew_emb(rew)
        else:
            rew_emb = None
        return obs_emb, ret_emb, act_emb, rew_emb

    # Mod by Tim:
    # Note in subsequent forward passes, "token_emb" becomes "x"
    def forward(self, inputs: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        r"""Process sequence."""
        # logger.info(f"forward() - inputs:{inputs}")

        num_batch = inputs["actions"].shape[0]
        num_steps = inputs["actions"].shape[1] 
        # input_actions = inputs["actions"]
        # logger.info(f"forward() - input_actions:{input_actions.shape}")               # input_actions:torch.Size([1, 4])
        # input_observations = inputs["observations"]
        # logger.info(f"forward() - input_observations:{input_observations.shape}")     # input_observations:torch.Size([1, 4, 1, 84, 84])
        # input_RTG = inputs["returns-to-go"]
        # logger.info(f"  input_RTG:{input_RTG.shape}")                                 # input_RTG:torch.Size([1, 4])
        # input_rewards = inputs["rewards"]
        # logger.info(f"  input_rewards:{input_rewards.shape}")                         # input_rewards:torch.Size([1, 4])


        # Embed inputs.
        obs_emb, ret_emb, act_emb, rew_emb = self._embed_inputs(
            inputs["observations"],
            inputs["returns-to-go"],
            inputs["actions"],
            inputs["rewards"],
        )
        device = obs_emb.device

        # logger.info(f"forward() ")
        # logger.info(f"  obs_emb:{obs_emb.shape}")   # [B x T x P*P x D] - obs_emb:torch.Size([1, 4, 36, 1280])
        # logger.info(f"  ret_emb:{ret_emb.shape}")   # ([1, 4, 1280])
        # logger.info(f"  act_emb:{act_emb.shape}")   # ([1, 4, 1280])
        # logger.info(f"  rew_emb:{rew_emb.shape}")   # ([1, 4, 1280])

        if self.spatial_tokens:
            # obs is [B x T x W x D]
            num_obs_tokens = obs_emb.shape[2]
            # logger.info(f"  num_obs_tokens:{num_obs_tokens}")   # 36

            obs_emb = torch.reshape(obs_emb, obs_emb.shape[:2] + (-1,))
            # logger.info(f"  obs_emb:{obs_emb.shape}")   # obs_emb:torch.Size([1, 4, 46080])
            # obs is [B x T x W*D]
        else:
            num_obs_tokens = 1

        # logger.info(f"forward() self.spatial_tokens:{self.spatial_tokens} num_obs_tokens:{num_obs_tokens}")
        # forward() self.spatial_tokens:True num_obs_tokens:36
        

        # Collect sequence.
        # Embeddings are [B x T x D].
        if self.predict_reward:
            token_emb = torch.cat([obs_emb, ret_emb, act_emb, rew_emb], dim=-1)
            # logger.info(f"  token_emb:{token_emb.shape}")   # token_emb:torch.Size([1, 4, 49920]), bcos 46080+1280+1280+1280 = 49920

            tokens_per_step = num_obs_tokens + 3
            # tokens_per_step is 36+3=39
            # sequence is [obs ret act rew ... obs ret act rew]
        else:
            token_emb = torch.cat([obs_emb, ret_emb, act_emb], dim=-1)
            tokens_per_step = num_obs_tokens + 2
            # sequence is [obs ret act ... obs ret act]

        # logger.info(f"  (1)token_emb:{token_emb.shape}, tokens_per_step:{tokens_per_step}")   
        # token_emb:torch.Size([1, 4, 49920]), tokens_per_step:36+3=39

        token_emb = torch.reshape(token_emb, [num_batch, tokens_per_step * num_steps, self.d_model])
        # logger.info(f"  (2)token_emb:{token_emb.shape}")   # token_emb:torch.Size([1, 156, 1280])
        # Note:Tokens_per_step(39)*4 = 156
        # So (4*49920)/156 = 1280

        # Create position embeddings.
        token_emb = token_emb + self.positional_embedding
        # logger.info(f"  (3)token_emb:{token_emb.shape}") # token_emb:torch.Size([1, 156, 1280])

        # Run the transformer over the inputs.
        # Token dropout.
        batch_size = token_emb.shape[0]
        obs_mask = np.ones([batch_size, num_steps, num_obs_tokens], dtype=bool)
        # logger.info(f"  obs_mask:{obs_mask.shape}")     # obs_mask:(1, 4, 36)

        ret_mask = np.ones([batch_size, num_steps, 1], dtype=bool)
        act_mask = np.ones([batch_size, num_steps, 1], dtype=bool)
        rew_mask = np.ones([batch_size, num_steps, 1], dtype=bool)
        if self.single_return_token:
            # Mask out all return tokens expect the first one.
            ret_mask[:, 1:] = 0
        if self.predict_reward:
            mask = [obs_mask, ret_mask, act_mask, rew_mask]
            # logger.info(f"  predict_reward - yes")
        else:
            mask = [obs_mask, ret_mask, act_mask]
            # logger.info(f"  predict_reward - no")

        mask = np.concatenate(mask, axis=-1)    
        # logger.info(f"  (1)mask:{mask.shape}") # mask:(1, 4, 39)

        mask = np.reshape(mask, [batch_size, tokens_per_step * num_steps])
        # mask:(1, 156), batch_size:1, tokens_per_step:39, num_steps:4
        # logger.info(f"  (2)mask:{mask.shape}, batch_size:{batch_size}, tokens_per_step:{tokens_per_step}, num_steps:{num_steps}") # 

        mask = torch.tensor(mask, dtype=torch.bool, device=device)
        # logger.info(f"  mask:{mask.shape}")     # mask:torch.Size([1, 156])

        custom_causal_mask = None
        if self.spatial_tokens:
            # Temporal transformer by default assumes sequential causal relation.
            # This makes the transformer causal mask a lower triangular matrix.
            #     P1 P2 R  a  P1 P2 ... (Ps: image patches)
            # P1  1  0* 0  0  0  0
            # P2  1  1  0  0  0  0
            # R   1  1  1  0  0  0
            # a   1  1  1  1  0  0
            # P1  1  1  1  1  1  0*
            # P2  1  1  1  1  1  1
            # ... (0*s should be replaced with 1s in the ideal case)
            # But, when we have multiple tokens for an image (e.g. patch tokens, conv
            # feature map tokens, etc) as inputs to transformer, this assumption does
            # not hold, because there is no sequential dependencies between tokens.
            # Therefore, the ideal causal mask should not mask out tokens that belong
            # to the same images from each others.
            seq_len = token_emb.shape[1]
            sequential_causal_mask = np.tril(np.ones((seq_len, seq_len)))
            num_timesteps = seq_len // tokens_per_step
            num_non_obs_tokens = tokens_per_step - num_obs_tokens
            # logger.info(f"  seq_len:{seq_len}, num_timesteps:{num_timesteps}, num_non_obs_tokens:{num_non_obs_tokens}")
            # seq_len:156, num_timesteps:4, num_non_obs_tokens:3

            diag = [
                np.ones((num_obs_tokens, num_obs_tokens)) if i % 2 == 0 else np.zeros((num_non_obs_tokens, num_non_obs_tokens))
                for i in range(num_timesteps * 2)
            ]
            block_diag = scipy.linalg.block_diag(*diag)
            # logger.info(f"  block_diag:{block_diag.shape} ")     # block_diag:(156, 156) 

            custom_causal_mask = np.logical_or(sequential_causal_mask, block_diag)
            # logger.info(f"  (1)custom_causal_mask.shape:{custom_causal_mask.shape} ")    # custom_causal_mask.shape:(156, 156)

            custom_causal_mask = torch.tensor(custom_causal_mask, dtype=torch.bool, device=device)
            # logger.info(f"  (2)custom_causal_mask.shape:{custom_causal_mask.shape} ")   # custom_causal_mask.shape:torch.Size([156, 156]) 
 
        # logger.info(f"  token_emb.shape:{token_emb.shape} ")                    # token_emb.shape:torch.Size([1, 156, 1280])
        # logger.info(f"  mask.shape:{mask.shape} ")                              # mask.shape:torch.Size([1, 156])
        # logger.info(f"  custom_causal_mask.shape:{custom_causal_mask.shape} ")  # custom_causal_mask.shape:torch.Size([156, 156])

        output_emb = self.transformer(token_emb, mask, custom_causal_mask)      # Transformer.forward(1) called here

        # logger.info(f"  output_emb:{output_emb.shape} ")    # output_emb:torch.Size([1, 156, 1280]) 

        # Output_embeddings are [B x 3T x D].
        # Next token predictions (tokens one before their actual place).
        ret_pred = output_emb[:, (num_obs_tokens - 1) :: tokens_per_step, :]
        # logger.info(f"  (1)ret_pred.shape:{ret_pred.shape} ")      # ret_pred.shape:torch.Size([1, 4, 1280]) 
        act_pred = output_emb[:, (num_obs_tokens - 0) :: tokens_per_step, :]
        # logger.info(f"  (1)act_pred.shape:{act_pred.shape} ")      # act_pred.shape:torch.Size([1, 4, 1280])
        embeds = torch.cat([ret_pred, act_pred], dim=-1)
        # logger.info(f"  embeds.shape:{embeds.shape} ")      # embeds.shape:torch.Size([1, 4, 2560]) Note: 2560/1280 = 2

        # Project to appropriate dimensionality.
        # See L441; nn.Linear()
        ret_pred = self.ret_linear(ret_pred)
        # logger.info(f"  (2)ret_pred.shape:{ret_pred.shape} ")   
        # ret_pred.shape:torch.Size([1, 4, 120]) See run_atari.py L114 self.num_returns; RETURN_RANGE = [-20, 100]
        act_pred = self.act_linear(act_pred)
        # logger.info(f"  (2)act_pred.shape:{act_pred.shape} ") 
        # act_pred.shape:torch.Size([1, 4, 18])  See run_atari.py L111 self.num_actions; NUM_ACTIONS=18  

        # Return logits as well as pre-logits embedding.
        result_dict = {
            "embeds": embeds,
            "action_logits": act_pred,
            "return_logits": ret_pred,
        }
        if self.predict_reward:
            rew_pred = output_emb[:, (num_obs_tokens + 1) :: tokens_per_step, :]
            rew_pred = self.rew_linear(rew_pred)
            result_dict["reward_logits"] = rew_pred
        # Return evaluation metrics.
        result_dict["loss"] = self.sequence_loss(inputs, result_dict)
        result_dict["accuracy"] = self.sequence_accuracy(inputs, result_dict)
        return result_dict

    def _objective_pairs(self, inputs: Mapping[str, Tensor], model_outputs: Mapping[str, Tensor]) -> Tensor:
        r"""Get logit-target pairs for the model objective terms."""
        act_target = inputs["actions"]
        ret_target = encode_return(inputs["returns-to-go"], self.return_range)
        act_logits = model_outputs["action_logits"]
        ret_logits = model_outputs["return_logits"]
        # logger.info(f"  act_logits:{act_logits.shape} ret_logits:{ret_logits.shape}")

        if self.single_return_token:
            ret_target = ret_target[:, :1]
            ret_logits = ret_logits[:, :1, :]
        obj_pairs = [(act_logits, act_target), (ret_logits, ret_target)]
        if self.predict_reward:
            rew_target = encode_reward(inputs["rewards"])
            rew_logits = model_outputs["reward_logits"]
            obj_pairs.append((rew_logits, rew_target))
        return obj_pairs

    def sequence_loss(self, inputs: Mapping[str, Tensor], model_outputs: Mapping[str, Tensor]) -> Tensor:
        r"""Compute the loss on data wrt model outputs."""
        obj_pairs = self._objective_pairs(inputs, model_outputs)
        obj = [cross_entropy(logits, target) for logits, target in obj_pairs]
        return sum(obj) / len(obj)

    def sequence_accuracy(self, inputs: Mapping[str, Tensor], model_outputs: Mapping[str, Tensor]) -> Tensor:
        r"""Compute the accuracy on data wrt model outputs."""
        obj_pairs = self._objective_pairs(inputs, model_outputs)
        obj = [accuracy(logits, target) for logits, target in obj_pairs]
        return sum(obj) / len(obj)

    def optimal_action(
        self,
        inputs: Mapping[str, Tensor],
        return_range: Tuple[int] = (-100, 100),
        single_return_token: bool = False,
        opt_weight: Optional[float] = 0.0,
        num_samples: Optional[int] = 128,
        action_temperature: Optional[float] = 1.0,
        return_temperature: Optional[float] = 1.0,
        action_top_percentile: Optional[float] = None,
        return_top_percentile: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
        deterministic: bool = False,
        torch_device: torch.device = None
    ):
        r"""Calculate optimal action for the given sequence model."""
        logits_fn = self.forward
        # logger.info(f"logits_fn:{logits_fn} ")


        obs, act, rew = inputs["observations"], inputs["actions"], inputs["rewards"]
        # logger.info(f"obs.shape:{obs.shape} act:{act} rew:{rew}")
        # obs.shape:torch.Size([2, 4, 1, 84, 84]) act:tensor([[1, 1, 1, 0],
        # [0, 0, 4, 0]], device='cuda:0') rew:tensor([[0., 0., 0., 0.],
        # [0., 0., 0., 0.]], device='cuda:0', dtype=torch.float64)
        

        assert len(obs.shape) == 5
        assert len(act.shape) == 2
        inputs = {
            "observations": obs,
            "actions": act,
            "rewards": rew,
            "returns-to-go": torch.zeros_like(act),
        }
        sequence_length = obs.shape[1]
        # Use samples from the last timestep.
        timestep = -1
        # A biased sampling function that prefers sampling larger returns.
        def ret_sample_fn(rng, logits):
            assert len(logits.shape) == 2
            # Add optimality bias.
            if opt_weight > 0.0:
                # Calculate log of P(optimality=1|return) := exp(return) / Z.
                logits_opt = torch.linspace(0.0, 1.0, logits.shape[1])
                logits_opt = torch.repeat_interleave(logits_opt[None, :], logits.shape[0], dim=0)
                # Sample from log[P(optimality=1|return)*P(return)].
                logits = logits + opt_weight * logits_opt
            logits = torch.repeat_interleave(logits[None, ...], num_samples, dim=0)
            ret_sample = sample_from_logits(
                logits,
                generator=rng,
                deterministic=deterministic,
                temperature=return_temperature,
                top_percentile=return_top_percentile,
            )

            # logger.info(f"logits:{logits} ")
            # logger.info(f"len(logits):{len(logits)} type:{type(logits)} ")  #len logits 128
            # logger.info(f"logits.shape:{logits.shape} ") # logits.shape:torch.Size([128, 2, 120])

            # Pick the highest return sample.
            ret_sample, _ = torch.max(ret_sample, dim=0)
            # Convert return tokens into return values.
            ret_sample = decode_return(ret_sample, return_range)
            return ret_sample

        # Set returns-to-go with an (optimistic) autoregressive sample.
        if single_return_token:
            # Since only first return is used by the model, only sample that (faster).
            ret_logits = logits_fn(inputs)["return_logits"][:, 0, :]        # MultiGameDecisionTransformer.forward(0) called here
            # logger.info(f"  ret_logits.shape:{ret_logits.shape}")   # ret_logits.shape:torch.Size([1, 120])

            ret_sample = ret_sample_fn(rng, ret_logits)
            inputs["returns-to-go"][:, 0] = ret_sample
        else:
            # Auto-regressively regenerate all return tokens in a sequence.
            ret_logits_fn = lambda input: logits_fn(input)["return_logits"]
            ret_sample = autoregressive_generate(
                ret_logits_fn,
                inputs,
                "returns-to-go",
                sequence_length,
                generator=rng,
                deterministic=deterministic,
                sample_fn=ret_sample_fn,
            )
            inputs["returns-to-go"] = ret_sample

        # --- Extract attention map(s)
        # logger.info(f"  X) Get Attention Map(s)")
        # self.get_attention_map()
        # self._np_attn_mean = attention_layers_mean(self._np_attn_container)

        # --- Visualize Attention
        # if self._np_rgb_img is not None:
        #     visualize_predict(self, self._np_rgb_img, self.img_size, self.patch_size, torch_device)


        # Generate a sample from action logits.
        act_logits = logits_fn(inputs)["action_logits"][:, timestep, :]
        act_sample = sample_from_logits(
            act_logits,
            generator=rng,
            deterministic=deterministic,
            temperature=action_temperature,
            top_percentile=action_top_percentile,
        )
        # logger.info(f"len(act_sample):{len(act_sample)}, type:{type(act_sample)}")

        return act_sample
