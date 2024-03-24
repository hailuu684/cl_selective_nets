from avalanche.models.timm_vit import ViTWithPrompt
from timm.models.layers import PatchEmbed
from timm.models.vision_transformer import Block


def get_prompt_model(img_size, patch_size, num_classes, embed_dim, depth, num_heads, drop_rate,
                     kv_bias=True, init_values=None, class_token=True, no_embed_class=False, pre_norm=False,
                     fc_norm=None, attn_drop_rate=0.0, drop_path_rate=0.0, weight_init="", norm_layer=None,
                     act_layer=None, prompt_length=None, embedding_key="cls", prompt_init="uniform",
                     prompt_pool=False, prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False,
                     prompt_key_init="uniform", head_type="token", use_prompt_mask=False):

    model = ViTWithPrompt(img_size=img_size, patch_size=patch_size, in_chans=3, num_classes=num_classes,
                          global_pool="token", embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=4,
                          qkv_bias=kv_bias, init_values=init_values, class_token=class_token,
                          no_embed_class=no_embed_class, pre_norm=pre_norm,
                          fc_norm=fc_norm, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                          drop_path_rate=drop_path_rate, weight_init=weight_init,
                          embed_layer=PatchEmbed, norm_layer=norm_layer, act_layer=act_layer,
                          block_fn=Block, prompt_length=prompt_length, embedding_key=embedding_key,
                          prompt_init=prompt_init, prompt_pool=prompt_pool, prompt_key=prompt_key, pool_size=pool_size,
                          top_k=top_k, batchwise_prompt=batchwise_prompt, prompt_key_init=prompt_key_init,
                          head_type=head_type, use_prompt_mask=use_prompt_mask)

    return model
