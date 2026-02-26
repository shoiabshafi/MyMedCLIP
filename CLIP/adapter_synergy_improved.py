import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

class SegmentationDecoder(nn.Module):
    """
    Multi-Scale Segmentation Decoder - fixes blurry upsampling
    """
    def __init__(self, num_layers=4):
        super(SegmentationDecoder, self).__init__()
        self.num_layers = num_layers
        in_channels = 2 * num_layers

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Conv2d(8, 2, kernel_size=1)

    def forward(self, multi_scale_scores):
        """
        Args: List of [B, L, 2] tensors
        Returns: [B, 2, 240, 240]
        """
        concat_scores = torch.cat(multi_scale_scores, dim=-1)
        B, L, C = concat_scores.shape
        H = W = int(np.sqrt(L))
        
        if H * W != L:
            # If not square, interpolate to 17x17 (the default for 240px with ViT-L-14)
            H = W = 17
            x = concat_scores.permute(0, 2, 1) # [B, C, L]
            # Interpolate to B, C, L_new where L_new = 289
            x = F.interpolate(x, size=H*W, mode='linear', align_corners=False)
            x = x.reshape(B, C, H, W)
        else:
            x = concat_scores.permute(0, 2, 1).reshape(B, C, H, W)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final_conv(x)
        x = x[:, :, 16:256, 16:256]

        return x


class TGVA_Improved(nn.Module):
    """Separate detection/segmentation paths"""
    def __init__(self, c_in, bottleneck=768, text_dim=768, use_additive_gating=False):
        super(TGVA_Improved, self).__init__()
        self.use_additive_gating = use_additive_gating

        self.gate_det = nn.Sequential(nn.Linear(text_dim, bottleneck), nn.Sigmoid())
        self.gate_seg = nn.Sequential(nn.Linear(text_dim, bottleneck), nn.Sigmoid())

        self.fc_down_det = nn.Linear(c_in, bottleneck, bias=False)
        self.norm_down_det = nn.InstanceNorm1d(bottleneck)

        self.fc_down_seg = nn.Linear(c_in, bottleneck, bias=False)
        self.norm_down_seg = nn.InstanceNorm1d(bottleneck)

        self.fc_up_seg = nn.Sequential(nn.Linear(bottleneck, bottleneck, bias=False), nn.LeakyReLU(inplace=False))
        self.fc_up_det = nn.Sequential(nn.Linear(bottleneck, bottleneck, bias=False), nn.LeakyReLU(inplace=False))

        self.act = nn.LeakyReLU(inplace=False)
        self.dropout_det = nn.Dropout(0.1)
        self.dropout_seg = nn.Dropout(0.05)

    def forward(self, x, prompt_emb):
        g_det = self.gate_det(prompt_emb)
        g_seg = self.gate_seg(prompt_emb)

        x_perm = x.permute(1, 0, 2)

        h_down_det = self.fc_down_det(x_perm)
        h_down_det = h_down_det.permute(0, 2, 1)
        h_down_det = self.norm_down_det(h_down_det)
        h_down_det = h_down_det.permute(0, 2, 1)

        h_down_seg = self.fc_down_seg(x_perm)
        h_down_seg = h_down_seg.permute(0, 2, 1)
        h_down_seg = self.norm_down_seg(h_down_seg)
        h_down_seg = h_down_seg.permute(0, 2, 1)

        if not self.use_additive_gating:
            h_gated_det = h_down_det * g_det.unsqueeze(1)
            h_gated_seg = h_down_seg * g_seg.unsqueeze(1)
        else:
            h_gated_det = h_down_det + g_det.unsqueeze(1)
            h_gated_seg = h_down_seg + g_seg.unsqueeze(1)

        h_act_det = self.dropout_det(self.act(h_gated_det))
        h_act_seg = self.dropout_seg(self.act(h_gated_seg))

        out_det = self.fc_up_det(h_act_det)
        out_seg = self.fc_up_seg(h_act_seg)

        out_det = out_det.permute(1, 0, 2)
        out_seg = out_seg.permute(1, 0, 2)

        return out_det, out_seg


class DPAS_CLIP_Improved(nn.Module):
    """Improved DPAS with multi-scale decoder"""
    def __init__(self, args, clip_model):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.features = args.features_list
        self.img_size = args.img_size
        self.contrast_mood = args.contrast_mood

        self.use_additive_gating = getattr(args, 'use_additive_gating', False)
        self.use_absolute_prompt = getattr(args, 'use_absolute_prompt', False)
        self.use_simple_avg = getattr(args, 'use_simple_avg', False)

        self.tgva_adapters = nn.ModuleList([
            TGVA_Improved(c_in=1024, bottleneck=768, text_dim=768, use_additive_gating=self.use_additive_gating)
            for _ in range(len(self.features))
        ])

        self.layer_importance = nn.Parameter(torch.ones(len(self.features)))

        num_layers = len(self.features)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * num_layers, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        self.seg_decoder = SegmentationDecoder(num_layers=num_layers)

        self.all_adapters_optimizer = torch.optim.Adam([
            {'params': self.tgva_adapters.parameters(), 'lr': args.learning_rate},
            {'params': [self.layer_importance], 'lr': args.learning_rate},
            {'params': self.fusion_mlp.parameters(), 'lr': args.learning_rate},
            {'params': self.seg_decoder.parameters(), 'lr': args.learning_rate}
        ], betas=(0.5, 0.999))

        if self.contrast_mood == "no":
            self.contrast = lambda a, b: (a)
        elif self.contrast_mood== "yes":
            self.contrast = lambda a, b: (a - b)
        else:
            print("ERROR, no such a contrast mood")

    def forward(self, x, text_features):
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)

        det_scores = []
        seg_scores = []

        t_normal = text_features[:, 0]
        t_abnormal = text_features[:, 1]
        batch_size = x.shape[1]

        if not self.use_absolute_prompt:
            diff_prompt = t_abnormal - t_normal
        else:
            diff_prompt = t_abnormal

        prompt_gate_input = diff_prompt.unsqueeze(0).expand(batch_size, -1)

        for i in range(24):
            if i + 1 == 12:
                x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            else:
                x, attn_map = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)

            if (i + 1) in self.features:
                adapter_idx = self.features.index(i+1)

                feat_det, feat_seg = self.tgva_adapters[adapter_idx](x, prompt_gate_input)

                feat_det = feat_det.permute(1, 0, 2)
                feat_seg = feat_seg.permute(1, 0, 2)

                feat_det = feat_det[:, 1:, :]
                feat_seg = feat_seg[:, 1:, :]

                feat_det = feat_det / (feat_det.norm(dim=-1, keepdim=True) + 1e-8)
                feat_seg = feat_seg / (feat_seg.norm(dim=-1, keepdim=True) + 1e-8)

                sim_det_normal = self.dual_contrast(feat_det, t_normal, t_abnormal)
                sim_det_abnormal = self.dual_contrast(feat_det, t_abnormal, t_normal)
                det_scores_cur = torch.cat([sim_det_normal, sim_det_abnormal], dim=-1)
                det_scores.append(det_scores_cur)

                sim_seg_normal = self.dual_contrast(feat_seg, t_normal, t_abnormal)
                sim_seg_abnormal = self.dual_contrast(feat_seg, t_abnormal, t_normal)
                seg_scores_cur = torch.cat([sim_seg_normal, sim_seg_abnormal], dim=-1)
                seg_scores.append(seg_scores_cur)

        x = x.permute(1, 0, 2)
        pooled, tokens = self.image_encoder._global_pool(x)
        pooled = self.image_encoder.ln_post(pooled)

        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj

        if not self.use_simple_avg:
            concat_scores = torch.cat(det_scores, dim=-1)
            fused_det = self.fusion_mlp(concat_scores)
        else:
            fused_det = torch.stack(det_scores).mean(dim=0)

        seg_map = self.seg_decoder(seg_scores)

        return pooled, [fused_det], [seg_map]

    def dual_contrast(self, features, same_text, opposite_text):
        same_view = (features @ same_text.unsqueeze(-1))
        cross_view = (features @ opposite_text.unsqueeze(-1))
        return self.contrast(same_view, cross_view)
