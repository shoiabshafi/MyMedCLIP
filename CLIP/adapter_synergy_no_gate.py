import torch
from torch import nn
from torch.nn import functional as F

class TGVA(nn.Module):
    """
    Text-Gated Visual Adapter (TGVA).
    
    This module implements the core synergy mechanism where visual features are modulated 
    by a semantic gating vector derived from the learnable text prompts.
    
    Mathematics:
    g = Sigmoid(W_g * p)
    h_down = Down(x) * g
    x' = x + Up(ReLU(h_down))
    """
    def __init__(self, c_in, bottleneck=768, text_dim=768, use_additive_gating=False):
        super(TGVA, self).__init__()
        self.use_additive_gating = use_additive_gating
        
        # Branch-Specific Gating Projections
        # Detection Gate: emphasizes global anomaly cues
        self.gate_det = nn.Sequential(
            nn.Linear(text_dim, bottleneck),
            nn.Sigmoid()
        )
        # Segmentation Gate: emphasizes local spatial details
        self.gate_seg = nn.Sequential(
            nn.Linear(text_dim, bottleneck),
            nn.Sigmoid()
        )
        
        # Visual Down-Projection (shared)
        self.fc_down = nn.Linear(c_in, bottleneck, bias=False)
        self.norm_down = nn.InstanceNorm1d(bottleneck)
        
        # Visual Up-Projection (Branch 1 - for Segmentation)
        self.fc_up_seg = nn.Sequential(
            nn.Linear(bottleneck, bottleneck, bias=False), 
            nn.LeakyReLU(inplace=False)
        )
        
        # Visual Up-Projection (Branch 2 - for Detection)
        self.fc_up_det = nn.Sequential(
            nn.Linear(bottleneck, bottleneck, bias=False),  
            nn.LeakyReLU(inplace=False)
        )
        
        self.act = nn.LeakyReLU(inplace=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, prompt_emb):
        """
        Args:
            x: Visual features [SeqLen, Batch, Channels]
            prompt_emb: Text embedding [Batch, TextDim] (differential prompt)
        """
        # Generate Branch-Specific Gates
        g_det = self.gate_det(prompt_emb)  # [B, bottleneck] - global focus
        g_seg = self.gate_seg(prompt_emb)  # [B, bottleneck] - spatial focus
        
        # Project Visual Features Down (shared bottleneck)
        x_perm = x.permute(1, 0, 2)  # [B, 290, 1024]
        h_down = self.fc_down(x_perm)  # [B, 290, bottleneck]
        
        # Apply normalization
        h_down = h_down.permute(0, 2, 1)
        h_down = self.norm_down(h_down)
        h_down = h_down.permute(0, 2, 1)
        
        # Apply Branch-Specific Gating
        if not self.use_additive_gating:
            h_gated_det = h_down * g_det.unsqueeze(1)  # Multiplicative (Default)
            h_gated_seg = h_down * g_seg.unsqueeze(1)
        else:
            h_gated_det = h_down + g_det.unsqueeze(1)  # Additive Ablation
            h_gated_seg = h_down + g_seg.unsqueeze(1)
        
        # Activation and Dropout per branch
        h_act_det = self.dropout(self.act(h_gated_det))
        h_act_seg = self.dropout(self.act(h_gated_seg))
        
        # Project Up (separate branches)
        out_det = self.fc_up_det(h_act_det)  # [B, 290, 768]
        out_seg = self.fc_up_seg(h_act_seg)  # [B, 290, 768]
        
        # Permute back
        out_det = out_det.permute(1, 0, 2)
        out_seg = out_seg.permute(1, 0, 2)
        
        return out_det, out_seg



class DPAS_CLIP(nn.Module):
    """
    Dynamic Prompt-Adapter Synergy (DPAS) CLIP Model.
    
    Replaces the dual-branch adapter logic with a single Text-Gated branch.
    """
    def __init__(self, args, clip_model):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.features = args.features_list
        self.img_size = args.img_size
        self.contrast_mood = args.contrast_mood
        
        # Text-Gated Visual Adapters
        self.use_additive_gating = getattr(args, 'use_additive_gating', False)
        self.use_absolute_prompt = getattr(args, 'use_absolute_prompt', False)
        self.use_simple_avg = getattr(args, 'use_simple_avg', False)

        self.tgva_adapters = nn.ModuleList([
            TGVA(c_in=1024, bottleneck=768, text_dim=768, use_additive_gating=self.use_additive_gating) 
            for _ in range(len(self.features))
        ])
        
        # Multi-Scale AC Fusion: Learnable Layer Importance Weights
        self.layer_importance = nn.Parameter(torch.ones(len(self.features)))
        
        # Concat-MLP Fusion Head for Advanced Multi-Scale AC
        # Input: concatenated scores from all layers [B, 289, 2*num_layers]
        # Output: fused score [B, 289, 2]
        num_layers = len(self.features)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * num_layers, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        self.all_adapters_optimizer = torch.optim.Adam([
            {'params': self.tgva_adapters.parameters(), 'lr': args.learning_rate},
            {'params': [self.layer_importance], 'lr': args.learning_rate},
            {'params': self.fusion_mlp.parameters(), 'lr': args.learning_rate}
        ], betas=(0.5, 0.999))
        
        if self.contrast_mood == "no":
            self.contrast = lambda a, b: (a)
        elif self.contrast_mood== "yes":
            self.contrast = lambda a, b: (a - b)
        else:
            print("ERROR, no such a contrast mood")

    def forward(self, x, text_features):
        """
        x: Images
        text_features: [768, 2] -> Col 0: Normal, Col 1: Abnormal
        """
        # Pre-process image
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                          device=x.device),
             x], dim=1)
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)

        det_scores = []
        seg_scores = []
        
        # Prepare Prompt Embeddings for Gating
        t_normal = text_features[:, 0]   # [768]
        t_abnormal = text_features[:, 1] # [768]
        
        batch_size = x.shape[1]
        
        # Broadcast abnormal text feature to batch
        # This means: "Find features that match THIS anomaly description"
        # Differential Gating Enhancement: Use (Abnormal - Normal) direction
        if not self.use_absolute_prompt:
            diff_prompt = t_abnormal - t_normal
        else:
            diff_prompt = t_abnormal # Absolute Ablation (ignores Normal prompt context)
            
        prompt_gate_input = diff_prompt.unsqueeze(0).expand(batch_size, -1) # [B, 768]

        for i in range(24):
            if i + 1 == 12:
                x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            else:
                x, attn_map = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            
            if (i + 1) in self.features:
                adapter_idx = self.features.index(i+1)
                
                # Apply TGVA
                # Returns features adapted to highlight anomalies
                feat_det, feat_seg = self.tgva_adapters[adapter_idx](x, prompt_gate_input)
                                
                # Reshape from [290, B, 768] -> [B, 290, 768]
                feat_det = feat_det.permute(1, 0, 2)
                feat_seg = feat_seg.permute(1, 0, 2)
                
                # Remove CLS token
                feat_det = feat_det[:, 1:, :] # [B, 289, 768]
                feat_seg = feat_seg[:, 1:, :]
                
                # Normalize
                feat_det = feat_det / feat_det.norm(dim=-1, keepdim=True)
                feat_seg = feat_seg / feat_seg.norm(dim=-1, keepdim=True)
                
                # Calculate Similarity
                # Score 1: How similar is this to "Normal"?
                sim_det_normal = self.dual_contrast(feat_det, t_normal, t_abnormal)
                
                # Score 2: How similar is this to "Abnormal"?
                sim_det_abnormal = self.dual_contrast(feat_det, t_abnormal, t_normal)
                
                # Concatenate [NormalScore, AbnormalScore]
                det_scores_cur = torch.cat([sim_det_normal, sim_det_abnormal], dim=-1)
                det_scores.append(det_scores_cur)
                
                # Same for Segmentation
                sim_seg_normal = self.dual_contrast(feat_seg, t_normal, t_abnormal)
                sim_seg_abnormal = self.dual_contrast(feat_seg, t_abnormal, t_normal)
                seg_scores_cur = torch.cat([sim_seg_normal, sim_seg_abnormal], dim=-1)
                seg_scores.append(seg_scores_cur)

        x = x.permute(1, 0, 2)
        pooled, tokens = self.image_encoder._global_pool(x)
        pooled = self.image_encoder.ln_post(pooled)

        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj

        # Multi-Scale AC Fusion
        if not self.use_simple_avg:
            # Concat-MLP for advanced layer interaction learning (Default)
            concat_scores = torch.cat(det_scores, dim=-1)
            fused_det = self.fusion_mlp(concat_scores)
        else:
            # Simple Average Ablation
            fused_det = torch.stack(det_scores).mean(dim=0)
        
        return pooled, [fused_det], seg_scores

    def dual_contrast(self, features, same_text, opposite_text):
        same_view = (features @ same_text.unsqueeze(-1)) 
        cross_view = ( features @ opposite_text.unsqueeze(-1)) 
        return self.contrast(same_view, cross_view)
