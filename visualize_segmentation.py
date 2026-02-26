import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from dataset.medical_few import MedDataset
from CLIP.clip import create_model
from CLIP.adapter_synergy import DPAS_CLIP
from Prompt.promptChooser import PromptChooser
from loss import Loss_detection

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_to_unit_interval(x):
    """
    Robust normalization to [0,1] for visualization.
    """
    x = x.astype(np.float32)
    min_val = x.min()
    max_val = x.max()

    if max_val > min_val:
        x = (x - min_val) / (max_val - min_val)
    else:
        x = np.zeros_like(x)

    return x

def visualize_sample(image, gt_mask, pred_mask, save_path, title=""):
    """
    4-panel visualization with professional formatting.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), facecolor='white')
    
    # Image normalization for display (ensure RGB is 0-1)
    img_display = image.cpu().numpy().transpose(1, 2, 0)
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)

    # 1. Original Image
    axes[0, 0].imshow(img_display)
    axes[0, 0].set_title("Original Image", fontsize=15, fontweight="bold", pad=10)
    axes[0, 0].axis("off")

    # 2. Ground Truth
    axes[0, 1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Ground Truth", fontsize=15, fontweight="bold", pad=10)
    axes[0, 1].axis("off")

    # 3. Predicted Mask (Smoothed Probabilities)
    axes[1, 0].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Predicted Mask", fontsize=15, fontweight="bold", pad=10)
    axes[1, 0].axis("off")

    # 4. Overlay
    axes[1, 1].imshow(img_display)
    # Using 'jet' or 'hot' for heatmap; hot is often preferred for medical
    heatmap = axes[1, 1].imshow(pred_mask, cmap="hot", alpha=0.5, vmin=0, vmax=1)
    axes[1, 1].set_title("Overlay", fontsize=15, fontweight="bold", pad=10)
    axes[1, 1].axis("off")

    # Professional colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(heatmap, cax=cbar_ax)
    cbar.set_label("Anomaly Probability", size=14, labelpad=15)
    cbar.ax.tick_params(labelsize=12)

    # Clean Title without sample index
    plt.suptitle(title, fontsize=20, fontweight="bold", y=0.98)
    
    # Adjust spacing
    plt.subplots_adjust(left=0.05, right=0.9, top=0.92, bottom=0.05, wspace=0.1, hspace=0.15)
    
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Professional DPAS Visualization")
    parser.add_argument("--obj", type=str, required=True, choices=["Brain", "Liver", "Retina_RESC"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="./visualizations_clean/")
    parser.add_argument("--seed", type=int, default=333)
    
    # Model defaults
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument('--shot', type=int, default=16)
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336')
    parser.add_argument('--pretrain', type=str, default='openai')
    parser.add_argument('--features_list', type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument('--text_mood', type=str, default='learnable_all')
    parser.add_argument('--contrast_mood', type=str, default='yes')
    parser.add_argument('--dec_type', type=str, default='attention')
    parser.add_argument('--loss_type', type=str, default='sigmoid')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    
    args = parser.parse_args()
    setup_seed(args.seed)
    
    # Specific output folder for this run
    final_output_dir = os.path.join(args.output_dir, args.obj)
    os.makedirs(final_output_dir, exist_ok=True)

    print(f"Generating professional visualizations for {args.obj}...")
    
    # Initialize Models
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device, pretrained=args.pretrain)
    model = DPAS_CLIP(args, clip_model=clip_model).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.tgva_adapters.load_state_dict(checkpoint["tgva_adapters"])
    model.fusion_mlp.load_state_dict(checkpoint["fusion_mlp"])
    model.layer_importance.data = checkpoint["layer_importance"]
    model.eval()

    text_chooser = PromptChooser(clip_model, args, device)
    if "prompt_normal" in checkpoint:
        text_chooser.load_prompt(checkpoint)

    # Dataset
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, 0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    loss_det = Loss_detection(args=args, device=device, loss_type=args.loss_type, dec_type=args.dec_type, lr=0.001)

    sample_count = 0
    abnormal_idx = 0
    
    with torch.no_grad():
        text_features = text_chooser()
        
        for image, label, mask in tqdm(test_loader):
            if sample_count >= args.num_samples:
                break
                
            # Focus only on abnormal samples for qualitative results
            if label.item() == 0:
                continue
            
            image = image.to(device)
            mask_np = mask.squeeze().numpy()
            
            # Skip samples with empty/invalid masks if any
            if mask_np.sum() == 0:
                abnormal_idx += 1
                continue

            with torch.cuda.amp.autocast():
                _, _, seg_model = model(image, text_features)

            # Aggregate Anomaly Maps
            anomaly_maps = []
            for layer in range(len(seg_model)):
                # sync_AS returns [B, 2, H, W] after softening/sigmoid
                scores = loss_det.sync_AS(seg_model[layer])
                # Probability of being abnormal (channel 1)
                prob_map = scores[:, 1, :, :].squeeze()
                anomaly_maps.append(prob_map)
            
            # Average layers for better stability
            seg_map = torch.stack(anomaly_maps).mean(dim=0).cpu().numpy()
            
            # PROFESSIONAL STEP: Gaussian smoothing to remove blocky artifacts
            # Sigma=4 is usually good for 240x240 maps derived from 24x24 tokens
            seg_map_smoothed = gaussian_filter(seg_map.astype(np.float32), sigma=4)
            
            # Standardize to 0-1 for consistent display
            seg_map_final = normalize_to_unit_interval(seg_map_smoothed)
            
            # Save Path
            save_path = os.path.join(final_output_dir, f"sample_{abnormal_idx:03d}.png")
            
            visualize_sample(
                image.squeeze(0),
                mask_np,
                seg_map_final,
                save_path,
                title=f"Dataset: {args.obj}"
            )
            
            sample_count += 1
            abnormal_idx += 1

    print(f"\nSUCCESS: Generated {sample_count} clean visualizations in {final_output_dir}")

if __name__ == '__main__':
    main()