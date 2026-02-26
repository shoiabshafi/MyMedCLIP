import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset.medical_few import MedDataset
from CLIP.clip import create_model
from CLIP.adapter_synergy import DPAS_CLIP
from Prompt.promptChooser import PromptChooser
from loss import Loss_detection

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_to_unit_interval(x):
    """
    Robust normalization to [0,1]
    Used only for visualization.
    """
    x = x.astype(np.float32)

    # If accidentally in 0-255 range
    if x.max() > 1.0:
        x = x / 255.0

    min_val = x.min()
    max_val = x.max()

    if max_val > min_val:
        x = (x - min_val) / (max_val - min_val)
    else:
        x = np.zeros_like(x)

    return x.astype(np.float32)


def visualize_sample(image, gt_mask, pred_mask, save_path, title=""):
    """
    4-panel visualization:
    1. Original Image
    2. Ground Truth
    3. Prediction
    4. Heatmap Overlay
    """

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Image normalization for display
    img_display = image.cpu().numpy().transpose(1, 2, 0)
    img_display = normalize_to_unit_interval(img_display)

    # Original
    axes[0, 0].imshow(img_display)
    axes[0, 0].set_title("Original Image", fontsize=14, fontweight="bold")
    axes[0, 0].axis("off")

    # GT
    axes[0, 1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Ground Truth", fontsize=14, fontweight="bold")
    axes[0, 1].axis("off")

    # Prediction
    axes[1, 0].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Predicted Mask", fontsize=14, fontweight="bold")
    axes[1, 0].axis("off")

    # Heatmap overlay
    axes[1, 1].imshow(img_display)
    heatmap = axes[1, 1].imshow(pred_mask, cmap="hot", alpha=0.5, vmin=0, vmax=1)
    axes[1, 1].set_title("Overlay", fontsize=14, fontweight="bold")
    axes[1, 1].axis("off")

    cbar = plt.colorbar(heatmap, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar.set_label("Anomaly Score", rotation=270, labelpad=20)

    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize DPAS Segmentation Results")
    parser.add_argument("--obj", type=str, default="Brain",
                        choices=["Brain", "Liver", "Retina_RESC"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--img_size", type=int, default=240)
    parser.add_argument("--shot", type=int, default=16)
    parser.add_argument("--iterate", type=int, default=0)
    parser.add_argument("--seed", type=int, default=333)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="./visualizations/")
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--pretrain", type=str, default="openai")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--text_mood", type=str, default="learnable_all")
    parser.add_argument("--contrast_mood", type=str, default="yes")
    parser.add_argument("--dec_type", type=str, default="attention")
    parser.add_argument("--loss_type", type=str, default="sigmoid")
    parser.add_argument("--learning_rate", type=float, default=0.001)

    args = parser.parse_args()
    setup_seed(args.seed)

    output_dir = os.path.join(args.output_dir, f"{args.obj}_seed{args.seed}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Visualizing {args.obj}")
    print(f"Loading checkpoint: {args.checkpoint}")

    # CLIP
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained=args.pretrain,
        require_pretrained=True
    )
    clip_model.to(device)
    clip_model.eval()

    # DPAS
    model = DPAS_CLIP(args, clip_model=clip_model).to(device)
    model.eval()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.tgva_adapters.load_state_dict(checkpoint["tgva_adapters"])
    model.fusion_mlp.load_state_dict(checkpoint["fusion_mlp"])
    model.layer_importance.data = checkpoint["layer_importance"]

    print("✓ Model loaded")

    text_chooser = PromptChooser(clip_model, args, device)
    if "prompt_normal" in checkpoint:
        text_chooser.load_prompt(checkpoint)
        print("✓ Prompts loaded")

    test_dataset = MedDataset(
        args.data_path,
        args.obj,
        args.img_size,
        args.shot,
        args.iterate
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )

    loss_det = Loss_detection(
        args=args,
        device=device,
        loss_type=args.loss_type,
        dec_type=args.dec_type,
        lr=0.001
    )

    print(f"Processing {args.num_samples} abnormal samples...")

    sample_count = 0
    abnormal_count = 0

    with torch.no_grad():
        text_features = text_chooser()

        for image, label, mask in tqdm(test_loader):

            if sample_count >= args.num_samples:
                break

            if label.item() == 0:
                continue

            image = image.to(device)
            mask = mask.to(device)

            mask = (mask > 0.5).float()

            with torch.cuda.amp.autocast():
                _, _, seg_model = model(image, text_features)

            anomaly_maps = []
            for layer in range(len(seg_model)):
                seg_scores = loss_det.sync_AS(seg_model[layer])
                anomaly_map = (
                    0.5 * (1 - seg_scores[:, 0, :, :])
                    + 0.5 * seg_scores[:, 1, :, :]
                )
                anomaly_maps.append(anomaly_map)

            seg_map = torch.sum(torch.stack(anomaly_maps), dim=0).squeeze()

            # ===== FINAL CORRECT 0–1 SCALING =====
            seg_map_np = seg_map.detach().cpu().numpy()
            seg_map_np = normalize_to_unit_interval(seg_map_np)

            gt_mask_np = mask.squeeze().cpu().numpy()

            save_path = os.path.join(
                output_dir,
                f"sample_{abnormal_count:03d}.png"
            )

            visualize_sample(
                image.squeeze(0),
                gt_mask_np,
                seg_map_np,
                save_path,
                title=f"Dataset: {args.obj}"
            )

            abnormal_count += 1
            sample_count += 1

    print(f"\n✓ Saved {abnormal_count} visualizations to {output_dir}")


if __name__ == "__main__":
    main()