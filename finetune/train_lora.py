"""Fine-tune MiDaS model using LoRA"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midas.model_loader import load_model
from finetune.dataset import NYU2Dataset
from finetune.loss import MiDaSLoss
from finetune.lora import inject_lora_to_linear, mark_only_lora_as_trainable, save_lora_weights


def setup_logging(output_dir):
    """Setup logging to both file and console"""
    os.makedirs(output_dir, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_dir, f'training_{timestamp}.log')

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return log_file


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_ssi = 0
    total_grad = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        targets = batch['disparity'].to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)

        # Compute loss
        loss, ssi_loss, grad_loss = criterion(predictions, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Record
        total_loss += loss.item()
        total_ssi += ssi_loss.item()
        total_grad += grad_loss.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'ssi': f"{ssi_loss.item():.4f}",
            'grad': f"{grad_loss.item():.4f}"
        })

    avg_loss = total_loss / len(dataloader)
    avg_ssi = total_ssi / len(dataloader)
    avg_grad = total_grad / len(dataloader)

    return avg_loss, avg_ssi, avg_grad


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_ssi = 0
    total_grad = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch['image'].to(device)
            targets = batch['disparity'].to(device)

            # Forward pass
            predictions = model(images)

            # Compute loss
            loss, ssi_loss, grad_loss = criterion(predictions, targets)

            total_loss += loss.item()
            total_ssi += ssi_loss.item()
            total_grad += grad_loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_ssi = total_ssi / len(dataloader)
    avg_grad = total_grad / len(dataloader)

    return avg_loss, avg_ssi, avg_grad


def main(args):
    # Setup logging
    log_file = setup_logging(args.output_dir)
    logging.info("="*80)
    logging.info("MiDaS LoRA Fine-tuning")
    logging.info("="*80)

    # Log configuration
    logging.info("\nConfiguration:")
    for arg in vars(args):
        logging.info(f"  {arg}: {getattr(args, arg)}")
    logging.info("="*80)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"\nDevice: {device}")

    # Load pretrained model
    logging.info(f"\nLoading model: {args.model_type}")
    model, transform, net_w, net_h = load_model(
        device=device,
        model_path=args.model_path,
        model_type=args.model_type,
        optimize=False
    )
    logging.info(f"Model input size: {net_w}x{net_h}")

    # Inject LoRA
    logging.info(f"\nInjecting LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
    lora_params = inject_lora_to_linear(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        target_modules=None  # None = all linear layers
    )
    mark_only_lora_as_trainable(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params / 1e6:.2f}M")
    logging.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")

    # Load dataset and split train/val
    logging.info(f"\nLoading dataset from: {args.train_csv}")
    full_dataset = NYU2Dataset(args.train_csv, img_size=net_h)

    # Split into train and validation
    train_size = int((1 - args.val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    logging.info(f"Dataset split (val_split={args.val_split}):")
    logging.info(f"  Total samples: {len(full_dataset)}")
    logging.info(f"  Training samples: {len(train_dataset)}")
    logging.info(f"  Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Loss function (following paper: alpha=0.5, trim_ratio=0.2, scales=4)
    criterion = MiDaSLoss(alpha=0.5, trim_ratio=0.2, scales=4, reduction='batch-based')

    # Optimizer with stratified learning rates (following MiDaS paper)
    # Paper uses: pretrained layers 1e-5, randomly initialized layers 1e-4
    # For LoRA: we use 1e-4 for LoRA parameters (they are "randomly initialized")
    # Note: In LoRA fine-tuning, only LoRA parameters are trainable
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lora_lr,  # Use LoRA learning rate for all trainable params
        betas=(0.9, 0.999)
    )

    logging.info(f"\nOptimizer: Adam (β1=0.9, β2=0.999)")
    logging.info(f"Learning rate: {args.lora_lr} (LoRA parameters)")

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step,
        gamma=0.5
    )

    # Training loop
    best_val_loss = float('inf')
    logging.info(f"\n{'='*80}")
    logging.info(f"Starting training ({args.epochs} epochs)")
    logging.info(f"{'='*80}\n")

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_ssi, train_grad = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_ssi, val_grad = validate(
            model, val_loader, criterion, device
        )

        # Adjust learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log statistics
        logging.info(f"\nEpoch {epoch}/{args.epochs}")
        logging.info(f"  Train - Loss: {train_loss:.4f}, SSI: {train_ssi:.4f}, Grad: {train_grad:.4f}")
        logging.info(f"  Val   - Loss: {val_loss:.4f}, SSI: {val_ssi:.4f}, Grad: {val_grad:.4f}")
        logging.info(f"  LR: {current_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.output_dir, "best_lora.pth")
            save_lora_weights(model, save_path)
            logging.info(f"  ✓ Saved best model: {save_path}")

        # Save checkpoint periodically
        if epoch % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"lora_epoch_{epoch}.pth")
            save_lora_weights(model, save_path)
            logging.info(f"  ✓ Saved checkpoint: {save_path}")

        logging.info("-" * 80)

    # Save final model
    final_save_path = os.path.join(args.output_dir, "final_lora.pth")
    save_lora_weights(model, final_save_path)
    logging.info(f"\n{'='*80}")
    logging.info(f"Training completed!")
    logging.info(f"Final model saved to: {final_save_path}")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")
    logging.info(f"Log file: {log_file}")
    logging.info(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune MiDaS model using LoRA")

    # Model parameters
    parser.add_argument("--model_path", type=str, default="weights/dpt_large_384.pt",
                        help="Path to pretrained model")
    parser.add_argument("--model_type", type=str, default="dpt_large_384",
                        help="Model type")

    # LoRA parameters
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank (smaller = fewer parameters)")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                        help="LoRA alpha scaling factor")

    # Data parameters
    parser.add_argument("--train_csv", type=str, default="input/nyu2_train.csv",
                        help="Training CSV file")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio (default: 0.1 = 10%)")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs")
    parser.add_argument("--lora_lr", type=float, default=1e-4,
                        help="Learning rate for LoRA parameters (paper uses 1e-4 for new layers)")
    parser.add_argument("--lr_step", type=int, default=10,
                        help="Learning rate decay step (halves LR every lr_step epochs)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")

    # Output parameters
    parser.add_argument("--output_dir", type=str, default="finetune/checkpoints",
                        help="Output directory")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="Checkpoint save interval")

    args = parser.parse_args()

    main(args)
