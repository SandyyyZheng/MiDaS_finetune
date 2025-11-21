"""MiDaS Loss Functions - Based on Official Implementation"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_scale_and_shift(prediction, target, mask):
    """
    Compute scale and shift for aligning prediction to target
    Solves 2x2 linear system using least squares

    Args:
        prediction: [B, H, W] predicted disparity
        target: [B, H, W] ground truth disparity
        mask: [B, H, W] valid pixel mask (0.0 or 1.0)

    Returns:
        scale: [B] scale parameter
        shift: [B] shift parameter
    """
    # System matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # Right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # Solution: x = A^-1 . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    """
    Batch-based reduction: average of all valid pixels in the batch

    Args:
        image_loss: [B] per-image loss
        M: [B] number of valid pixels per image

    Returns:
        scalar loss
    """
    divisor = torch.sum(M)
    if divisor == 0:
        return torch.tensor(0.0, device=image_loss.device)
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    """
    Image-based reduction: mean of average of valid pixels per image

    Args:
        image_loss: [B] per-image loss
        M: [B] number of valid pixels per image

    Returns:
        scalar loss
    """
    valid = M.nonzero()
    image_loss_normalized = torch.zeros_like(image_loss)
    image_loss_normalized[valid] = image_loss[valid] / M[valid]
    return torch.mean(image_loss_normalized)


def trimmed_mae_loss(prediction, target, mask, trim_ratio=0.2, reduction=reduction_batch_based):
    """
    Trimmed MAE Loss: discard largest 20% errors
    This is the core innovation of MiDaS for handling noisy ground truth

    Args:
        prediction: [B, H, W] predicted disparity (already aligned)
        target: [B, H, W] ground truth disparity
        mask: [B, H, W] valid pixel mask
        trim_ratio: ratio of largest errors to discard (default: 0.2)
        reduction: reduction function

    Returns:
        scalar loss
    """
    batch_size = prediction.shape[0]
    M = torch.sum(mask, (1, 2))

    # Compute absolute errors
    res = torch.abs(prediction - target)
    image_loss = torch.zeros(batch_size, device=prediction.device)

    for i in range(batch_size):
        # Get valid errors for this image
        valid_errors = res[i][mask[i] > 0]

        if len(valid_errors) == 0:
            continue

        # Sort errors and trim largest trim_ratio
        num_pixels = len(valid_errors)
        num_keep = int(num_pixels * (1 - trim_ratio))

        sorted_errors, _ = torch.sort(valid_errors)
        trimmed_errors = sorted_errors[:num_keep]

        # Sum of trimmed errors
        image_loss[i] = torch.sum(trimmed_errors)

    return reduction(image_loss, M * (1 - trim_ratio))


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):
    """
    Gradient loss: compute gradient of the difference map (prediction - target)
    This encourages sharp edges aligned with ground truth

    Key insight from MiDaS paper: compute gradient of DIFFERENCE, not individual gradients

    Args:
        prediction: [B, H, W] predicted disparity (already aligned)
        target: [B, H, W] ground truth disparity
        mask: [B, H, W] valid pixel mask
        reduction: reduction function

    Returns:
        scalar loss
    """
    M = torch.sum(mask, (1, 2))

    # Compute difference map
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients of difference (not prediction and target separately!)
    # Horizontal gradient
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Vertical gradient
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Sum gradients
    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class TrimmedMAELoss(nn.Module):
    """
    Scale and Shift Invariant Trimmed MAE Loss
    Core data loss of MiDaS
    """
    def __init__(self, trim_ratio=0.2, reduction='batch-based'):
        super().__init__()
        self.trim_ratio = trim_ratio

        if reduction == 'batch-based':
            self.reduction = reduction_batch_based
        else:
            self.reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return trimmed_mae_loss(prediction, target, mask, self.trim_ratio, self.reduction)


class GradientLoss(nn.Module):
    """
    Multi-scale Gradient Loss
    Regularization term of MiDaS
    """
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()
        self.scales = scales

        if reduction == 'batch-based':
            self.reduction = reduction_batch_based
        else:
            self.reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        total = 0

        # Multi-scale: use stride downsampling (not avg_pool!)
        for scale in range(self.scales):
            step = pow(2, scale)

            # Downsample using stride (official implementation)
            pred_scaled = prediction[:, ::step, ::step]
            target_scaled = target[:, ::step, ::step]
            mask_scaled = mask[:, ::step, ::step]

            total += gradient_loss(pred_scaled, target_scaled, mask_scaled, self.reduction)

        return total


class MiDaSLoss(nn.Module):
    """
    Complete MiDaS Loss Function
    L_total = L_trimmed_mae + alpha * L_gradient

    Based on the paper "Towards Robust Monocular Depth Estimation:
    Mixing Datasets for Zero-shot Cross-dataset Transfer"
    """
    def __init__(self, alpha=0.5, trim_ratio=0.2, scales=4, reduction='batch-based'):
        """
        Args:
            alpha: weight for gradient loss (paper uses 0.5)
            trim_ratio: ratio to trim in MAE loss (paper uses 0.2)
            scales: number of scales for gradient loss (paper uses 4)
            reduction: 'batch-based' or 'image-based'
        """
        super().__init__()

        self.alpha = alpha
        self.data_loss = TrimmedMAELoss(trim_ratio=trim_ratio, reduction=reduction)
        self.regularization_loss = GradientLoss(scales=scales, reduction=reduction)

        # Store aligned prediction for debugging
        self.prediction_aligned = None

    def forward(self, prediction, target, mask=None):
        """
        Args:
            prediction: [B, H, W] or [B, 1, H, W] predicted disparity
            target: [B, H, W] or [B, 1, H, W] ground truth disparity
            mask: [B, H, W] or [B, 1, H, W] valid pixel mask (optional)

        Returns:
            total_loss, data_loss, regularization_loss
        """
        # Ensure shape is [B, H, W]
        if prediction.dim() == 4:
            prediction = prediction.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)

        # Create mask if not provided
        if mask is None:
            mask = torch.ones_like(prediction)
        else:
            if mask.dim() == 4:
                mask = mask.squeeze(1)
            mask = mask.float()

        # Step 1: Align prediction to target (scale and shift invariant)
        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.prediction_aligned = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        # Step 2: Compute data loss (trimmed MAE)
        data_loss = self.data_loss(self.prediction_aligned, target, mask)

        # Step 3: Compute regularization loss (multi-scale gradient)
        reg_loss = torch.tensor(0.0, device=prediction.device)
        if self.alpha > 0:
            reg_loss = self.regularization_loss(self.prediction_aligned, target, mask)

        # Step 4: Total loss
        total_loss = data_loss + self.alpha * reg_loss

        return total_loss, data_loss, reg_loss
