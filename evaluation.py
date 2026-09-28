import torch
import torch.nn.functional as F

try:
    from .distribution_losses import ssim_score
except ImportError:
    from distribution_losses import ssim_score

def evaluate_on_loader(model_obj, loader, dist_manager, device, phys_min, phys_max):
    model_obj.eval()
    mse_sum, ssim_sum, count, sample_count = 0.0, 0.0, 0, 0
    ssim_range = max(phys_max - phys_min, 1e-6)

    with torch.no_grad():
        for img, target in loader:
            img = img.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits, _, _ = model_obj(img)
            if logits.shape[2:] != target.shape[2:]:
                logits = F.interpolate(logits, size=target.shape[2:], mode='bilinear', align_corners=False)
            pred_phys = dist_manager.get_val(logits)
            mse_sum += F.mse_loss(pred_phys.reshape(-1), target.reshape(-1), reduction='sum').item()
            ssim_sum += ssim_score(pred_phys, target, data_range=ssim_range).item() * img.size(0)
            count += target.numel()
            sample_count += img.size(0)

    return mse_sum / max(count, 1), ssim_sum / max(sample_count, 1)
