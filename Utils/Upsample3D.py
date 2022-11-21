import torch.nn.functional as F

# upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    # src = F.upsample(src, size=tar.shape[2:], mode='trilinear', align_corners=False)
    src = F.interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=False)

    return src

