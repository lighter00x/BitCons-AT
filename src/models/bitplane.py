import torch.nn as nn

from losses import apply_bitplane_mask


def bitplane_mask_ste(inputs, planes):
    """Use an exact bit mask in the forward pass and identity BPDA backward."""
    masked = apply_bitplane_mask(inputs, planes)
    return inputs + (masked - inputs).detach()


class BitPlaneBPDA(nn.Module):
    """Apply deterministic low-bit masking as part of the defended model."""

    def __init__(self, backbone, planes):
        super().__init__()
        self.backbone = backbone
        self.planes = tuple(int(plane) for plane in planes)

    def forward(self, inputs, return_feat=False):
        defended_inputs = bitplane_mask_ste(inputs, self.planes)
        return self.backbone(defended_inputs, return_feat=return_feat)


class QuantizeBPDA(BitPlaneBPDA):
    """Round inputs to the 8-bit grid without clearing any bit-plane."""

    def __init__(self, backbone):
        super().__init__(backbone, planes=())
