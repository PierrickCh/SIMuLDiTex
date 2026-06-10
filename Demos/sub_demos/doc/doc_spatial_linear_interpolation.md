# Spatial linear interpolation between two textures
Blend two textures from right to left with a gradient.
## Parameters
nc : 16 or 32 correspond respectively to the 1M and 4M parameters models.

S : Number of sampling steps.

r : Renoising time ratio.

patch_size : Maximum size of patches used if inference triggers memory error, to lower in case this happens.

size : Size of the output.