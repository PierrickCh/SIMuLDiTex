# Stylization
Transfer the style from a texture to an image.

## Parameters
nc : 16 or 32 correspond respectively to the 1M and 4M parameters models.

S : Number of sampling steps.

r : Renoising time ratio.

patch_size : Maximum size of patches used if inference triggers memory error, to lower in case this happens.

size : Size of the output.

zeta : Controls the strengh of the data fidelity.

res : Controls the relative size of texture patterns in the output image. Higher value will give an impression of zoom out of the texure. 

zoom_factor : ...