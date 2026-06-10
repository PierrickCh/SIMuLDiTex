# code from Mahé DUVAL
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports and file pathsto work from the Demos folder more easily
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# UI Imports
import gradio as gr
import numpy as np

# SIMuLDiTex Imports
from SIMuLDiTex.SIMuLDiTex import Unet, GaussianDiffusion, Trainer
from torchvision.utils import make_grid,save_image
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import re

from Demos.utilities.useful_functions import *

# Global variables
DOCUMENTATION_PATH = 'Demos/sub_demos/doc/doc_spatial_interpolation.md'

width, height = 2048, 1024
scale_factor = 1.
out_width, out_height = width, height

name1,name2 = 'gold','wall' # 2 textures for background anbd font: 'wall' 'carpet' 'rust' 'crepe' 'ananaskin' 'ananaskin2','gold'
nc = 16 # 16, 32    for 1M or 4M parameters
S = 2 # Sampling steps
r = .8 # renoising time ratio
patch_size=3000 # Maximum side of patches used if inference triggers memory error, to lower in case this happens.
octaves = 2
total_time_step_ratio = 1.

is_model_loaded = False

## Functions
diffusion:GaussianDiffusion = None
def load_models_spatial_interp():
    global diffusion, is_model_loaded
    is_model_loaded = False
    
    # Texture 1
    folder='runs/%s_lr1e-4_bs32_T200_100000_dim%d_octaves_3/'%(name1,nc)    
    model1 = Unet(
        dim =nc,
        dim_mults = (1, 2, 4, 4),
        mid_fourier=True)
    diffusion1 = GaussianDiffusion(
        model1,
        image_size = 128,
        timesteps = 200,
        sampling_timesteps=S)
    trainer1 = Trainer(
        diffusion1,
        'images/data/%s'%name1,
        results_folder=folder)
    trainer1.load(get_latest_model_index(folder))   

    # Texture 2
    folder='runs/%s_lr1e-4_bs32_T200_100000_dim%d_octaves_3/'%(name2,nc)
    model2 = Unet(
        dim =nc,
        dim_mults = (1, 2, 4, 4),
        mid_fourier=True)
    diffusion2 = GaussianDiffusion(
        model2,
        image_size = 128,
        timesteps = 200,
        sampling_timesteps=S)
    trainer2 = Trainer(
        diffusion2,
        'images/data/%s'%name2,
        results_folder=folder)
    trainer2.load(get_latest_model_index(folder))   
    diffusion1.model2=model2

    diffusion = diffusion1

    is_model_loaded = True
    return gr.update(interactive=True, value = "Force Generate")
is_running:bool = False
def run_spatial_interp():   
    torch.cuda.empty_cache()
    """
    tex1 : background texture
    tex2 : the masking texture
    """
    global is_running
    is_running = True
    
    #omega = load_image_tensor('./Demos/results/test_mask.jpg', device='cuda',scaling_factor=4)
    omega = load_array_tensor(np.abs(255 - canvas_tex), device='cuda')
    #save_image(omega[0],'./Demos/results/test_mask.jpg')
    _, c, h, w = omega.shape
    size = (out_height, out_width)

    for im in diffusion.spatial_interp(size=size, time_ratio=r, omega=omega, patch_size=patch_size, octaves=octaves,total_time_step_ratio=total_time_step_ratio):
        
        # To go from torch.Tensor to a numpy image
        img = im[0].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8) # Convert range to [0, 255] and uint8 for Gradio
        
        if is_running == False : return img
        
        
        yield img   
    is_running = False
    
## Globals update
def update_texture_1(tex1):
    global name1
    name1 = tex1
def update_texture_2(tex2):
    global name2
    name2 = tex2

def update_parameter_number(x):
    global nc
    match x:
        case "1M":
            nc = 16
        case "4M":
            nc = 32
def update_simulditex_params(i_s,i_r,i_patch_size, i_octaves):
    global S,r,patch_size, octaves
    
    S = i_s
    r = i_r 
    patch_size = i_patch_size   
    octaves = i_octaves
def update_dimensions(i_w,i_h):
    global width, height, out_width, out_height
    width = i_w
    height = i_h
    
    out_width = width
    out_height = height
def update_scaling_factor_dimensions(i_f):
    global scale_factor, out_width, out_height
    scale_factor = i_f
    out_width = width*scale_factor
    out_height = height*scale_factor
    return round(out_width), round(out_height)

canvas_tex = np.full((height, width), 255, dtype=np.uint8)
def update_zoom_factor(i_z):
    global zoom_factor
    zoom_factor = i_z
def update_res(i_r):
    global res
    res = i_r
def update_zeta(i_z):
    global zeta
    zeta = i_z
def update_global_canvas_tex(im):
    global canvas_tex
    canvas_tex = im['composite']
def stop_running():
    global is_running
    is_running = False
def update_total_time_step_ratio(i_t):
    global total_time_step_ratio
    total_time_step_ratio = i_t
# Interface

load_models_spatial_interp() # pre-load the models to gain in speed during the real-time drawing (otherwise there's a delay of ~5s)

def demo_spatial_interpolation():
    with gr.Blocks():
        # Parameter Number
        with gr.Row(equal_height=True,variant="default"):
            in_radio_nc = gr.Radio(
                choices=["1M", "4M"],
                label="Parameters number",
                value="1M",
                elem_classes="radio_group",
            )
        # Input and output row
        with gr.Row(equal_height=True,variant="panel", elem_classes="fixed_height_image_row flex_display"):
            im_layers = gr.LayerOptions(
                layers=["Mask"],
                allow_additional_layers=False 
            )
            im_brushes = gr.Brush(
                default_size="auto",
                colors=["rgb(0, 0, 0)"],
                
            )
            im = gr.ImageEditor(
                type="numpy",
                label="Input",
                sources=(),
                elem_classes="full_height",
                elem_id="image_canvas",
                canvas_size=(width,height),
                brush=im_brushes
            )
            
            
            
            
            im_preview = gr.Image(
                type="numpy",
                label="Output",
                elem_classes="output-image-fill",
                interactive=False,
                streaming=False
                
            )
        with gr.Row(equal_height=True):
            reload_btn = gr.Button(value="Force Stop")
            btn_generate = gr.Button(value="Force Generate", variant="primary", interactive=True)
            clear_cuda_cache_btn = gr.Button(value="Clear CUDA Cache", variant="stop", interactive=True)
        with gr.Row(equal_height=True):
            in_drop_tex_1 = gr.Dropdown(
                choices=['wall','carpet','rust','crepe','ananaskin','ananaskin2','gold','blue_up'],
                label="Texture 1",
                info="Background texture displayed behind the masking texture (Texture 2)",
                value=name1,
                )
            in_drop_tex_2 = gr.Dropdown(
                choices=['wall','carpet','rust','crepe','ananaskin','ananaskin2','gold','blue_up'],
                label="Texture 2",
                info="Texture displayed above the background texture (Texture 1). The mask input will use this texture.",
                value=name2,
                )
        # Bind events
        ## Bind canvas edit
        update_preview = im.change(
            fn=stop_running,

        ).then(
            fn=update_global_canvas_tex,
            inputs=im,
        ).then(
            fn=run_spatial_interp,
            inputs=[],
            outputs=[im_preview],
            show_progress=False
        )
        
        reload_btn.click(
            fn = stop_running,
            cancels=update_preview
        )
        ## Bind texture change
        in_drop_tex_1.change(
            fn = lambda : gr.update(interactive=False, value="Loading models"),
            outputs=btn_generate
        ).then(
            fn=update_texture_1,
            inputs=[in_drop_tex_1],
            outputs=[]
        ).then(
            fn = load_models_spatial_interp,
            outputs=btn_generate
        )
        in_drop_tex_2.change(
            fn = lambda : gr.update(interactive=False, value="Loading models"),
            outputs=btn_generate
        ).then(
            fn=update_texture_2,
            inputs=[in_drop_tex_2],
            outputs=[]
        ).then(
            fn=load_models_spatial_interp,
            outputs=btn_generate
        )
        
        

        btn_generate.click(
            fn=run_spatial_interp,
            inputs=[],
            outputs=[im_preview]
        )
        clear_cuda_cache_btn.click(
            fn = clear_cuda_cache
        )
        
        with gr.Row(equal_height=True,):
            with gr.Column(scale=1):
                with gr.Column():
                    with gr.Row():
                        in_width = gr.Slider(
                            label="Width (px)",
                            info="Width of the image",
                            value=width,
                            minimum=2**6,
                            maximum=2**12,
                            step=2**6
                        )

                        in_height = gr.Slider(
                            label="Height (px)",
                            info="Width of the image",
                            value=height,
                            minimum=2**6,
                            maximum=2**12,
                            step=2**6
                        )

                    in_scale_factor = gr.Slider(
                        label="Scale factor",
                        info="Scaling factor for width and height conserving the aspect ratio (from the initial resolution)",
                        value=scale_factor,
                        minimum=0.3,
                        maximum=2,
                        step=0.1
                    )
                
            with gr.Column(scale=2):
                with gr.Row(equal_height=True):
                    out_width_display = gr.Text(
                        label="Output Width (px)",
                        info="Width of the output image in pixels",
                        value=width,
                    )
                    out_height_display = gr.Text(
                        label="Output Height (px)",
                        info="Height of the output image in pixels",
                        value=out_height
                    )
                in_S = gr.Slider(
                    label="Sampling Steps (S)",
                    info="Number of sampling temps",
                    value=2,
                    minimum=1,
                    maximum=20,
                    step=1
                )
                in_total_time_step_ratio = gr.Slider(
                    label="Total time step ratio",
                    info=".",
                    value=1,
                    minimum=0.01,
                    maximum=1.,
                    step=0.01
                )
                in_r = gr.Slider(
                    label="Renoising Time Ration (r)",
                    info="Renoising time ratio",
                    value=0.8,
                    minimum=0,
                    maximum=1,
                    step=0.01
                )
                in_patch_size = gr.Slider(
                    label="Patch Size",
                    info="Maximum size of patches used (if inference triggers memory error, lower it)",
                    value=3000,
                    minimum=1,
                    maximum=10000,
                    step=1
                )
                in_octaves = gr.Slider(
                    label="Octaves",
                    info="",
                    value=3,
                    minimum=1,
                    maximum=10,
                    step=1
                )
        ## Bind general simulditex parameters
        in_S.change(
            fn=update_simulditex_params,
            inputs=[in_S, in_r, in_patch_size, in_octaves],
            outputs=[]
        )
        in_r.change(
            fn=update_simulditex_params,
            inputs=[in_S, in_r, in_patch_size, in_octaves],
            outputs=[]
        )
        in_patch_size.change(
            fn=update_simulditex_params,
            inputs=[in_S, in_r, in_patch_size, in_octaves],
            outputs=[]
        )
        in_octaves.change(
            fn=update_simulditex_params,
            inputs=[in_S, in_r, in_patch_size, in_octaves],
            outputs=[]
        )
        ## Bind texture dimension
        in_width.change(
            fn=update_dimensions,
            inputs=[in_width,in_height],
        )
        in_height.change(
            fn=update_dimensions,
            inputs=[in_width,in_height],
        )
        in_scale_factor.change(
            fn=update_scaling_factor_dimensions,
            inputs=in_scale_factor,
            outputs=[out_width_display,out_height_display]
        )
        in_radio_nc.change(
            fn = lambda : gr.update(interactive=False, value="Loading models"),
            outputs=btn_generate
        ).then(
            fn=update_parameter_number,
            inputs=[in_radio_nc],
            outputs=[]
        ).then(
            fn = load_models_spatial_interp,
            outputs=btn_generate
        )
        in_total_time_step_ratio.change(
            fn=update_total_time_step_ratio,
            inputs=in_total_time_step_ratio
        )
    with gr.Accordion(label="Documentation", open=True):
        with open(DOCUMENTATION_PATH,'r') as f : 
            gr.Markdown(f.read(),latex_delimiters=[{ "left": "$$", "right": "$$", "display": True },{"left": "$", "right": "$", "display": False},])
if __name__ == "__main__":
    with gr.Blocks() as demo:
        demo_spatial_interpolation()
    demo.queue().launch()