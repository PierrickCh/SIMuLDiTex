# code from Mahé DUVAL
import sys
from pathlib import Path

# Add parent directory to path for imports and file pathsto work from the Demos folder more easily
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# UI Imports
import gradio as gr
import numpy as np

# SIMuLDiTex Imports
from SIMuLDiTex.SIMuLDiTex import Unet, GaussianDiffusion, Trainer
import numpy as np

from Demos.utilities.useful_functions import *

# Global variables
DOCUMENTATION_PATH = 'Demos/sub_demos/doc/doc_stylization.md'

width, height = 2048, 1024
scale_factor = 1.
out_width, out_height = width, height

name = 'gold'
nc = 16
S = 2
r = .8
patch_size = 4096
octaves = 2
zeta = 0.5
res = 1
zoom_factor = 1.

is_model_loaded = False

maximum_step_number = 1.
img_steps = []

final_img = None

## Functions
diffusion:GaussianDiffusion = None
trainer:Trainer = None

def load_models_stylization():
    global is_model_loaded, diffusion, trainer
    is_model_loaded = False

    folder='runs/%s_lr1e-4_bs32_T200_100000_dim%d_octaves_3/'%(name,nc)
    
    model = Unet(
        dim =nc,
        dim_mults = (1, 2, 4, 4),
        mid_fourier=True)
    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 200,
        sampling_timesteps=S)
    trainer1 = Trainer(
        diffusion,
        'images/data/%s'%name,
        results_folder=folder)
    trainer1.load(get_latest_model_index(folder))
    trainer = trainer1

    is_model_loaded = True
    
    return gr.update(interactive=True, value="Generate")
def stylize(pth):
    
    global final_img, img_steps
    
    gt=trainer.ds.images[0][0].unsqueeze(0)
    input=load_image_tensor(pth)
    last_img = None
    list_of_steps = []
    final_steps = []
    
    for im in diffusion.stylize(size=(input.shape[-2]*zoom_factor,input.shape[-1]*zoom_factor),input=input,gt=gt,zeta=zeta,output_texture_resolution_ind=res,time_ratio=r) :
        img = im[0][0].permute(1,2,0).cpu()
        img = (img * 255).clip(0, 255).numpy().astype(np.uint8)
        last_img = img
        list_of_steps = im[1]
        yield img
    final_img = last_img
    for step in list_of_steps :
        
        img = step[0].permute(1,2,0).cpu()
        img = (img * 255).clip(0, 255).numpy().astype(np.uint8)
        final_steps.append(img)
    img_steps = final_steps
    global maximum_step_number
    maximum_step_number=  len(list_of_steps)-1
## Globals update
def update_texture(tex1):
    global name
    name = tex1

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

def update_zoom_factor(i_z):
    global zoom_factor
    zoom_factor = i_z
    
def update_res(i_r):
    global res
    res = i_r

def update_zeta(i_z):
    global zeta
    zeta = i_z

def udpate_viewed_step(i_s):
    return img_steps[i_s], gr.update(maximum = maximum_step_number)

# Interface

load_models_stylization()
      
def demo_stylization():
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
            input_img = gr.Image(
                type="filepath",
                label="Input",
                elem_classes="full_height",
            )
            with gr.Column():
                preview_img = gr.Image(
                    type="numpy",
                    label="Output",
                    elem_classes="output-image-fill fixed_height_230",
                    interactive=False,
                )
                in_step_slider = gr.Slider(
                    minimum=0.,
                    maximum= maximum_step_number,
                    value = 0.,
                    step = 1.0,
                    label="View scale N",
                    interactive = False
                )
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                in_zoom_factor = gr.Slider(
                    minimum=0.1,
                    maximum=2.,
                    value=zoom_factor,
                    label="Zoom Factor",
                )
                in_zeta = gr.Slider(
                    minimum=0.,
                    maximum=1.,
                    label="Zeta",
                    value=zeta
                )
                in_res = gr.Slider(
                    minimum=1.,
                    maximum=5.,
                    step=1.,
                    label="Resolution",
                    value=res
                )
            with gr.Column(scale=1):
                generate_btn = gr.Button("Stylize", variant="primary", visible=True, interactive=True)
                cancel_btn_stylize = gr.Button("Cancel", variant="stop", visible=False)
                clear_cuda_cache_btn = gr.Button("Clear CUDA Cache", variant='stop', visible=True, interactive=True)
                in_drop_tex = gr.Dropdown(
                    choices=['wall','carpet','rust','crepe','ananaskin','ananaskin2','gold', "blue_up"],
                    label="Texture",
                    info="Texture used as the base style for the style transfer",
                    value=name,
                )

        # Event binding
        in_zoom_factor.change(
            fn = update_zoom_factor,
            inputs= [in_zoom_factor]
        )
        in_zeta.change(
            fn = update_zeta,
            inputs= [in_zeta]
        )
        in_res.change(
            fn = update_res,
            inputs= [in_res]
        )
        in_drop_tex.change(
            fn = lambda : gr.update(interactive=False, value="Loading models"),
            outputs=generate_btn
        ).then(
            fn=update_texture,
            inputs=[in_drop_tex],
            outputs=[]
        ).then(
            fn = load_models_stylization,
            outputs=generate_btn
        )
        stylize_process = generate_btn.click(
            fn = lambda : (
                gr.update(interactive=False,visible=False),
                gr.update(interactive=True, visible= True),
                gr.update(interactive=False, visible= True),
            ),
            outputs=[generate_btn,cancel_btn_stylize, in_step_slider]
        ).then(
            fn=stylize,
            inputs=input_img,
            outputs=preview_img,
            show_progress=False,
        ).success(
            fn= lambda : gr.update(maximum = maximum_step_number),
            outputs = in_step_slider
        ).then(
            fn= lambda : gr.update(value = maximum_step_number, interactive = True),
            outputs = in_step_slider
        )
        stylize_process.then(
            fn = lambda : (
                gr.update(interactive=True,visible=True),
                gr.update(interactive=False, visible=False)
            ),
            outputs=[generate_btn,cancel_btn_stylize]
        )
        
        cancel_btn_stylize.click(
            fn=lambda: (
                gr.update(interactive=True, visible=True),
                gr.update(interactive = False,visible=False)),
            outputs=[generate_btn, cancel_btn_stylize],
            cancels=[stylize_process],
            queue=False
	    )
        
        in_radio_nc.change(
            fn = lambda : gr.update(interactive=False, value="Loading models"),
            outputs=generate_btn
        ).then(
            fn=update_parameter_number,
            inputs=[in_radio_nc],
            outputs=[]  
        ).then(
            fn=load_models_stylization,
            outputs=generate_btn
        )
        clear_cuda_cache_btn.click(
            fn = clear_cuda_cache
        )
    with gr.Row(equal_height=True,):   
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
        
        in_step_slider.change(
            fn = udpate_viewed_step,
            inputs=in_step_slider,
            outputs=[preview_img,in_step_slider],
            show_progress=False,
            show_progress_on = preview_img
        )
    with gr.Accordion(label="Documentation", open=True):
        with open(DOCUMENTATION_PATH,'r') as f : 
            gr.Markdown(f.read(),latex_delimiters=[{ "left": "$$", "right": "$$", "display": True },{"left": "$", "right": "$", "display": False},])
if __name__ == "__main__":
    with gr.Blocks() as demo:
        demo_stylization()
    demo.queue().launch()
        