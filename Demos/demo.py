# code from Mahé DUVAL
import sys
from pathlib import Path

# Add parent directory to path for imports and file pathsto work from the Demos folder more easily
sys.path.insert(0, str(Path(__file__).parent.parent))

# UI Imports
import gradio as gr
from Demos.utilities.theme import *
from Demos.sub_demos.spatial_interpolation import demo_spatial_interpolation
from Demos.sub_demos.stylization import demo_stylization
from Demos.sub_demos.interpolate_two_imgs import demo_interp
from Demos.sub_demos.spatial_linear_interpolation import demo_lin_interp

with gr.Blocks() as demo:
    gr.HTML(HTML_LOGO_HEADER)
    gr.HTML(HTML_HEADER + HTML_AUTHORS)
    with gr.Tabs(selected="a"):
        with gr.Tab("Spatial Interpolation", id="a"):
            demo_spatial_interpolation()
        with gr.Tab("Stylization", id="b"):
            demo_stylization()
        with gr.Tab("Interpolate 2 images", id="c"):
            demo_interp()
        with gr.Tab("Spatial linear interpolation", id="d"):
            demo_lin_interp()
    gr.HTML(HTML_FOOTER)


demo.queue().launch(css=CUSTOM_CSS,head=HTML_CUSTOM_HEAD, theme=theme)