# code from Mahé DUVAL
import gradio as gr
from gradio.themes.utils import colors

# Custom color palette matching the GREYC logo color scheme
greyc_color = colors.Color(
    name="fuchsia",
    c50="#FBDEF4",
    c100="#F5B4E8",
    c200="#EE86DB",
    c300="#E056CB",
    c400="#C139AF",
    c500="#9C2A8D",
    c600="#76216B",
    c700="#631D59",
    c800="#501948",
    c900="#3E1438",
    c950="#2C0D27",
)
"""

"""

# Custom theme based on the default Gradio theme but with some changes to match the GREYC color scheme and look
theme = gr.themes.Default(
    primary_hue=greyc_color,
    radius_size="lg",
    font_mono=[gr.themes.GoogleFont('IBM Plex Mono'), 'ui-monospace', 'Consolas', 'monospace'],
).set(
    body_text_weight='300',
    button_border_width_dark='1px',
    button_transition='all 0.3s ease',
    button_large_radius='*radius_md',
    button_small_radius='*radius_md',
    button_primary_border_color_hover_dark='*primary_900',
    button_secondary_border_color_hover_dark='*neutral_700',
    button_cancel_background_fill='*secondary_300',
    button_cancel_border_color_dark='hsl(0, 73%, 30%)',
    button_cancel_border_color_hover='hsl(0, 73%, 60%)',
    button_cancel_border_color_hover_dark='hsl(0, 73%, 60%)',
    loader_color="#78216b",
)

HTML_CUSTOM_HEAD = """
    <title>
        SIMuLDiTex
    </title>
    <meta name="SIMuLDiTex demo app" content="A Single Image Multiscale and Lightweight Diffusion Model for Texture Synthesis">
"""

# Footer and links to the paper and code
HTML_FOOTER ="""
    <div style="text-align: center; padding: 0px;;margin-top:30px;">
        <a href="https://hal.science/hal-04994907">HAL</a>
        <a href="https://github.com/PierrickCh/SIMuLDiTex">Github</a>
        <!-- <a href="https://arxiv.org/abs/2509.22318" >ArXiv</a> --!>
    </div>
"""

HTML_SEPARATOR = """
    <div class='separator'>
    </div>
"""
HTML_V_SEPARATOR = """
    <div class='v_separator'>
    </div>
"""

HTML_LOGO_HEADER ="""
    <header>
        <a href="https://www.greyc.fr/">
            <img src="https://greycflix.greyc.fr/demo-portal/images/logo-GREYC-dark.svg" style="position: absolute; width: 12em;">
        </a>
    </header>
"""

HTML_HEADER = """
    <div style="text-align: center; padding: 0px;">
        <h1 id="title">
            SIMuLDiTex
        </h1>
        <p id="subtitle">
            A Single Image Multiscale and Lightweight Diffusion Model for Texture Synthesis
        </p>
    </div>
"""

HTML_AUTHORS = """
    <div style="text-align: center; padding: 0px;">
        <p id="authors">Pierrick Chatillon, Julien Rabin, David Tschumperlé</p>
    </div>
"""

CUSTOM_CSS = """

footer {visibility: hidden}

#title{
    font-size: 48px; 
    font-weight: 700; 
    margin: 0; 
    color: #ffffff; 
    letter-spacing: -0.02em;
}

#subtitle{
    font-size: 20px; 
    color: #86868b; 
    margin:0;
    font-weight: 400;
}
#authors{
    font-size: 15px; 
    color: #86868b; 
    margin:0;
    font-style: italic;
}

.radio_group .wrap {
    display: grid !important;
    grid-template-columns: 1fr 1fr;
}

.generating {
    border-color: #76216B !important;
}

.separator{
    border : none !important;
    background-color: var(--button-primary-background-fill-hover) !important;
    padding:2px;
    height: 4px !important;
    border-radius: 5px!important;
}

.v_separator{
    border : none !important;
    background-color: var(--button-primary-background-fill-hover) !important;
    padding:2px;
    width: 4px !important;
    height: 100% !important;
    border-radius: 5px!important;
}


.output-image-fill img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
    padding: 15px;
}
.output-image-fill > div {
    width: 100% !important;
    height: 100% !important;
}

.flex_display{
    display: flex;
    flex-wrap: nowrap;
}
.tabitem{
	display:flex;
	flex:auto;
}
.tabs{
	display: flex;
	flex-basis: content;
	flex-direction: column;
	flex-grow: inherit;
}

.filled_flex_display {display: flex !important; align-content: stretch; justify-content: space-between;}
.filled_flex_display > div{display: grid !important; align-content: stretch; align-items: stretch; justify-items: stretch; flex-grow: 1 !important;}
.full_width {
    width: -webkit-fill-available !important;
    }


.full_height {
    height: -webkit-fill-available !important;
    }

.column{
    justify-content: space-between;
}


/**#input_row{height: 300px !important;}**/
/**#input_column{height: -webkit-fill-available;}**/
/**#output_row{height : 256px !important;}**/


.shrink{
    flex-shrink: 1 !important;
    flex-grow: 0 !important;
    min-inline-size: fit-content;
    min-width: unset !important;
}


.full_size_image { 
	margin: 0px !important;
	width: 100% !important;
	height: 100% !important;
}
.fixed_height_image_row{
    height : 350px !important;
}
.fixed_height_image_150{
    height: 150px;
}
.fixed_height_230{
    height:230px;
}
.fixed_height_image_row_550{
    height : 550px !important;
}
.min_height{
    min-height : min-content;
}
.output_column {
height: -webkit-fill-available;
}
#input_general_settings{
	height: -webkit-fill-available;
	justify-content: space-between;
}
[data-testid="imageslider-image"] {
    max-block-size: -webkit-fill-available !important;
}
"""