import os, sys
import numpy as np
from PIL import Image
import math
from omegaconf import OmegaConf
from typing import List, Optional
import cv2
import time
from einops import rearrange, repeat
import torch

from spmr.util import instantiate_from_config

from insightface.app import FaceAnalysis
from spmr.face_parsing.facemaskdetector import FaceMaskDetector
from spmr.panohead_cropping.recrop_images import panohead_crop
from spmr.panohead_cropping.TDDFA import TDDFA
from spmr.panohead_cropping.FaceBoxes import FaceBoxes

from spmr.sampling_scipts import sampling_pipeline

np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.complex = np.complex_
np.object = np.object_
np.unicode = np.unicode_
np.str = np.str_

import gradio as gr

ROOT_DIR = os.path.dirname(__file__)
CROP = True ## 
remove_bg =  True


def load_model(config: str, device: str, num_frames: int = 8, num_steps: int = 50, verbose: bool = False,):
    config = OmegaConf.load(config)
    print('Using ckpt:', config.model.params.ckpt_path)

    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()
    return model


@torch.no_grad()
def sample(
        image,
        angle_sampling: str = '360',
        num_steps: int = 50,
        scale: float = 3.5,
        num_frames: int = 7,
        seed: int = 42,
        size: int = 512,
):
    num_frames = num_frames + 1
    print(f'Running for sampling inputs : {scale=} {num_frames=} {seed=}')

    if angle_sampling == '360':
        desired_angles = torch.tensor(np.linspace(0, 360, num_frames+1), device=device, dtype=torch.float)[:-1]
        desired_angles = torch.stack([torch.zeros_like(desired_angles), desired_angles], dim=0).transpose(1, 0)
    elif angle_sampling == '180':
        desired_angles = torch.tensor(np.linspace(0, 180, num_frames+1), device=device, dtype=torch.float)[:-1]
        desired_angles = torch.stack([torch.zeros_like(desired_angles), desired_angles], dim=0).transpose(1,0)
    else:
        desired_angles = torch.tensor(np.concatenate([np.zeros(1), np.linspace(-60, 60, num_frames)]),
                                        device=device, dtype=torch.float)[:-1]
        desired_angles = torch.stack([desired_angles, torch.zeros_like(desired_angles)], dim=0).transpose(1,0)
        
    print('Desired viewpoints : ', desired_angles.transpose(1,0))

    dtype = torch.float32
    image = np.array(image)[:,:,:3]
    image_input_arcface = image[:, :, [2, 1, 0]]
    faces = app.get(image_input_arcface)
    faces = sorted(faces, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
        -1]  # select largest face (if more than one detected)
    if len(faces) == 0:
        print('No face found')
        return []

    ## Crop image following Panohead pipeline
    with torch.inference_mode():
        lmks_ = faces['kps']
        cropped_img, *_ = panohead_crop(image, lmks_, face_boxes=face_boxes,
                                        tddfa=tddfa, padd=True)
        if cropped_img is None:
            print('No cropping')
            return []
        cropped_img = cv2.resize(cropped_img, (size, size))

    rgb = np.array(Image.fromarray(cropped_img).resize((size, size))) / 255.

    ## Remove background
    rgb_pt = torch.tensor(rgb.transpose(2, 0, 1)[None], device=device, dtype=torch.float)
    mask = maskdetector(rgb_pt).cpu().numpy()[:, :, None]
    rgb = rgb * mask + (1 - mask) * np.ones_like(rgb)

    rgb_arcface = (rgb * 255).astype(np.uint8)[:, :, [2, 1, 0]]
    faces = app.get(rgb_arcface)
    if len(faces) == 0:
        print('No faces found (2)')
        return []
    faces = sorted(faces, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
        -1]  # select largest face (if more than one detected)
    id_emb = torch.tensor(faces['embedding'], dtype=dtype)[None].to(device)
    id_emb = id_emb / torch.norm(id_emb, dim=1, keepdim=True)  # normalize embedding

    input_image = (rgb * 255).astype(np.uint8)

    samples, samples_normals = sampling_pipeline(input_image, id_emb, model, desired_angles, scale, num_steps, 
                                                 num_frames=num_frames, device='cuda')

    for i in range(samples.shape[0]):
        pred_faces = app.get(
            (255 * samples[i].permute(1, 2, 0).cpu()).numpy().astype(np.uint8)[:, :, [2, 1, 0]])
        if len(pred_faces) == 0:
            print(f"{i} -> No face")
        else:
            pred_faces = \
            sorted(pred_faces, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
                -1]  # select largest face (if more than one detected)

            pred_id_emb = torch.tensor(pred_faces['embedding'], dtype=dtype)[None].to(device)
            pred_id_emb = pred_id_emb / torch.norm(pred_id_emb, dim=1, keepdim=True)  # normalize embedding

            id_distance = 1 - torch.nn.functional.cosine_similarity(pred_id_emb, id_emb)
            print(f"{i} -> {id_distance.item()}")

    return [(255 * sample.permute(1,2,0)).cpu().numpy().astype(np.uint8) for sample in [*samples, *samples_normals]]



device = torch.device('cuda:0')
model_config = "configs/inference.yaml"

## Initialize all the necessary models

# 1. Face Mask Detector
maskdetector = FaceMaskDetector(save_path=os.path.join(ROOT_DIR,'weights/79999_iter.pth'),
                                ids2keep=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]).to(device)
maskdetector.eval()

# 2. Panohead cropping models
cfg = {'arch': 'mobilenet',
       'widen_factor': 1.0,
       'checkpoint_fp': os.path.join(ROOT_DIR, 'spmr/panohead_cropping/weights/mb1_120x120.pth'),
       'bfm_fp': os.path.join(ROOT_DIR, 'spmr/panohead_cropping/configs/bfm_noneck_v3.pkl'),
       'size': 120,
       'num_params': 62}
device_index = device.index if device.index is not None else 0
tddfa = TDDFA(gpu_mode='gpu',
              gpu_id= device_index if device_index is not None else 0,
              **cfg)
face_boxes = FaceBoxes()

# 3. ArcFace 
app = FaceAnalysis(name='antelopev2', root=os.path.join(ROOT_DIR, 'weights'),
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(448, 448))

# 4. SpinMeRound 
model = load_model(
    model_config,
    'cpu',
)
model = model.to(device)
seed = 42

_TITLE = r"""
<h1>SpinMeRound: Consistent Multi-View Identity Generation Using Diffusion Models</h1>
"""
_DESCRIPTIONS = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://spin-me-round.github.io/' target='_blank'><b>SpinMeRound: Consistent Multi-View Identity Generation Using Diffusion Models</b></a>.<br>

Steps:<br>
1. Upload an image with a single face. If your image is already tightly cropped around the face, detection might not work properly. Try using an image with a bit more background or space around the face instead.
2. Choose your sampling angles.
3. Click <b>Submit</b> to generate new viepoints of the subject.
"""

_CITATION = r"""
---
📝 **Citation**
<br>
If you find SpinMeRound helpful for your research, please consider citing our paper:
```bibtex

      @misc{galanakis2025spinmeroundconsistentmultiviewidentity,
        title={SpinMeRound: Consistent Multi-View Identity Generation Using Diffusion Models}, 
        author={Stathis Galanakis and Alexandros Lattas and Stylianos Moschoglou and Bernhard Kainz and Stefanos Zafeiriou},
        year={2025},
        eprint={2504.10716},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2504.10716}, 
    }
    
```
"""

# Compose demo layout & data flow.
demo = gr.Blocks(title=_TITLE)

css = '''
    .gradio-container {width: 85% !important}
    '''
with gr.Blocks(css=css) as demo:
    gr.Markdown('# ' + _TITLE)
    gr.Markdown(_DESCRIPTIONS)

    with gr.Row():
        with gr.Column(scale=0.9, variant='panel'):

            image_block = gr.Image(type='pil', image_mode='RGBA',
                                   label='Input image of single object')
            use_sapies = False
            angle_sampling = gr.Radio(["360", "180", "Yaw"], label="Sampling angles", info="Predefined_angles for sampling", value='360')

            with gr.Accordion("Advanced options", open=False):
                num_steps = gr.Slider(
                    label="Number of sample steps",
                    minimum=20,
                    maximum=100,
                    step=1,
                    value=50,
                )
                scale = gr.Slider(
                    label="Guidance scale",
                    minimum=1,
                    maximum=10.0,
                    step=0.1,
                    value=3.5,
                )

                num_frames = gr.Slider(
                    label="Number of output frames",
                    minimum=0,
                    maximum=15,
                    step=1,
                    value=7,
                )
        with gr.Column():
            gallery = gr.Gallery(label="Generated Images")

    run_button = gr.Button(value="Run")
    ips = [image_block, angle_sampling, num_steps, scale, num_frames]
    run_button.click(
        fn=sample,
        inputs=ips,
        outputs=[gallery],
        queue=False,
        api_name=False,
    )

    gr.Markdown(_CITATION)
demo.launch()