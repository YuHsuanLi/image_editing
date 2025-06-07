import numpy as np
import pandas as pd
import argparse
from functools import partial
import cv2
import requests
import PIL
from io import BytesIO
from PIL import Image
from pathlib import Path
import warnings
import torch
from torchvision.ops import box_convert
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict
import groundingdino.datasets.transforms as T
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionPipeline, RePaintScheduler
import ast
import io
import json

from torchvision import transforms
warnings.filterwarnings("ignore")

BOX_TRESHOLD = 0.45
TEXT_TRESHOLD = 0.25
save_folder = "method1"
input_file_path = "./Data/High-Low/high_low_final_w_idx.csv"

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model  

def generate_masks_with_grounding(image_source, boxes):
    h, w, _ = image_source.shape
    boxes_unnorm = boxes * torch.Tensor([w, h, w, h])
    boxes_xyxy = box_convert(boxes=boxes_unnorm, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    mask = np.zeros_like(image_source)
    for box in boxes_xyxy:
        x0, y0, x1, y1 = box
        mask[int(y0):int(y1), int(x0):int(x1), :] = 255
    return mask

ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swint_ogc.pth"
ckpt_config_filename = "GroundingDINO_SwinT_OGC.cfg.py"
model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)



# select color related idx (low)
# ask gpt to find the main part of color changing part
# load the data
df = pd.read_csv(input_file_path)
granularity = df["granularity"].values
category = df["category"].values
low_color_idx = np.where((granularity == 'low') & (category == 'color'))[0]


# method 1
# grounding then use repaint to do color changing
# need chatgot to find the color changing part. store in a file
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")
transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
input_description_path = "color_differences.json"
input_description = json.load(open(input_description_path, "r"))
counter = 0
for i in low_color_idx:
    bytes_im = ast.literal_eval(df["input_image"].iloc[i])
    pil_image = Image.open(io.BytesIO(bytes_im)).convert("RGB")#.resize((512, 512))
    image_source = np.asarray(pil_image)
    image, _ = transform(pil_image, None)
    difference = json.loads(input_description[str(i)]['difference'])
    old_caption = difference["changed"]["ori"] + difference["changed"]["obj"]
    caption = difference["changed"]["new"] + difference["changed"]["obj"]
    granularity = df["granularity"].iloc[i]
    category = df["category"].iloc[i]
    parquet_id = df["parquet_id"].iloc[i]
    row_id = df["row_id"].iloc[i]
    boxes, logits, phrases = predict(
        model=model, 
        image=image, 
        caption=old_caption, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB
    image_mask = generate_masks_with_grounding(image_source, boxes)
    image_source = Image.fromarray(image_source)
    annotated_frame = Image.fromarray(annotated_frame)
    image_mask = Image.fromarray(image_mask)
    
    image_source_for_inpaint = image_source.resize((512, 512))
    image_mask_for_inpaint = image_mask.resize((512, 512))
    prompt = caption
    #image and mask_image should be PIL images.
    #The mask structure is white for inpainting and black for keeping as is
    image_inpainting = pipe(prompt=prompt, image=image_source_for_inpaint, mask_image=image_mask_for_inpaint).images[0]
    image_inpainting = image_inpainting.resize((image_source.size[0], image_source.size[1]))

    # save mask, new_output
    image_inpainting.save(f"{save_folder}/{i}_{parquet_id}_{row_id}.jpg")
    image_mask.save(f"{save_folder}/{i}_{parquet_id}_{row_id}_mask.jpg")
    annotated_frame.save(f"{save_folder}/{i}_{parquet_id}_{row_id}_annotated_frame.jpg")

    counter += 1
