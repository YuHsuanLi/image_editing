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
from segment_anything import SamPredictor, sam_model_registry

from torchvision import transforms
warnings.filterwarnings("ignore")

device = 'cuda'
sam = sam_model_registry["default"](checkpoint="./fix_color/sam_vit_h_4b8939.pth")
sam.to(device=device)

BOX_TRESHOLD = 0.45
TEXT_TRESHOLD = 0.25
save_folder = "./method2/incongruent"
edited_image_folder = "./HQ-Edit/InfEdit/InfEdit_Results/Incongruent"
matching_file = "./HQ-Edit/InfEdit/incongruent_parsed.txt"

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
input_description_path = "/data/eva/multimodal/report4/fix_color/incongruent2.json"
input_description = json.load(open(input_description_path, "r"))
counter = 0

with open(matching_file, "r") as f:
    lines = f.readlines()
parquet_id_pre = ''
non_bbox = 0
for _, i in enumerate(range(1871)):
    i = 7
    # parquet_id, row_id = lines[i].strip().split(",")
    splited_lines = lines[i].strip().split(";")
    parquet_id = splited_lines[0]
    row_id = splited_lines[1]
    if parquet_id != parquet_id_pre:
        df = pd.read_parquet(f'/data/eva/multimodal/datapreprocessing/test_parquet/test_{parquet_id}.parquet', engine='pyarrow')
    parquet_id_pre = parquet_id
    bytes_im = df["input_image"].iloc[int(row_id)]
    pil_image = Image.open(io.BytesIO(bytes_im)).convert("RGB").resize((512, 512))
    image_source = np.asarray(pil_image)
    image, _ = transform(pil_image, None)
    difference = json.loads(input_description[str(i)]['difference'])
    old_caption = difference["changed"]["obj"]
    pil_image.save(f"{save_folder}/{i}_{parquet_id}_{row_id}_ori.jpg")
    predictor = SamPredictor(sam)
    predictor.set_image(image_source)
    
    boxes, logits, phrases = predict(
        model=model, 
        image=image, 
        caption=old_caption, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD
    )
    if len(boxes) == 0:
        pil_image.save(f"{save_folder}/{i}_{parquet_id}_{row_id}_no_box.jpg")
        non_bbox += 1
        print(f"non_bbox: {non_bbox}")
        continue
    h, w, _ = image_source.shape
    masks = []
    for box in boxes:
        boxes_xyxy = box_convert(boxes=box* torch.Tensor([w, h, w, h]), in_fmt="cxcywh", out_fmt="xyxy").numpy()
        mask, _, _ = predictor.predict(point_coords=None, point_labels=None, box=boxes_xyxy, multimask_output=False)
        masks.append(mask)
    masks = np.array(masks)
    masks = np.any(masks, axis=0)
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB
    mask = masks[0]  # shape: (1020, 941), dtype: bool

    # Convert to uint8 and scale (0 or 255)
    image_mask = Image.fromarray((mask * 255).astype(np.uint8))
    

    image_source = Image.fromarray(image_source)
    annotated_frame = Image.fromarray(annotated_frame)
    edited_image_path = edited_image_folder + "/" + str(parquet_id) + '_' + str(row_id) + ".png"
    edited_image = Image.open(edited_image_path).convert("RGB").resize((512, 512))

    image1_np = np.array(pil_image).astype(np.float32) / 255.0
    image2_np = np.array(edited_image).astype(np.float32) / 255.0

    if len(image1_np.shape) == 3 and mask.ndim == 2:
        mask = np.stack([mask]*3, axis=-1)

    # Fuse
    fused_np = image2_np * mask + image1_np * (1 - mask)

    # Convert back to PIL Image
    fused_img = Image.fromarray((fused_np * 255).astype(np.uint8))
    # save mask, new_output
    fused_img.save(f"{save_folder}/{i}_{parquet_id}_{row_id}.jpg")
    image_mask.save(f"{save_folder}/{i}_{parquet_id}_{row_id}_mask.jpg")
    annotated_frame.save(f"{save_folder}/{i}_{parquet_id}_{row_id}_annotated_frame.jpg")

    counter += 1
