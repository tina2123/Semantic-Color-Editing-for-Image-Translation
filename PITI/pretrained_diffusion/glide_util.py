import os
from typing import Tuple
from . import dist_util
import PIL
import numpy as np
import torch as th
import cv2
import json
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import torch
import torch as th
import numpy as np
from .script_util import (
    create_gaussian_diffusion,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from pretrained_diffusion import chosen_color, category, file_path
import pretrained_diffusion
COCO_ANNOTATION_FILE = 'C:\\Users\\Bouth\\OneDrive\\Desktop\\computer_vision\\data\\annotations_2017\\instances_val2017.json'
first_person_region = None
def get_person_mask_from_image(hr_img):
    """
    Get the mask for a person from the given image using COCO annotations.
    
    Parameters:
    - hr_img: PIL Image object for which the mask needs to be fetched
    
    Returns:
    - mask: Binary mask with the 'person' region highlighted
    """

    image_file_name = pretrained_diffusion.file_path# '000000570169.jpg' 
    with open(COCO_ANNOTATION_FILE, 'r') as f:
        data = json.load(f)

    person_category_id = pretrained_diffusion.category  # As per COCO annotations
    
    selected_image = next((image for image in data['images'] if image['file_name'] == image_file_name), None)

    mask = np.zeros(hr_img.size[::-1], dtype=np.uint8)  # Initialize empty mask
    
    if selected_image:
        image_id = selected_image['id']
        image_annotations = [a for a in data['annotations'] if a['image_id'] == image_id and a['category_id'] == person_category_id]
        
        original_width, original_height = selected_image['width'], selected_image['height']
        scale_x = hr_img.width / original_width
        scale_y = hr_img.height / original_height
       
        
        for annotation in image_annotations:
            segmentation = annotation.get('segmentation', [])
            
            # Convert segmentation data into suitable format for cv2.fillPoly
            polygons = [np.array(seg, np.int32).reshape((-1, 1, 2)) for seg in segmentation]
            
            # Scale the polygons
            scaled_polygons = []
            for poly in polygons:
                scaled_poly = poly.copy()
                scaled_poly[:, :, 0] = (scaled_poly[:, :, 0] * scale_x).astype(np.int32)
                scaled_poly[:, :, 1] = (scaled_poly[:, :, 1] * scale_y).astype(np.int32)
                scaled_polygons.append(scaled_poly)
            
            cv2.fillPoly(mask, scaled_polygons, 1)
    
    return mask

def custom_noise(batch_size, side_y, side_x, center_size=(40, 40)):
    # Start with standard Gaussian noise
    noise = th.randn((batch_size, 3, side_y, side_x))
    
    # Define the center region
    y_center, x_center = side_y // 2, side_x // 2
    y_start, y_end = y_center - center_size[0] // 2, y_center + center_size[0] // 2
    x_start, x_end = x_center - center_size[1] // 2, x_center + center_size[1] // 2

    # Amplify the blue channel in the center
    noise[:, 2, y_start:y_end, x_start:x_end] += 2.0  # Adjust the 2.0 as needed

    return noise

# Sample from the base model.

#@th.inference_mode()
def sample(
    glide_model,
    glide_options,
    side_x,
    side_y,
    prompt,
    batch_size=1,
    guidance_scale=4,
    device="cpu",
    prediction_respacing="100",
    upsample_enabled=False,
    upsample_temp=0.997,
    mode = '',
):

    eval_diffusion = create_gaussian_diffusion(
        steps=glide_options["diffusion_steps"],
        learn_sigma=glide_options["learn_sigma"],
        noise_schedule=glide_options["noise_schedule"],
        predict_xstart=glide_options["predict_xstart"],
        rescale_timesteps=glide_options["rescale_timesteps"],
        rescale_learned_sigmas=glide_options["rescale_learned_sigmas"],
        timestep_respacing=prediction_respacing
    )
 
    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    cond_ref   =  prompt['ref']
    uncond_ref = th.ones_like(cond_ref) 
    
    model_kwargs = {}
    model_kwargs['ref'] =  th.cat([cond_ref, uncond_ref], 0).to(dist_util.dev())
    hr_img = Image.open('C:\\Users\\Bouth\\OneDrive\\Desktop\\computer_vision\\data\\val2017\\val2017\\000000570169.jpg')

    person_mask = get_person_mask_from_image(hr_img)
    person_mask_tensor = th.from_numpy(person_mask).to(device).float()
    person_mask_tensor = person_mask_tensor.unsqueeze(0).unsqueeze(1)  # (B, C, H, W) shape
    person_mask_tensor = F.interpolate(person_mask_tensor, size=(side_y, side_x), mode='nearest')

    


    def cfg_model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = glide_model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
 
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)

        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)
   

    def cfg_model_fn_noise_failed(x_t, ts, **kwargs):
        global first_person_region
        
        correct_device = x_t.device 
        
        # Your existing code
        half = x_t[: len(x_t) // 2].to(correct_device)
        combined = th.cat([half, half], dim=0)
        model_out = glide_model(combined, ts, **kwargs).to(correct_device)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        
        eps = eps.to(correct_device)
        rest = rest.to(correct_device)
        
        to_pil = ToPILImage()
        
        batch_size = eps.size(0) // 2
        modified_images = []
        
        for i in range(batch_size):
            img_tensor = eps[i].detach().to(correct_device)
            img_pil = to_pil(img_tensor.cpu())
            
            person_mask_tensor = get_person_mask_from_image(img_pil)
            if isinstance(person_mask_tensor, np.ndarray):
                person_mask_tensor = torch.tensor(person_mask_tensor).to(img_tensor.dtype).to(correct_device)
            
            if first_person_region is None:
                first_person_region = img_tensor * person_mask_tensor.to(img_tensor.dtype).to(correct_device)
            
            modified_image = img_tensor * (1 - person_mask_tensor.to(img_tensor.dtype).to(correct_device)) + first_person_region
            modified_images.append(modified_image)
        
        half_eps = th.stack(modified_images, dim=0).to(correct_device)
        
        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)




    def cfg_model_fn_noise(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = glide_model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        
        half_eps = half_eps * (1 - person_mask_tensor) + x_t[: len(x_t) // 2] * person_mask_tensor

        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)

    if upsample_enabled:
        model_kwargs['low_res'] = prompt['low_res'].to(dist_util.dev())
        noise = th.randn((batch_size, 3, side_y, side_x), device=device) * upsample_temp
        model_fn = glide_model # just use the base model, no need for CFG.
        model_kwargs['ref'] =  model_kwargs['ref'][:batch_size]

        samples = eval_diffusion.p_sample_loop(
        model_fn,
        (batch_size, 3, side_y, side_x),  # only thing that's changed
        noise=noise,
        device=device,
        clip_denoised=True,
        progress=False,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]

    else:
        model_fn = cfg_model_fn # so we use CFG for the base model.
        #noise = th.randn((batch_size, 3, side_y, side_x), device=device) 
        noise = custom_noise(batch_size, side_y, side_x).to(device)

        base_noise = custom_noise(batch_size, side_y, side_x).to(device)
        noise = th.cat([base_noise, base_noise], 0)
 
        samples = eval_diffusion.p_sample_loop(
            model_fn,
            (full_batch_size, 3, side_y, side_x),  # only thing that's changed
            noise=noise,
            device=device,
            clip_denoised=True,
            progress=False,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
    
    return samples

 