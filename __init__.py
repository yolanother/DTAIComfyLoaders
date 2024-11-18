from io import BytesIO

import requests as requests
import torch

import os
import hashlib
import numpy as np
import safetensors.torch

import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils

import comfy.clip_vision

import comfy.model_management

import comfy
from PIL import Image, ImageOps
import requests

import folder_paths
from comfy_extras.chainner_models import model_loading
from custom_nodes.DTAIComfyLoaders.loaders import *


class DTNodeCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (checkpoints.list(), ), }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "DoubTech/Loaders"

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = checkpoints.download(ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True,
                                                    embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out


class DTVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "vae_name": (vae.list(), )}}
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"

    CATEGORY = "DoubTech/Loaders"

    #TODO: scale factor?
    def load_vae(self, vae_name):
        v = comfy.sd.VAE(ckpt_path=vae.download(vae_name))
        return (v,)


class DTLoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "lora_name": (lora.list(), ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    CATEGORY = "DoubTech/Loaders"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = lora.download(lora_name)
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                loaded_lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if loaded_lora is None:
            loaded_lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, loaded_lora, strength_model, strength_clip)
        return (model_lora, clip_lora)


class DTLorasLoader:
    def __init__(self):
        self.loaded_loras = dict()
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "loras": ("STRING", {
                                    "multiline": False,
                                    "default": "disney-princess:1.0:1.0,cyberpunk-animal:.5:1"
                                }),
                              }}
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    CATEGORY = "DoubTech/Loaders"

    def load_lora(self, model, clip, loras):
        # split loras by commas
        loras = loras.split(",")

        model_lora = model
        model_clip = clip

        for l in loras:
            # split each lora by colons. 0: name, 1: strength_model (optional=1), 2: strength_clip (optional=1)
            lora_request = l.split(":")
            lora_name = lora_request[0]
            if len(lora_name) == 0:
                continue

            # if the name doesn't have an extension add .safetensors
            if not lora_name.endswith(".safetensors") and len(lora_name) > 0:
                lora_name += ".safetensors"
            strength_model = float(lora_request[1]) if len(lora_request) > 1 else 1.0
            strength_clip = float(lora_request[2]) if len(lora_request) > 2 else 1.0

            if lora_name not in self.loaded_loras:
                lora_path = lora.download(lora_name)
                loaded_lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_loras[lora_name] = loaded_lora
            else:
                loaded_lora = self.loaded_loras[lora_name]
            model_lora, model_clip = comfy.sd.load_lora_for_models(model_lora, model_clip, loaded_lora, strength_model, strength_clip)

        return (model_lora, model_clip,)

class DTCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name": (clip.list(), ),
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "DoubTech/Loaders"

    def load_clip(self, clip_name):
        clip_path = clip.download(clip_name)
        c = comfy.sd.load_clip(ckpt_path=clip_path, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return (c,)


class DTCLIPVisionLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name": (clipVision.list(), ),
                             }}
    RETURN_TYPES = ("CLIP_VISION",)
    FUNCTION = "load_clip"

    CATEGORY = "DoubTech/Loaders"

    def load_clip(self, clip_name):
        clip_path = clipVision.download(clip_name)
        clip_vision = comfy.clip_vision.load(clip_path)
        return (clip_vision,)


class DTStyleModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "style_model_name": (style.list(), )}}

    RETURN_TYPES = ("STYLE_MODEL",)
    FUNCTION = "load_style_model"

    CATEGORY = "DoubTech/Loaders"

    def load_style_model(self, style_model_name):
        style_model_path = style.download(style_model_name)
        style_model = comfy.sd.load_style_model(style_model_path)
        return (style_model,)


class DTGLIGENLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"gligen_name": (gligen.list(),)}}

    RETURN_TYPES = ("GLIGEN",)
    FUNCTION = "load_gligen"

    CATEGORY = "DoubTech/Loaders"

    def load_gligen(self, gligen_name):
        gligen_path = gligen.download(gligen_name)
        g = comfy.sd.load_gligen(gligen_path)
        return (g,)


class DTControlNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "control_net_name": (controlNet.list(), )}}

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"

    CATEGORY = "DoubTech/Loaders"

    def load_controlnet(self, control_net_name):
        controlnet_path = controlNet.download(control_net_name)
        controlnet = comfy.sd.load_controlnet(controlnet_path)
        return (controlnet,)


class DTDiffControlNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "control_net_name": (controlNetDiff.list(), )}}

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"

    CATEGORY = "DoubTech/Loaders"

    def load_controlnet(self, model, control_net_name):
        controlnet_path = controlNetDiff.download(control_net_name)
        controlnet = comfy.sd.load_controlnet(controlnet_path, model)
        return (controlnet,)


class DTunCLIPCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (unclipCheckpoint.list(), ),
                             }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CLIP_VISION")
    FUNCTION = "load_checkpoint"

    CATEGORY = "DoubTech/Loaders"

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = unclipCheckpoint.download(ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, output_clipvision=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out

class DTCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "config_name": (configs.list(), ),
                              "ckpt_name": (checkpoints.list(), )}}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "DoubTech/Loaders/Advanced"

    def load_checkpoint(self, config_name, ckpt_name, output_vae=True, output_clip=True):
        config_path = configs.download(config_name)
        ckpt_path = checkpoints.download(ckpt_name)
        return comfy.sd.load_checkpoint(config_path, ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))

class DTDiffusersLoader:
    @classmethod
    def INPUT_TYPES(cls):
        paths = []
        for search_path in folder_paths.get_folder_paths("diffusers"):
            if os.path.exists(search_path):
                for root, subdir, files in os.walk(search_path, followlinks=True):
                    if "model_index.json" in files:
                        paths.append(os.path.relpath(root, start=search_path))

        return {"required": {"model_path": (paths,), }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "DoubTech/Loaders/Advanced"

    def load_checkpoint(self, model_path, output_vae=True, output_clip=True):
        for search_path in folder_paths.get_folder_paths("diffusers"):
            if os.path.exists(search_path):
                path = os.path.join(search_path, model_path)
                if os.path.exists(path):
                    model_path = path
                    break

        return comfy.diffusers_load.load_diffusers(model_path, fp16=comfy.model_management.should_use_fp16(), output_vae=output_vae, output_clip=output_clip, embedding_directory=folder_paths.get_folder_paths("embeddings"))


class DTLoadLatent:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".latent")]
        return {"required": {"latent": [sorted(files), ]}, }

    CATEGORY = "DoubTech/Loaders"

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "load"

    def load(self, latent):
        latent_path = folder_paths.get_annotated_filepath(latent)
        latent = safetensors.torch.load_file(latent_path, device="cpu")
        samples = {"samples": latent["latent_tensor"].float()}
        return (samples, )

    @classmethod
    def IS_CHANGED(s, latent):
        image_path = folder_paths.get_annotated_filepath(latent)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, latent):
        if not folder_paths.exists_annotated_filepath(latent):
            return "Invalid latent file: {}".format(latent)
        return True

def load_image_from_url(url):
    try:
        # if the image is a data url, extract the base64 data and load it
        if url.startswith("data:"):
            import base64
            data = url.split(",")[1]
            data = base64.b64decode(data)
            image = Image.open(BytesIO(data))
            return image

        # Send a GET request to fetch the image data
        response = requests.get(url)

        # Check if the request was successful
        response.raise_for_status()

        # Read the image data and create a PIL image
        image = Image.open(BytesIO(response.content))

        return image

    except requests.exceptions.RequestException as e:
        print(f"Error loading image from URL: {url}")
        print(e)
        return None

class DTLoadImage:
    loaded_path = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("STRING", {
                    "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                    "default": "https://www.doubtech.ai/img/doubtech.ai-qrcode.png"
                })},
                }

    CATEGORY = "DoubTech/Image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    def load_image(self, image):
        i = load_image_from_url(image)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, mask)

    @classmethod
    def IS_CHANGED(s, image):
        return s.loaded_path != image

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not image:
            return "Invalid image file: {}".format(image)

        return True


class DTLoadImageMask:
    loaded_path = None
    _color_channels = ["alpha", "red", "green", "blue"]
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": ("STRING", {
                    "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                    "default": "https://www.doubtech.ai/img/doubtech.ai-qrcode.png"
                }),
                     "channel": (s._color_channels, ), }
                }

    CATEGORY = "DoubTech/Image"

    RETURN_TYPES = ("MASK",)
    FUNCTION = "load_image"
    def load_image(self, image, channel):
        i = load_image_from_url(image)
        i = ImageOps.exif_transpose(i)
        if i.getbands() != ("R", "G", "B", "A"):
            i = i.convert("RGBA")
        mask = None
        c = channel[0].upper()
        if c in i.getbands():
            mask = np.array(i.getchannel(c)).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)
            if c == 'A':
                mask = 1. - mask
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (mask,)

    @classmethod
    def IS_CHANGED(s, image, channel):
        return s.loaded_path != image

    @classmethod
    def VALIDATE_INPUTS(s, image, channel):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        if channel not in s._color_channels:
            return "Invalid color channel: {}".format(channel)

        return True

class DTUpscaleModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model_name": (upscalers.list(), ), }}

    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "DoubTech/Loaders"

    def load_model(self, model_name):
        model_path = upscalers.download(model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        out = model_loading.load_state_dict(sd).eval()
        return (out, )

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "DTCheckpointLoaderSimple": DTNodeCheckpointLoader,
    "DTVAELoader": DTVAELoader,
    "DTLoraLoader": DTLoraLoader,
    "DTCLIPLoader": DTCLIPLoader,
    "DTControlNetLoader": DTControlNetLoader,
    "DTDiffControlNetLoader": DTDiffControlNetLoader,
    "DTStyleModelLoader": DTStyleModelLoader,
    "DTCLIPVisionLoader": DTCLIPVisionLoader,
    "DTunCLIPCheckpointLoader": DTunCLIPCheckpointLoader,
    "DTGLIGENLoader": DTGLIGENLoader,
    "DTCheckpointLoader": DTCheckpointLoader,
    "DTDiffusersLoader": DTDiffusersLoader,
    "DTLoadLatent": DTLoadLatent,
    "DTLoadImage": DTLoadImage,
    "DTLoadImageMask": DTLoadImageMask,
    "DTUpscaleModelLoader": DTUpscaleModelLoader,
    "DTLorasLoader": DTLorasLoader
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "DTCheckpointLoader": "Load Checkpoint (With Config - Online)",
    "DTCheckpointLoaderSimple": "Load Checkpoint (Online)",
    "DTDiffusersLoader": "Load Diffusers (Online)",
    "DTVAELoader": "Load VAE (Online)",
    "DTLoraLoader": "Load LoRA (Online)",
    "DTCLIPLoader": "Load CLIP (Online)",
    "DTControlNetLoader": "Load ControlNet Model (Online)",
    "DTDiffControlNetLoader": "Load ControlNet Model (diff) (Online)",
    "DTStyleModelLoader": "Load Style Model (Online)",
    "DTCLIPVisionLoader": "Load CLIP Vision (Online)",
    "DTUpscaleModelLoader": "Load Upscale Model (Online)",
    "DTPreviewImage": "Preview Image (Online)",
    "DTLoadImage": "Load Image from URL",
    "DTLoadImageMask": "Load Image Mask from URL",
    "DTLorasLoader": "Load Multiple LoRAs by name (Online)",
}
