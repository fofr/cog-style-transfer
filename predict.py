import os
import shutil
import json
import mimetypes
import random
from PIL import Image
from typing import List
from cog import BasePredictor, Input, Path
from helpers.comfyui import ComfyUI

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"

mimetypes.add_type("image/webp", ".webp")

with open("style-transfer-api.json", "r") as file:
    STYLE_TRANSFER_WORKFLOW_JSON = file.read()

with open("style-transfer-with-structure-api.json", "r") as file:
    STYLE_TRANSFER_WITH_STRUCTURE_WORKFLOW_JSON = file.read()


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)
        self.comfyUI.load_workflow(
            STYLE_TRANSFER_WORKFLOW_JSON, handle_inputs=False, handle_weights=True
        )

    def cleanup(self):
        self.comfyUI.clear_queue()
        for directory in [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

    def handle_input_file(self, input_file: Path, filename: str = "image.png"):
        image = Image.open(input_file)
        image.save(os.path.join(INPUT_DIR, filename))

    def log_and_collect_files(self, directory, prefix=""):
        files = []
        for f in os.listdir(directory):
            if f == "__MACOSX":
                continue
            path = os.path.join(directory, f)
            if os.path.isfile(path):
                print(f"{prefix}{f}")
                files.append(Path(path))
            elif os.path.isdir(path):
                print(f"{prefix}{f}/")
                files.extend(self.log_and_collect_files(path, prefix=f"{prefix}{f}/"))
        return files

    def set_weights(self, workflow, model: str):
        loader = workflow["2"]["inputs"]
        sampler = workflow["3"]["inputs"]

        if model == "fast":
            sampler["steps"] = 4
            sampler["cfg"] = 2
            sampler["sampler_name"] = "dpmpp_sde_gpu"
        else:
            sampler["steps"] = 20
            sampler["cfg"] = 8
            sampler["sampler_name"] = "dpmpp_2m_sde_gpu"

        if model == "fast":
            loader["ckpt_name"] = "dreamshaperXL_lightningDPMSDE.safetensors"
        elif model == "high-quality":
            loader["ckpt_name"] = "albedobaseXL_v21.safetensors"
        elif model == "realistic":
            loader["ckpt_name"] = "RealVisXL_V4.0.safetensors"
        elif model == "cinematic":
            loader["ckpt_name"] = "CinematicRedmond.safetensors"
        elif model == "animated":
            loader["ckpt_name"] = "starlightXLAnimated_v3.safetensors"

    def update_workflow(self, workflow, **kwargs):
        self.set_weights(workflow, kwargs["model"])
        workflow["6"]["inputs"]["text"] = kwargs["prompt"]
        workflow["7"]["inputs"]["text"] = f"nsfw, nude, {kwargs['negative_prompt']}"

        sampler = workflow["3"]["inputs"]
        sampler["seed"] = kwargs["seed"]

        if kwargs["is_structure"]:
            sampler["denoise"] = kwargs["structure_denoising_strength"]
            workflow["18"]["inputs"]["strength"] = kwargs[
                "structure_denoising_strength"
            ]
            workflow["24"]["inputs"]["amount"] = kwargs["batch_size"]
        else:
            empty_latent_image = workflow["10"]["inputs"]
            empty_latent_image["width"] = kwargs["width"]
            empty_latent_image["height"] = kwargs["height"]
            empty_latent_image["batch_size"] = kwargs["batch_size"]

    def predict(
        self,
        style_image: Path = Input(
            description="Copy the style from this image",
        ),
        structure_image: Path = Input(
            description="An optional image to copy structure from. Output images will use the same aspect ratio.",
            default=None,
        ),
        prompt: str = Input(
            description="Prompt for the image",
            default="An astronaut riding a unicorn",
        ),
        negative_prompt: str = Input(
            description="Things you do not want to see in your image",
            default="",
        ),
        width: int = Input(
            description="Width of the output image (ignored if structure image given)",
            default=1024,
        ),
        height: int = Input(
            description="Height of the output image (ignored if structure image given)",
            default=1024,
        ),
        model: str = Input(
            description="Model to use for the generation",
            choices=["fast", "high-quality", "realistic", "cinematic", "animated"],
            default="fast",
        ),
        number_of_images: int = Input(
            description="Number of images to generate", default=1, ge=1, le=10
        ),
        structure_depth_strength: float = Input(
            description="Strength of the depth controlnet",
            default=1.0,
            ge=0,
            le=2.0,
        ),
        structure_denoising_strength: float = Input(
            description="How much of the original image (and colors) to preserve (0 is all, 1 is none, 0.65 is a good balance)",
            default=0.65,
            ge=0,
            le=1,
        ),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality of the output images, from 0 to 100. 100 is best quality, 0 is lowest quality.",
            default=80,
            ge=0,
            le=100,
        ),
        seed: int = Input(
            description="Set a seed for reproducibility. Random by default.",
            default=None,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.cleanup()

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            print(f"Random seed set to: {seed}")

        if not style_image:
            raise ValueError("Style image is required")

        self.handle_input_file(style_image)

        if structure_image:
            self.handle_input_file(structure_image, "structure.png")
            workflow = json.loads(STYLE_TRANSFER_WITH_STRUCTURE_WORKFLOW_JSON)
        else:
            workflow = json.loads(STYLE_TRANSFER_WORKFLOW_JSON)

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            batch_size=number_of_images,
            model=model,
            is_structure=bool(structure_image),
            structure_depth_strength=structure_depth_strength,
            structure_denoising_strength=structure_denoising_strength,
        )

        wf = self.comfyUI.load_workflow(workflow, handle_weights=True)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)
        files = self.log_and_collect_files(OUTPUT_DIR)

        if output_quality < 100 or output_format in ["webp", "jpg"]:
            optimised_files = []
            for file in files:
                if file.is_file() and file.suffix in [".jpg", ".jpeg", ".png"]:
                    image = Image.open(file)
                    optimised_file_path = file.with_suffix(f".{output_format}")
                    image.save(
                        optimised_file_path,
                        quality=output_quality,
                        optimize=True,
                    )
                    optimised_files.append(optimised_file_path)
                else:
                    optimised_files.append(file)

            files = optimised_files

        return files
