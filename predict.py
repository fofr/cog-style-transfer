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
    WORKFLOW_JSON = file.read()


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)
        self.comfyUI.load_workflow(
            WORKFLOW_JSON, handle_inputs=False, handle_weights=True
        )

    def cleanup(self):
        self.comfyUI.clear_queue()
        for directory in [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

    def handle_input_file(self, input_file: Path):
        image = Image.open(input_file)
        image.save(os.path.join(INPUT_DIR, "image.png"))

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

    def update_workflow(self, workflow, **kwargs):
        workflow["6"]["inputs"]["text"] = kwargs["prompt"]
        workflow["7"]["inputs"]["text"] = kwargs["negative_prompt"]
        workflow["3"]["inputs"]["seed"] = kwargs["seed"]
        empty_latent_image = workflow["10"]["inputs"]
        empty_latent_image["width"] = kwargs["width"]
        empty_latent_image["height"] = kwargs["height"]
        empty_latent_image["batch_size"] = kwargs["batch_size"]

    def predict(
        self,
        style_image: Path = Input(
            description="Copy the style from this image",
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
            description="Width of the output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of the output image",
            default=1024,
        ),
        number_of_images: int = Input(
            description="Number of images to generate", default=1, ge=1, le=10
        ),
        optimise_output_images: bool = Input(
            description="Optimise output images by using webp",
            default=True,
        ),
        optimise_output_images_quality: int = Input(
            description="Quality of the output images, from 0 to 100",
            default=80,
        ),
        seed: int = Input(
            description="Seed for the random number generator",
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

        workflow = json.loads(WORKFLOW_JSON)
        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            batch_size=number_of_images,
        )

        wf = self.comfyUI.load_workflow(
            workflow, handle_inputs=True, handle_weights=False
        )

        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        files = self.log_and_collect_files(OUTPUT_DIR)

        if optimise_output_images:
            optimised_files = []
            for file in files:
                if file.is_file() and file.suffix in [".jpg", ".jpeg", ".png"]:
                    image = Image.open(file)
                    optimised_file_path = file.with_suffix(".webp")
                    image.save(
                        optimised_file_path,
                        quality=optimise_output_images_quality,
                        optimize=True,
                    )
                    optimised_files.append(optimised_file_path)
                else:
                    optimised_files.append(file)

            files = optimised_files

        return files
