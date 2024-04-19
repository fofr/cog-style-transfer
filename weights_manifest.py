import json
import os

BASE_URL = "https://weights.replicate.delivery/default/comfy-ui"
BASE_PATH = "ComfyUI/models"

from helpers.ComfyUI_Controlnet_Aux import ComfyUI_Controlnet_Aux

class WeightsManifest:
    def __init__(self):
        self.weights_manifest = self._load_weights_manifest()
        self.weights_map = self._initialize_weights_map()

    def _load_weights_manifest(self):
        return self._load_local_manifest()

    def _load_local_manifest(self):
        WEIGHTS_MANIFEST_PATH = "weights.json"
        if os.path.exists(WEIGHTS_MANIFEST_PATH):
            with open(WEIGHTS_MANIFEST_PATH, "r") as f:
                return json.load(f)
        else:
            print("Local weights manifest file does not exist.")
            return {}

    def _generate_weights_map(self, keys, dest):
        return {
            key: {
                "url": f"{BASE_URL}/{dest}/{key}.tar",
                "dest": f"{BASE_PATH}/{dest}",
            }
            for key in keys
        }

    def _initialize_weights_map(self):
        weights_map = {}
        for key in self.weights_manifest.keys():
            if key.isupper():
                weights_map.update(
                    self._generate_weights_map(self.weights_manifest[key], key.lower())
                )

        weights_map.update(ComfyUI_Controlnet_Aux.weights_map(BASE_URL))
        return weights_map
