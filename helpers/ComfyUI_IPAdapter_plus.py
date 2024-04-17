import os

# List of presets
PRESETS = [
    # IPAdapterUnifiedLoader
    "LIGHT - SD1.5 only (low strength)",
    "STANDARD (medium strength)",
    "VIT-G (medium strength)",
    "PLUS (high strength)",
    "PLUS FACE (portraits)",
    "FULL FACE - SD1.5 only (portraits stronger)",
    # IPAdapterUnifiedLoaderFaceID
    "FACEID",
    "FACEID PLUS - SD1.5 only",
    "FACEID PLUS V2",
    "FACEID PORTRAIT (style transfer)",

    # IPAdapterUnifiedLoaderCommunity
    "Composition",
]


class ComfyUI_IPAdapter_plus:
    @staticmethod
    def prepare():
        # create the ipadapter folder in ComfyUI/models/ipadapter
        # if it doesn't exist at setup time then the plugin defers to the base directory
        # and won't look for our ipadaters that are downloaded on demand
        if not os.path.exists("ComfyUI/models/ipadapter"):
            os.makedirs("ComfyUI/models/ipadapter")

    @staticmethod
    def get_preset_weights(preset):
        is_insightface = False
        weights_to_add = []

        # clipvision
        if preset.startswith("VIT-G"):
            weights_to_add.append("CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors")
        else:
            weights_to_add.append("CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors")

        # ipadapters
        if preset.startswith("LIGHT"):
            weights_to_add.append("ip-adapter_sd15_light_v11.bin")

        if preset.startswith("STANDARD"):
            weights_to_add.extend(
                ["ip-adapter_sd15.safetensors", "ip-adapter_sdxl_vit-h.safetensors"]
            )

        if preset.startswith("VIT-G"):
            weights_to_add.extend(
                ["ip-adapter_sd15_vit-G.safetensors", "ip-adapter_sdxl.safetensors"]
            )

        if preset.startswith("PLUS ("):
            weights_to_add.extend(
                [
                    "ip-adapter-plus_sd15.safetensors",
                    "ip-adapter-plus_sdxl_vit-h.safetensors",
                ]
            )

        if preset.startswith("PLUS FACE"):
            weights_to_add.extend(
                [
                    "ip-adapter-plus-face_sd15.safetensors",
                    "ip-adapter-plus-face_sdxl_vit-h.safetensors",
                ]
            )

        if preset.startswith("FULL FACE"):
            weights_to_add.append("ip-adapter-full-face_sd15.safetensors")

        if preset == "FACEID":
            is_insightface = True
            weights_to_add.extend(
                [
                    "ip-adapter-faceid_sd15.bin",
                    "ip-adapter-faceid_sdxl.bin",
                    "ip-adapter-faceid_sd15_lora.safetensors",
                    "ip-adapter-faceid_sdxl_lora.safetensors",
                ]
            )

        if preset.startswith("FACEID PORTRAIT"):
            is_insightface = True
            weights_to_add.extend(
                [
                    "ip-adapter-faceid-portrait-v11_sd15.bin",
                    "ip-adapter-faceid-portrait_sdxl.bin",
                ]
            )

        if preset.startswith("FACEID PLUS - "):
            is_insightface = True
            weights_to_add.extend(
                [
                    "ip-adapter-faceid-plus_sd15.bin",
                    "ip-adapter-faceid-plus_sd15_lora.safetensors",
                ]
            )

        if preset.startswith("FACEID PLUS V2"):
            is_insightface = True
            weights_to_add.extend(
                [
                    "ip-adapter-faceid-plusv2_sd15.bin",
                    "ip-adapter-faceid-plusv2_sdxl.bin",
                    "ip-adapter-faceid-plusv2_sd15_lora.safetensors",
                    "ip-adapter-faceid-plusv2_sdxl_lora.safetensors",
                ]
            )

        if preset.startswith("Composition"):
            weights_to_add.extend(
                [
                    "ip_plus_composition_sd15.safetensors",
                    "ip_plus_composition_sdxl.safetensors",
                ]
            )

        if is_insightface:
            weights_to_add.append("models/buffalo_l")

        return weights_to_add

    @staticmethod
    def add_weights(weights_to_download, node):
        if "class_type" in node and node["class_type"] in [
            "IPAdapterUnifiedLoader",
            "IPAdapterUnifiedLoaderFaceID",
            "IPAdapterUnifiedLoaderCommunity",
        ]:
            preset = node["inputs"]["preset"]
            print(f"Including weights for IPAdapter preset: {preset}")
            if preset:
                weights_to_download.extend(
                    ComfyUI_IPAdapter_plus.get_preset_weights(preset)
                )
        elif "class_type" in node and node["class_type"] in [
            "IPAdapterInsightFaceLoader",
        ]:
            weights_to_download.append("models/buffalo_l")
