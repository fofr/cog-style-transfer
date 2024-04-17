# Style transfer

Run this model on Replicate:

https://replicate.com/fofr/style-transfer

Or run it in ComfyUI:

https://github.com/fofr/cog-style-transfer/blob/main/style-transfer-ui.json

You’ll need the IPAdapter Plus custom nodes:

- [ComfyUI IPAdapter Plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus/tree/1ac1cae)

## Developing locally

Clone this repository:

```sh
git clone --recurse-submodules https://github.com/fofr/cog-style-transfer.git && cd cog-style-transfer/ComfyUI
```

Create python venv and activate

```sh
python3 -m venv . && source bin/activate
```

Install the required dependencies

```sh
pip install -r requirements.txt
```

Download dreamshaperXL_lightningDPMSDE.safetensors to models/checkpoints

```sh
wget https://huggingface.co/gingerlollipopdx/ModelsXL/resolve/main/dreamshaperXL_lightningDPMSDE.safetensors?download=true -O models/checkpoints/dreamshaperXL_lightningDPMSDE.safetensors
```

Download CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors to models/clip_vision

```sh
wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors?download=true -O models/checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors
```

Download ip-adapter-plus_sdxl_vit-h.safetensors to models/ipadapter

```sh
wget https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors?download=true -O models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors
```

Run the [following script](https://github.com/fofr/cog-style-transfer/blob/main/scripts/clone_plugins.sh) to install all the custom nodes:

```sh
./scripts/clone_plugins.sh
```

Finally, install it, run it and enjoy it!

```sh
python3 main.py
```

### Running the Web UI from your Cog container

1. **GPU Machine**: Start the Cog container and expose port 8188:
```sh
sudo cog run -p 8188 bash
```
Running this command starts up the Cog container and let's you access it

2. **Inside Cog Container**: Now that we have access to the Cog container, we start the server, binding to all network interfaces:
```sh
cd ComfyUI/
python main.py --listen 0.0.0.0
```

3. **Local Machine**: Access the server using the GPU machine's IP and the exposed port (8188):
`http://<gpu-machines-ip>:8188`

When you goto `http://<gpu-machines-ip>:8188` you'll see the classic ComfyUI web form!
