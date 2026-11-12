import json, urllib.request, time, sys

SERVER = "http://localhost:8188"
CLIENT_ID = "orbit_wars_wan_job"

# Wan 2.2 TI2V 5B workflow — from Kijai's official example wanvideo_2_2_5B_I2V_example_WIP.json
# Key differences from 2.1 I2V: no CLIP vision encoder, uses WanVideoEncode+WanVideoEmptyEmbeds,
# Wan 2.2 VAE, flowmatch_pusa scheduler, EasyCache (not TeaCache)
workflow = {
    "1": {"class_type": "LoadWanVideoT5TextEncoder", "inputs": {
        "model_name": "umt5-xxl-enc-bf16.safetensors",
        "precision": "bf16",
        "load_device": "offload_device",
        "quantization": "disabled"
    }},
    "2": {"class_type": "LoadImage", "inputs": {"image": "sana.jpg", "upload": "image"}},
    "3": {"class_type": "ImageScale", "inputs": {
        "image": ["2", 0],
        "upscale_method": "lanczos",
        "width": 832,
        "height": 480,
        "crop": "center"
    }},
    "4": {"class_type": "WanVideoVAELoader", "inputs": {
        "model_name": "wan2.2_vae.safetensors",
        "precision": "bf16"
    }},
    "5": {"class_type": "WanVideoBlockSwap", "inputs": {
        "blocks_to_swap": 10,
        "offload_img_emb": False,
        "offload_txt_emb": False,
        "vace_blocks_to_swap": 0
    }},
    "6": {"class_type": "WanVideoModelLoader", "inputs": {
        "model": "wan2.2_ti2v_5B_fp16.safetensors",
        "base_precision": "fp16_fast",
        "quantization": "disabled",
        "load_device": "offload_device",
        "attention_mode": "sageattn",
        "block_swap_args": ["5", 0]
    }},
    "7": {"class_type": "WanVideoTextEncode", "inputs": {
        "positive_prompt": "A bearded man sitting still on a sandstone rock at a cliff-top viewpoint, brown bush hat on his head, hands resting at his sides. He slowly raises his right hand up toward his head, grips the brim of the hat, and smoothly lifts it off his head in one clean motion, revealing short dark hair. He holds the hat in his hand. Cinematic, photorealistic, smooth natural motion.",
        "negative_prompt": "static, blurry, low quality, worst quality, distorted, ugly, morphing, flickering",
        "force_offload": True,
        "use_disk_cache": False,
        "device": "gpu",
        "t5": ["1", 0],
        "model_to_offload": ["6", 0]
    }},
    "8": {"class_type": "WanVideoEncode", "inputs": {
        "vae": ["4", 0],
        "image": ["3", 0],
        "enable_vae_tiling": False,
        "tile_x": 272,
        "tile_y": 272,
        "tile_stride_x": 144,
        "tile_stride_y": 128,
        "start_idx": 0,
        "end_idx": 1
    }},
    "9": {"class_type": "WanVideoEmptyEmbeds", "inputs": {
        "width": 832,
        "height": 480,
        "num_frames": 121,
        "extra_latents": ["8", 0]
    }},
    "10": {"class_type": "WanVideoEasyCache", "inputs": {
        "easycache_thresh": 0.015,
        "start_step": 10,
        "end_step": -1,
        "cache_device": "offload_device"
    }},
    "11": {"class_type": "WanVideoSampler", "inputs": {
        "model": ["6", 0],
        "image_embeds": ["9", 0],
        "text_embeds": ["7", 0],
        "steps": 30,
        "cfg": 5.0,
        "shift": 8.0,
        "seed": 42,
        "force_offload": True,
        "scheduler": "flowmatch_pusa",
        "riflex_freq_index": 0,
        "denoise_strength": 1.0,
        "cache_args": ["10", 0]
    }},
    "12": {"class_type": "WanVideoDecode", "inputs": {
        "enable_vae_tiling": False,
        "tile_x": 272,
        "tile_y": 272,
        "tile_stride_x": 144,
        "tile_stride_y": 128,
        "vae": ["4", 0],
        "samples": ["11", 0]
    }},
    "13": {"class_type": "VHS_VideoCombine", "inputs": {
        "frame_rate": 24,
        "loop_count": 0,
        "filename_prefix": "wan22_5b_i2v",
        "format": "video/h264-mp4",
        "pix_fmt": "yuv420p",
        "crf": 19,
        "save_metadata": True,
        "trim_to_audio": False,
        "pingpong": False,
        "save_output": True,
        "images": ["12", 0]
    }}
}

def wait_for_comfy(timeout=120):
    for _ in range(timeout):
        try:
            r = urllib.request.urlopen(f"{SERVER}/system_stats", timeout=2)
            if r.status == 200:
                print("ComfyUI ready"); return True
        except Exception:
            pass
        time.sleep(1)
    return False

def submit(wf):
    data = json.dumps({"prompt": wf, "client_id": CLIENT_ID}).encode()
    req = urllib.request.Request(f"{SERVER}/prompt", data=data, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read())
    print(f"Queued: {result}")
    return result.get("prompt_id")

if not wait_for_comfy():
    print("ComfyUI not ready after 120s"); sys.exit(1)

pid = submit(workflow)
print(f"Job submitted: {pid}")
