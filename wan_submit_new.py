
import json, urllib.request, time, sys

SERVER = "http://localhost:8188"
CLIENT_ID = "orbit_wars_wan_job"

workflow = {
    "1": {"class_type": "LoadWanVideoT5TextEncoder", "inputs": {"model_name": "umt5-xxl-enc-bf16.safetensors", "precision": "bf16", "load_device": "offload_device", "quantization": "disabled"}},
    "2": {"class_type": "LoadImage", "inputs": {"image": "sana.jpg", "upload": "image"}},
    "3": {"class_type": "ImageScale", "inputs": {"image": ["2", 0], "upscale_method": "lanczos", "width": 832, "height": 480, "crop": "center"}},
    "4": {"class_type": "LoadWanVideoClipTextEncoder", "inputs": {"model_name": "open-clip-xlm-roberta-large-vit-huge-14_visual_fp16.safetensors", "precision": "fp16", "load_device": "offload_device"}},
    "5": {"class_type": "WanVideoClipVisionEncode", "inputs": {"clip_vision": ["4", 0], "image_1": ["3", 0], "strength_1": 1.0, "strength_2": 1.0, "crop": "center", "combine_embeds": "average", "force_offload": True}},
    "6": {"class_type": "WanVideoVAELoader", "inputs": {"model_name": "Wan2_1_VAE_bf16.safetensors", "precision": "bf16"}},
    "7": {"class_type": "WanVideoModelLoader", "inputs": {"model": "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors", "base_precision": "fp16", "quantization": "fp8_e4m3fn", "load_device": "offload_device", "attention_mode": "sageattn"}},
    "8": {"class_type": "WanVideoBlockSwap", "inputs": {"blocks_to_swap": 30, "offload_img_emb": False, "offload_txt_emb": False, "vace_blocks_to_swap": 0}},
    "9": {"class_type": "WanVideoSetBlockSwap", "inputs": {"model": ["7", 0], "block_swap_args": ["8", 0]}},
    "10": {"class_type": "WanVideoTextEncode", "inputs": {
        "positive_prompt": "A bearded man seated on a large pale sandstone rock at a dramatic cliff-top viewpoint in Saxon Switzerland. He slowly reaches up with his right hand, grips the brim of his wide brown Australian bush hat, and lifts it off his head, revealing short dark hair beneath. He sets the hat down beside him on the rock and glances out across the forested valley below. Cinematic, photorealistic, smooth natural motion.",
        "negative_prompt": "static, blurry, low quality, worst quality, distorted, ugly, morphing, flickering",
        "force_offload": True, "use_disk_cache": False, "device": "gpu", "t5": ["1", 0], "model_to_offload": ["7", 0]}},
    "11": {"class_type": "WanVideoImageToVideoEncode", "inputs": {"width": 832, "height": 480, "num_frames": 81, "noise_aug_strength": 0.03, "start_latent_strength": 1.0, "end_latent_strength": 1.0, "force_offload": True, "vae": ["6", 0], "clip_embeds": ["5", 0], "start_image": ["3", 0]}},
    "15": {"class_type": "WanVideoTeaCache", "inputs": {"rel_l1_thresh": 0.3, "start_step": 1, "end_step": -1, "cache_device": "offload_device", "use_coefficients": True}},
    "12": {"class_type": "WanVideoSampler", "inputs": {"model": ["9", 0], "image_embeds": ["11", 0], "text_embeds": ["10", 0], "steps": 20, "cfg": 1.0, "shift": 5.0, "seed": 42, "force_offload": True, "scheduler": "dpm++_sde", "riflex_freq_index": 0, "denoise_strength": 1.0, "cache_args": ["15", 0]}},
    "13": {"class_type": "WanVideoDecode", "inputs": {"enable_vae_tiling": True, "tile_x": 128, "tile_y": 128, "tile_stride_x": 64, "tile_stride_y": 64, "vae": ["6", 0], "samples": ["12", 0]}},
    "14": {"class_type": "VHS_VideoCombine", "inputs": {"frame_rate": 16, "loop_count": 0, "filename_prefix": "wan_sana_i2v", "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19, "save_metadata": True, "trim_to_audio": False, "pingpong": False, "save_output": True, "images": ["13", 0]}}
}

def wait_for_comfy(timeout=120):
    for _ in range(timeout):
        try:
            r = urllib.request.urlopen(f"{SERVER}/system_stats", timeout=2)
            if r.status == 200:
                print("ComfyUI ready"); return True
        except: pass
        time.sleep(1)
    return False

def submit(workflow):
    data = json.dumps({"prompt": workflow, "client_id": CLIENT_ID}).encode()
    req = urllib.request.Request(f"{SERVER}/prompt", data=data, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read())
    print(f"Queued: {result}")
    return result.get("prompt_id")

if not wait_for_comfy():
    print("ComfyUI not ready after 120s"); sys.exit(1)

pid = submit(workflow)
print(f"Job submitted: {pid}")
