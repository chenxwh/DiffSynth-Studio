# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
import torch
from cog import BasePredictor, Input, Path, BaseModel

from diffsynth import (
    save_video,
    ModelManager,
    SVDVideoPipeline,
    HunyuanDiTImagePipeline,
)


os.environ["TOKENIZERS_PARALLELISM"] = "True"

# Download the models following the instruction in https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ExVideo/ExVideo_svd_test.py
# Then push and load from replicate.delivery for faster booting
MODEL_URL = (
    "https://weights.replicate.delivery/default/modelscope/DiffSynth-Studio/ExVideo.tar"
)
MODEL_CACHE = "demo_model_cache"


class ModelOutput(BaseModel):
    image: Path
    video: Path
    upscale_video: Path


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        image_model_manager = ModelManager(
            torch_dtype=torch.float16,
            device="cuda",
            file_path_list=[
                f"{MODEL_CACHE}/HunyuanDiT/t2i/clip_text_encoder/pytorch_model.bin",
                f"{MODEL_CACHE}/HunyuanDiT/t2i/mt5/pytorch_model.bin",
                f"{MODEL_CACHE}/HunyuanDiT/t2i/model/pytorch_model_ema.pt",
                f"{MODEL_CACHE}/HunyuanDiT/t2i/sdxl-vae-fp16-fix/diffusion_pytorch_model.bin",
            ],
        )
        self.image_pipe = HunyuanDiTImagePipeline.from_model_manager(
            image_model_manager
        )

        video_model_manager = ModelManager(
            torch_dtype=torch.float16,
            device="cuda",
            file_path_list=[
                f"{MODEL_CACHE}/stable_video_diffusion/svd_xt.safetensors",
                f"{MODEL_CACHE}/stable_video_diffusion/model.fp16.safetensors",
            ],
        )
        self.video_pipe = SVDVideoPipeline.from_model_manager(video_model_manager)

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="bonfire, on the stone",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="错误的眼睛，糟糕的人脸，毁容，糟糕的艺术，变形，多余的肢体，模糊的颜色，模糊，重复，病态，残缺，",
        ),
        num_frames: int = Input(
            description="Number of the output frames", default=128, le=128
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps for image and video generation",
            ge=1,
            le=500,
            default=25,
        ),
        num_inference_steps_upscale_video: int = Input(
            description="Number of denoising steps for upscaling the video",
            ge=1,
            le=500,
            default=25,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        image = self.image_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            height=1024,
            width=1024,
        )
        # Now, generate a video with resolution of 512. 20GB VRAM is required.
        video = self.video_pipe(
            input_image=image.resize((512, 512)),
            num_frames=num_frames,
            fps=30,
            height=512,
            width=512,
            motion_bucket_id=127,
            num_inference_steps=num_inference_steps,
            min_cfg_scale=2,
            max_cfg_scale=2,
            contrast_enhance_scale=1.2,
        )

        # Upscale the video. 52GB VRAM is required.
        upscale_video = self.video_pipe(
            input_image=image.resize((1024, 1024)),
            input_video=[frame.resize((1024, 1024)) for frame in video],
            denoising_strength=0.5,
            num_frames=num_frames,
            fps=30,
            height=1024,
            width=1024,
            motion_bucket_id=127,
            num_inference_steps=num_inference_steps_upscale_video,
            min_cfg_scale=2,
            max_cfg_scale=2,
            contrast_enhance_scale=1.2,
        )

        out_image_path = "/tmp/image.png"
        image.save(out_image_path)

        out_video_path = "/tmp/video_512.mp4"
        save_video(video, out_video_path, fps=30)

        upscale_video_path = "/tmp/video_1024.mp4"
        save_video(upscale_video, upscale_video_path, fps=30)

        return ModelOutput(
            image=Path(out_image_path),
            video=Path(out_video_path),
            upscale_video=Path(upscale_video_path),
        )
