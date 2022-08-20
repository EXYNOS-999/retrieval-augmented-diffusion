from pathlib import Path
from typing import List, Union

import clip
import torch
import torchvision.transforms.functional as tvf
from clip_retrieval.clip_back import load_index
from PIL import Image
from torchvision.transforms import InterpolationMode

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def sample_and_decode(
    rdm_model: torch.nn.Module,
    sampler: Union[DDIMSampler, PLMSSampler],
    model_context: torch.Tensor,
    unconditional_conditioning: torch.Tensor,
    steps: int = 50,
    scale: float = 5.0,
    height: int = 768,
    width: int = 768,
    ddim_eta: float = 0.0,
) -> torch.Tensor:
    with rdm_model.ema_scope():  # type: ignore
        shape = [
            16,
            height // 16,
            width // 16,
        ]  # note: currently hardcoded for f16 model
        samples, _ = sampler.sample(
            S=steps,
            conditioning=model_context,
            batch_size=model_context.shape[0],
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=unconditional_conditioning,
            eta=ddim_eta,
        )
        decoded_generations = rdm_model.decode_first_stage(samples)  # type: ignore
        return torch.clamp((decoded_generations + 1.0) / 2.0, min=0.0, max=1.0)


def load_image_index_from_disk(database_name: str):
    image_index_path = Path(
        "data", "rdm", "faiss_indices", database_name, "image.index"
    )
    assert image_index_path.exists(), f"database at {image_index_path} does not exist"
    print(f"Loading semantic index from {image_index_path}")
    return load_index(str(image_index_path), enable_faiss_memory_mapping=True)


def clip_image_preprocess(image: torch.Tensor, clip_size: int = 224) -> torch.Tensor:
    image = tvf.resize(
        image, [clip_size, clip_size], interpolation=InterpolationMode.BICUBIC
    )
    image = tvf.to_tensor(image)
    image = (image + 1.0) / 2.0  # normalize to [0, 1]
    image = tvf.normalize(
        image, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
    )  # normalize to CLIP format
    return image


class Perceptor:
    def __init__(
        self,
        device,
        clip_name: str = "ViT-L/14",
    ):
        print(f"Loading CLIP model on {device}")
        self.device = device
        clip_model, clip_preprocess = clip.load(clip_name, device=device)
        clip_model.eval()
        clip_model.to(device)
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_size = clip_model.visual.input_resolution

    @torch.no_grad()
    def encode_prompts(
        self, prompts: Union[str, List[str]], normalize: bool = True
    ) -> torch.Tensor:
        if isinstance(prompts, str):
            # either a single prompt or a list of prompts separated by |
            prompts = prompts.split("|") if "|" in prompts else [prompts]
            prompts = [prompt.strip() for prompt in prompts]
        text_tokens = clip.tokenize(prompts).to(self.device)
        encoded_text = self.clip_model.encode_text(text_tokens)
        if normalize:
            encoded_text = encoded_text / torch.linalg.norm(
                encoded_text, dim=1, keepdim=True
            )
        if encoded_text.dim() == 2:
            encoded_text = encoded_text[:, None, :]
        return encoded_text

    @torch.no_grad()
    def encode_image(self, image_path: str, normalize: bool = True) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image = (
            clip_image_preprocess(image, clip_size=self.clip_size)
            .unsqueeze(0)
            .to(self.device)
        )
        image_features = self.clip_model.encode_image(image)
        if normalize:
            image_features /= image_features.norm(dim=1, keepdim=True)
        if image_features.ndim == 2:
            image_features = image_features[:, None, :]
        return image_features
