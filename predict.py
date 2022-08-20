from einops import repeat
import sys
import tempfile
import warnings
from typing import List, Optional

import numpy as np
import torch
from cog import BaseModel, BasePredictor, Input, Path
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image as PILImage

from ldm.knn_util import Perceptor, load_image_index_from_disk, sample_and_decode
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import load_model_from_config, set_seed, slugify

sys.path.append("src/taming-transformers")
warnings.filterwarnings("ignore")

DEBUG = True
DATABASE_NAMES = [
    "cars",
    "coco",
    "country211",
    "emotes",
    "faces",
    "food",
    "laion-aesthetic",
    "openimages",
    "pokemon",
    "prompt-engineer",
    "simulacra",
]
INIT_DATABASES = [
    "simulacra",
    "laion-aesthetic",
]  # start on cold boot, others will be loaded on first request
PROMPT_UPPER_BOUND = 8


class CaptionedImage(BaseModel):
    image: Path
    caption: str


def create_unconditional_embed(
    model_context: torch.Tensor,
    negative_clip_embed: Optional[torch.Tensor] = None,
) -> torch.Tensor:  # (batch_size, num_results, 768)
    """
    Create an unconditional embedding for the given model context.

    Args:
        model_context: The model context to use.
        device: The device to use.
        negative_clip_embed: A CLIP text embedding to use as the negative conditioning. Otherwise uses zeros.
    """
    if (
        negative_clip_embed
    ):  # if a negative is provided, then we use the negative as the unconditional embed
        return negative_clip_embed.expand_as(model_context)
    else:
        return torch.zeros_like(
            model_context
        )  # by default, the unconditional embed is all zeros


class Predictor(BasePredictor):
    @torch.inference_mode()
    def setup(self):
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        config = OmegaConf.load(f"configs/retrieval-augmented-diffusion/768x768.yaml")
        model: torch.nn.Module = load_model_from_config(config, f"models/rdm/rdm768x768/model.ckpt")  # type: ignore
        self.rdm_model = model.to(self.device)
        print("Loaded 1.4M param Retrieval Augmented Diffusion model.")

        self.clip_perceptor = Perceptor(self.device)

        self.image_indices = {
            database_name: load_image_index_from_disk(database_name)
            for database_name in INIT_DATABASES
        }
        print(f"Loaded searchers for {INIT_DATABASES}")

        self.output_directory = Path(tempfile.mkdtemp())

    def faiss_lookup(
        self,
        query_embedding: torch.Tensor,
        database_name: str,
        num_database_results: int,
    ) -> torch.Tensor:
        """
        Lookup the nearest neighbors in the given database.
        """
        current_image_index = self.image_indices[database_name]
        if query_embedding.ndim == 3:  # (b, 1, d)
            query_embedding = query_embedding.squeeze(
                1
            )  # need to reduce to (b, d) for faiss
        query_embeddings = query_embedding.cpu().detach().numpy().astype(np.float32)
        _, _, result_embeddings = current_image_index.search_and_reconstruct(
            query_embeddings, num_database_results
        )
        result_embeddings = torch.from_numpy(result_embeddings).to(self.device)
        print(
            f"Ran query w/ shape: {query_embedding.shape} in database: {database_name}"
        )
        return result_embeddings

    @torch.inference_mode()
    def predict(
        self,
        prompts: str = Input(
            default=None,
            description="(batched) Use up to 8 prompts by separating with a `|` character.",
        ),
        image_prompt: Path = Input(
            default=None,
            description="(overrides `prompts`) Use an image as the prompt to generate variations of an existing image.",
        ),
        number_of_variations: int = Input(
            default=4,
            description="Number of variations to generate when using only one `text_prompt`, or an `image_prompt`.",
            ge=1,
            le=8,
        ),
        database_name: str = Input(
            default="laion-aesthetic",
            description="Which database to use for the semantic search. Different databases have different capabilities.",
            choices=[  # TODO you have to copy this to the predict arg any time it is changed.
                "laion-aesthetic",
                "simulacra",
                "pokemon",
                "prompt-engineer",
                "emotes",
                "cars",
                "coco",
                "openimages",
                "country211",
                "faces",
                "food",
            ],
        ),
        use_database: bool = Input(
            default=True,
            description="Whether to use the database for the semantic search.",
        ),
        scale: float = Input(
            default=5.0,
            description="Classifier-free unconditional scale for the sampler.",
        ),
        num_database_results: int = Input(
            default=10,
            description="The number of search results from the retrieval backend to guide the generation with.",
            ge=1,
            le=20,
        ),
        height: int = Input(
            default=768,
            description="Desired width of generated images. Values beside 768 are likely to cause zooming issues.",
        ),
        width: int = Input(
            default=768,
            description="Desired width of generated images. Values beside 768 are not supported, likely to cause artifacts.",
        ),
        steps: int = Input(
            default=50,
            description="How many steps to run the model for. Using more will make generation take longer. 50 tends to work well.",
        ),
        ddim_sampling: bool = Input(
            default=False,
            description="Use ddim sampling instead of the faster plms sampling.",
        ),
        ddim_eta: float = Input(
            default=0.0,
            description="The eta parameter for ddim sampling.",
        ),
        negative_prompt: str = Input(
            default=None,
            description="(experimental) Use this prompt as a negative prompt for the sampler.",
        ),
        seed: int = Input(
            default=-1,
            description="Seed for the random number generator. Set to -1 to use a random seed.",
        ),
    ) -> List[CaptionedImage]:
        set_seed(seed)
        print(f"Seed: {seed}")

        if database_name not in self.image_indices:
            print(f"Loading database: {database_name}. May take a while...")
            self.image_indices[database_name] = load_image_index_from_disk(
                database_name
            )

        if image_prompt is not None:
            clip_image_embed = self.clip_perceptor.encode_image(
                image_prompt, normalize=True
            )
            clip_image_embed = repeat(
                clip_image_embed, "1 k d -> b k d", b=number_of_variations
            )
            model_context = clip_image_embed
            print(
                f"Using image as context: {image_prompt}, overriding any text prompts."
            )
        elif prompts is not None and len(prompts.strip()) > 0:
            clip_text_embed = self.clip_perceptor.encode_prompts(
                prompts, normalize=True
            )
            if clip_text_embed.shape[0] == 1:
                print(f"Only one prompt, repeating it {number_of_variations} times.")
                clip_text_embed = repeat(
                    clip_text_embed, "1 k d -> b k d", b=number_of_variations
                )

            if use_database:
                result_embeddings = self.faiss_lookup(
                    clip_text_embed, database_name, num_database_results
                )
                model_context = torch.cat(
                    [
                        clip_text_embed.to(self.device),
                        result_embeddings.to(self.device),
                    ],
                    dim=1,
                )
                print(
                    f"Using text as context: {prompts} (and {num_database_results} results from {database_name})"
                )
            else:
                model_context = clip_text_embed
                print(f"Using text as context: {prompts}. Database not being used.")

        assert model_context is not None, "Must provide either prompts or image_prompt"

        if (
            negative_prompt is not None
        ):  # if a negative is provided, then we use the negative as the unconditional embed
            negative_clip_embed = self.clip_perceptor.encode_prompts(
                negative_prompt, normalize=True
            )
            negative_clip_embed = negative_clip_embed.expand_as(model_context)
            print(f"(caution, experimental): Using negative prompt: {negative_prompt}")
        else:
            negative_clip_embed = torch.zeros_like(
                model_context
            )  # by default, the unconditional embed is all zeros

        with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
            decoded_generations = sample_and_decode(
                rdm_model=self.rdm_model,
                sampler=DDIMSampler(self.rdm_model)
                if ddim_sampling
                else PLMSSampler(self.rdm_model),
                model_context=model_context,
                steps=steps,
                scale=scale,
                ddim_eta=ddim_eta,
                unconditional_conditioning=negative_clip_embed,
                height=height,
                width=width,
            )

        def save_sample(generation, target_path):
            generation = 255.0 * rearrange(generation.cpu().numpy(), "c h w -> h w c")
            pil_image = PILImage.fromarray(generation.astype(np.uint8))
            pil_image.save(target_path, "png")

        prediction_output_paths = []
        if (
            len(prompts) == decoded_generations.shape[0]
        ):  # prompts are paired with the generated images
            labeled_generations = zip(decoded_generations, prompts)
        else:  # prompts are not paired with the generated images, use blank strings for the prompts
            labeled_generations = zip(
                decoded_generations, [""] * decoded_generations.shape[0]
            )

        for idx, (generation, prompt) in enumerate(labeled_generations):
            generation_stub = (
                f"sample_{idx:03d}__{slugify(prompt)}"
                if len(prompt) > 0
                else f"sample_{idx:03d}"
            )
            target_path = self.output_directory / f"{generation_stub}.png"
            save_sample(generation, target_path)
            if DEBUG:
                save_sample(generation, f"{generation_stub}_debug.png")
            prediction_output_paths.append(
                CaptionedImage(caption=prompt, image=target_path)
            )
        return prediction_output_paths
