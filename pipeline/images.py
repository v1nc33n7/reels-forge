from pathlib import Path
from openai import OpenAI
from openai.types.images_response import ImagesResponse
from pipeline.pipeline import PipelineContext, PipelineStage
import base64


class ImagesStage(PipelineStage):
    def __init__(self, llm: OpenAI) -> None:
        self.llm = llm

    @property
    def name(self) -> str:
        return "images"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.visuals:
            raise ValueError("Visuals cannot be None")

        if not ctx.images:
            ctx.images = []
        self._dir(ctx).mkdir(parents=True, exist_ok=True)

        index = 0
        ctx.images.append(self._save_img(ctx, index, ctx.visuals.hook_image_prompt))
        index += 1
        for s in ctx.visuals.scenes:
            ctx.images.append(self._save_img(ctx, index, s.image_prompt))
            index += 1
        ctx.images.append(self._save_img(ctx, index, ctx.visuals.cta_image_prompt))

        return ctx

    def load_from_disk(self, ctx: PipelineContext) -> PipelineContext:
        ctx.images = []

        directory = self._dir(ctx)
        if not directory.exists():
            raise FileNotFoundError(f"Image directory does not exist: {directory}")

        for p in directory.iterdir():
            ctx.images.append(p)

        ctx.images.sort()
        return ctx

    def _save_img(self, ctx: PipelineContext, index: int, prompt: str) -> Path:
        response = self.llm.images.generate(
            model="gpt-image-1-mini",
            quality="low",
            prompt=prompt,
            size="1024x1536",
        )

        path = self._dir(ctx) / f"{index}.png"
        with open(path, "wb") as f:
            f.write(self._decode_image(response))
        return path

    def _decode_image(self, response: ImagesResponse) -> bytes:
        data = self._require(
            response.data, "image response 'data' field is missing or empty"
        )
        decoded = self._require(
            data[0].b64_json,
            "image response does not contain base64-encoded image data",
        )

        try:
            return base64.b64decode(decoded)
        except Exception as e:
            raise ValueError("Failed to decode base64 image data from response") from e
