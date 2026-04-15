from pathlib import Path
from openai import OpenAI
from pipeline.pipeline import PipelineStage, PipelineContext


class VoiceStage(PipelineStage):
    def __init__(self, llm: OpenAI) -> None:
        self.llm = llm

    @property
    def name(self) -> str:
        return "voice"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.brief:
            raise ValueError("Brief cannot be None")

        if not ctx.voices:
            ctx.voices = []
        self._dir(ctx).mkdir(parents=True, exist_ok=True)

        index = 0
        ctx.voices.append(self._save_voice(ctx, index, ctx.brief.hook))
        index += 1
        for s in ctx.brief.scenes:
            ctx.voices.append(self._save_voice(ctx, index, s.narration))
            index += 1
        ctx.voices.append(self._save_voice(ctx, index, ctx.brief.cta))

        return ctx

    def load_from_disk(self, ctx: PipelineContext) -> PipelineContext:
        ctx.voices = []

        directory = self._dir(ctx)
        if not directory.exists():
            raise FileNotFoundError(f"Voice directory does not exist: {directory}")

        for p in directory.iterdir():
            ctx.voices.append(p)

        ctx.voices.sort()
        return ctx

    def _save_voice(self, ctx: PipelineContext, index: int, input: str) -> Path:
        response = self.llm.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=input.replace("|", ""),
        )
        path = self._dir(ctx) / f"{index}.mp3"
        response.write_to_file(path)
        return path
