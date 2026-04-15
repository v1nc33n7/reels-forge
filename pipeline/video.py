from pathlib import Path
from typing import List

import subprocess
from pipeline.pipeline import PipelineStage, PipelineContext


class VideoStage(PipelineStage):
    @property
    def name(self) -> str:
        return "video"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.images:
            raise ValueError("Images cannot be None")

        if not ctx.voices:
            raise ValueError("Voices cannot be None")

        if not ctx.voice_durations:
            raise ValueError("Voice Durations cannot be None")

        if not ctx.subtitles:
            raise ValueError("Subtitles cannot be None")

        Path(self._dir(ctx)).mkdir(parents=True, exist_ok=True)

        raw_path = self._dir(ctx) / "raw.mp4"
        cmd = self._merge_images_and_voices_cmd(
            images=ctx.images,
            voices=ctx.voices,
            voice_durations=ctx.voice_durations,
            write_path=raw_path,
        )
        self._run_cmd(cmd)

        final_path = self._dir(ctx) / "final.mp4"
        cmd = self._add_subtitles_cmd(
            video=raw_path,
            subtitles=ctx.subtitles,
            write_path=final_path,
        )
        self._run_cmd(cmd)

        ctx.video = final_path
        return ctx

    def load_from_disk(self, ctx: PipelineContext) -> PipelineContext:
        path = Path(ctx.dir) / ctx.topic / "final.mp4"
        if not path.exists():
            raise FileNotFoundError(f"Video file does not exist: {path}")

        ctx.video = path
        return ctx

    def _run_cmd(self, cmd: str) -> None:
        subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _merge_images_and_voices_cmd(
        self,
        images: List[Path],
        voices: List[Path],
        voice_durations: List[float],
        write_path: Path,
    ) -> str:
        cmd = "ffmpeg -y "
        for image, voice, duration in zip(images, voices, voice_durations):
            cmd += f'-loop 1 -t {duration} -i "{image}" -i "{voice}" '
        cmd += '-filter_complex "'
        index = 0
        for _ in images:
            cmd += f"[{index}:v][{index + 1}:a]"
            index += 2
        cmd += f' concat=n={len(images)}:v=1:a=1 [v][a]" '
        cmd += f'-map "[v]" -map "[a]" -c:v libx264 -tune stillimage -c:a aac -pix_fmt yuv420p "{write_path}"'

        return cmd

    def _add_subtitles_cmd(self, video: Path, subtitles: Path, write_path: Path) -> str:
        style = (
            "Fontname=Arial Black,"
            "Fontsize=32,"
            "PrimaryColour=&HFFFFFF&,"
            "OutlineColour=&H000000&,"
            "BackColour=&H80000000,"
            "Bold=1,"
            "Outline=3,"
            "Shadow=1,"
            "MarginV=70"
        )

        cmd = "ffmpeg -y "
        cmd += f"-i \"{video}\" -vf \"fps=30,subtitles='{subtitles}':force_style='{style}'\" "
        cmd += f'-c:a copy "{write_path}"'
        return cmd
