from datetime import timedelta
from pathlib import Path
from pipeline.pipeline import PipelineStage, PipelineContext
from mutagen.mp3 import MP3


class SubtitlesStage(PipelineStage):
    @property
    def name(self) -> str:
        return "subtitles"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.voices:
            raise ValueError("Voice cannot be None")

        if not ctx.brief:
            raise ValueError("Brief cannot be None")

        if not ctx.voice_durations:
            ctx.voice_durations = []

        all_narrations = [
            ctx.brief.hook,
            *[s.narration for s in ctx.brief.scenes],
            ctx.brief.cta,
        ]

        srt_file = ""
        current_time = 0.0
        index = 1
        for narration, audio_path in zip(all_narrations, ctx.voices):
            audio_file_length = MP3(audio_path).info.length
            narration_splitted = narration.split("|")
            chunk_duration = audio_file_length / len(narration_splitted)

            for narration_chunk in narration_splitted:
                srt_chunk = (
                    self._create_srt_chunk(
                        index,
                        time_from=current_time,
                        time_to=current_time + chunk_duration,
                        text=narration_chunk,
                    )
                    + "\n"
                )
                srt_file += srt_chunk
                current_time += chunk_duration
                index += 1
            ctx.voice_durations.append(audio_file_length)

        subtitles_path = self._write_srt_file(ctx, srt_file)
        ctx.subtitles = subtitles_path

        return ctx

    def load_from_disk(self, ctx: PipelineContext) -> PipelineContext:
        path = Path(ctx.dir) / ctx.topic / "subtitles.srt"
        if not path.exists():
            raise FileNotFoundError(f"Subtitles file does not exist: {path}")

        if not ctx.voices:
            raise ValueError("Voice cannot be None")

        if not ctx.voice_durations:
            ctx.voice_durations = []

        for audio_path in ctx.voices:
            audio_file_length = MP3(audio_path).info.length
            ctx.voice_durations.append(audio_file_length)

        ctx.subtitles = path
        return ctx

    def _write_srt_file(self, ctx: PipelineContext, srt_file: str) -> Path:
        path = Path(ctx.dir) / ctx.topic
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "subtitles.srt", "w", encoding="utf-8") as f:
            f.write(srt_file)
        return path / "subtitles.srt"

    def _create_srt_chunk(
        self, index: int, time_from: float, time_to: float, text: str
    ) -> str:
        chunk = f"{index}\n"
        chunk += f"{self._format_time(time_from)} --> {self._format_time(time_to)}\n"
        chunk += f"{text.strip().upper()}\n"
        return chunk

    def _format_time(self, seconds: float) -> str:
        td = str(timedelta(seconds=seconds))
        if td.rfind(".") == -1:
            srt_format = "0" + td + ",000"
        else:
            srt_format = "0" + td[:-3].replace(".", ",")
        return srt_format
