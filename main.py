import getpass
import os
from typing import Optional

import click
from langchain_openai import ChatOpenAI
from openai import OpenAI

from pipeline.brief import BriefStage
from pipeline.images import ImagesStage
from pipeline.pipeline import Pipeline, PipelineContext
from pipeline.subtitles import SubtitlesStage
from pipeline.video import VideoStage
from pipeline.visual import VisualStage
from pipeline.voice import VoiceStage


@click.command()
@click.option("--topic", required=True, help="Topic for the video")
@click.option("--start-from", default=None, help="Stage to start from")
@click.option("--only", default=None, help="Run only this stage")
def main(topic: str, start_from: Optional[str], only: Optional[str]) -> None:
    llm = ChatOpenAI(
        model="gpt-5-nano",
    )
    openai_llm = OpenAI()
    ctx = PipelineContext(
        dir="results",
        topic=topic,
    )
    stages = [
        BriefStage(llm),
        VisualStage(llm),
        ImagesStage(openai_llm),
        VoiceStage(openai_llm),
        SubtitlesStage(),
        VideoStage(),
    ]
    pipeline = Pipeline(stages)

    try:
        pipeline.run(ctx, start_from, only)
    except ValueError as e:
        raise click.BadParameter(str(e))


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

    main()
