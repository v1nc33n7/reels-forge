from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

from console import console

if TYPE_CHECKING:
    from pipeline.brief import Brief
    from pipeline.visual import Visuals


@dataclass
class PipelineContext:
    dir: str
    topic: str
    brief: Optional[Brief] = None
    visuals: Optional[Visuals] = None
    images: Optional[List[Path]] = None
    voice: Optional[List[Path]] = None
    subtitles: Optional[Any] = None
    video: Optional[Any] = None


class PipelineStage(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ""

    @abstractmethod
    def run(self, ctx: PipelineContext) -> PipelineContext:
        ""

    @abstractmethod
    def load_from_disk(self, ctx: PipelineContext) -> PipelineContext:
        ""

    def _dir(self, ctx: PipelineContext) -> Path:
        return Path(f"{ctx.dir}/{ctx.topic}/{self.name}")

    def _require(self, value: Optional[Any], name: str) -> Any:
        if not value:
            raise ValueError(f"{name} cannot be None")
        return value


class Pipeline:
    def __init__(self, stages: List[PipelineStage]) -> None:
        self.stages = stages

    def run(
        self,
        ctx: PipelineContext,
        start_from: Optional[str] = None,
        only: Optional[str] = None,
    ):
        if start_from and only:
            raise ValueError(
                "Cannot specify both 'start_from' and 'only' in the same run"
            )

        stage_name = only or start_from
        if stage_name:
            index = self._get_stage_index(stage_name)
            before_steps = self.stages[:index]
            after_steps = self.stages[index + 1 :]

            for s in before_steps:
                ctx = self._load_stage(s, ctx)

            step = self.stages[index]
            ctx = self._run_stage(step, ctx)

            if only:
                return

            for s in after_steps:
                ctx = self._run_stage(s, ctx)
        else:
            for s in self.stages:
                ctx = self._run_stage(s, ctx)

    def _get_stage_index(self, name: str) -> int:
        for i, stage in enumerate(self.stages):
            if stage.name == name:
                return i
        raise ValueError(f"Stage {name} doesn't exists")

    def _run_stage(self, stage: PipelineStage, ctx: PipelineContext) -> PipelineContext:
        with console.status(f"Running stage: {stage.name}"):
            ctx = stage.run(ctx)
        console.log(f"Completed stage: {stage.name}")
        return ctx

    def _load_stage(
        self, stage: PipelineStage, ctx: PipelineContext
    ) -> PipelineContext:
        console.log(f"Loading from disk: {stage.name}")
        return stage.load_from_disk(ctx)
