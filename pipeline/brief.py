from dataclasses import dataclass
import dataclasses
from pathlib import Path
from typing import List, cast

import yaml

from langchain.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from pipeline.pipeline import PipelineContext, PipelineStage


@dataclass
class Scene:
    scene_number: int
    duration_seconds: int
    narration: str
    visual_description: str
    emotional_beat: str


@dataclass
class Brief:
    title: str
    total_duration_seconds: int
    hook: str
    scenes: List[Scene]
    cta: str


class BriefStage(PipelineStage):
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm.with_structured_output(Brief)

    @property
    def name(self) -> str:
        return "brief"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        human_msg = HumanMessage(content=f"{ctx.topic}")
        ctx.brief = cast(Brief, self.llm.invoke(input=[_SYSTEM_PROMPT, human_msg]))
        self._save_on_disk(ctx)
        return ctx

    def load_from_disk(self, ctx: PipelineContext) -> PipelineContext:
        path = f"{ctx.dir}/{ctx.topic}/brief.yaml"
        with open(path, encoding="utf-8") as f:
            brief_dict = yaml.load(f, yaml.FullLoader)
            brief_dict["scenes"] = [Scene(**s) for s in brief_dict["scenes"]]
            ctx.brief = Brief(**brief_dict)
        return ctx

    def _save_on_disk(self, ctx: PipelineContext) -> None:
        if not ctx.brief:
            raise ValueError("Brief cannot be None")

        path = f"{ctx.dir}/{ctx.topic}"
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(path + "/brief.yaml", "w", encoding="utf-8") as f:
            yaml.dump(dataclasses.asdict(ctx.brief), f)


BRIEF_RULES = """
HOOK (first 3s):
- Open a real knowledge gap. Make viewer expect uncommon insight.
  Good: "Most people | have no idea | this plant | can save your life."
  Good: "NASA found something | 400 million km away | that changes everything."
  Bad: "In this video..." / "Today I..." / "You won't believe..."
- Must promise real value (mid-age audience distrusts clickbait).

STRUCTURE:
- Arc: Hook → Surprise → Tension → Payoff → CTA.
- Length: 45–75s (TT/Reels), ≤90s (Shorts).
- 7–10 scenes, 5–8s each.
- Every scene escalates. No filler.

WRITING:
- For the ear. Short sentences. Each period = breath.
- No bullets, sequencing words, or meta language ("I", "we", "this video").
- Use concrete numbers (e.g., "3 plants", "72 hours").
- Vary rhythm (long → short punch).
- End with a repeatable fact/tip.

VISUALS (per scene):
- Concrete, filmable, documentary-style (BBC/NatGeo).
- No people, faces, or hands.
- Avoid abstract prompts ("show dehydration").

EMOTION (per scene):
- One: curiosity / unease / revelation / awe / urgency / relief / wonder.
- Must escalate across scenes.

SUBTITLES:
- hook, narration, cta MUST use "|" for cuts.
- Each segment: 2–4 words (max 5).
- Split only at natural pauses.
  Bad: split compounds or numbers ("400 | million km").

CTA (exact match by niche):
- Survival:   "Save this. | You might actually need it."
- Health:     "Share this | with someone who needs to hear it."
- Astronomy:  "Follow for more facts | they never taught you in school."
- Never say "like and subscribe".
"""

_SYSTEM_PROMPT = SystemMessage(
    content="""
You write faceless short-form documentary scripts.
Niches: survival, health, astronomy.
Audience: age 35–60 (values credible, useful info).
Platforms: TikTok, Reels, Shorts.
Format: narration only, no presenters.
"""
    + BRIEF_RULES
)
