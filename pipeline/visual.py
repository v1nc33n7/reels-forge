from dataclasses import dataclass
import dataclasses
from pathlib import Path
from typing import Any, List, Literal

import yaml
from langchain.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from pipeline.brief import Scene
from pipeline.pipeline import PipelineContext, PipelineStage


@dataclass
class VisualScene:
    scene: Scene
    image_prompt: str
    motion_effect: Literal[
        "slow_zoom_in", "slow_pan_left", "slow_pan_right", "parallax", "static"
    ]
    subtitle_color: Literal["white", "black"]


@dataclass
class Visuals:
    anchor_theme: str
    hook_image_prompt: str
    cta_image_prompt: str
    scenes: List[VisualScene]


class VisualStage(PipelineStage):
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm.with_structured_output(Visuals)

    @property
    def name(self) -> str:
        return "visual"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.brief:
            raise ValueError("Brief cannot be None")

        human_msg = HumanMessage(
            content=f"Create a visual plan for this brief:\n{ctx.brief}"
        )
        ctx.visuals = self._parse_visuals(
            self.llm.invoke(input=[_SYSTEM_PROMPT, human_msg])
        )

        self._save_on_disk(ctx)
        return ctx

    def load_from_disk(self, ctx: PipelineContext) -> PipelineContext:
        path = Path(ctx.dir) / ctx.topic / "visual.yaml"
        with open(path, encoding="utf-8") as f:
            dictionary = yaml.load(f, yaml.FullLoader)
            ctx.visuals = self._parse_visuals(dictionary)
        return ctx

    def _save_on_disk(self, ctx: PipelineContext) -> None:
        if not ctx.visuals:
            raise ValueError("Visuals cannot be None")

        path = Path(ctx.dir) / ctx.topic
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "visual.yaml", "w", encoding="utf-8") as f:
            yaml.dump(dataclasses.asdict(ctx.visuals), f)

    def _parse_visuals(self, data: Any) -> Visuals:
        data["scenes"] = [
            VisualScene(
                scene=Scene(**s["scene"]),
                **{k: v for k, v in s.items() if k != "scene"},
            )
            for s in data["scenes"]
        ]
        return Visuals(**data)


_SYSTEM_PROMPT = SystemMessage(
    content="""
SCOPE:
- Subjects only: environments, landscapes, macro objects, plants, animals, celestial bodies, geology, instruments, natural phenomena.
- No humans in any form: no faces, hands, silhouettes, or body parts.
- If input implies a human, convert to the surrounding environment/object/phenomenon.

EXCLUSIONS (rewrite if present):
- People, faces, crowds, silhouettes, hands, fingers
- Illustrated/painted/cartoon/anime/CGI/3D styles
- Neon glow, holograms, sci-fi UI/HUD, energy beams, lens flares
- Text of any kind: captions, labels, numbers, watermarks, UI, infographics
- Split screens or multi-scene prompts
- Stock clichés (e.g., staged medical objects on plain backgrounds)

STYLE ANCHOR:
- Append verbatim (<20 words).
- Must include: film stock feel, lighting, lens character
  Example: "shot on Arri Alexa, anamorphic lens, natural light, shallow depth, muted earth tones"

PROMPT STRUCTURE (strict order):
1. Subject (no people, no text)
2. Composition (framing, spatial relationships)
3. Lighting (source, direction, matches emotional_beat)
4. Depth of field (focus vs falloff)
5. Camera angle (eye/low/overhead/macro)
6. [style_anchor]

MOTION:
- slow_zoom_in → detail, tension
- slow_pan → wide environments
- parallax → layered depth
- static → rare, high-impact stillness
"""
)
