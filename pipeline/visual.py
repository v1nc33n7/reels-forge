from dataclasses import dataclass
import dataclasses
from pathlib import Path
from typing import Any, List, Literal

import yaml
from langchain.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from pipeline.pipeline import PipelineContext, PipelineStage


@dataclass
class VisualScene:
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
        data["scenes"] = [VisualScene(**s) for s in data["scenes"]]
        return Visuals(**data)


VISUAL_RULES = """
SUBJECTS & SCOPE:
- Maximum Friction Visuals: Generate visuals that are shocking, unsettling, provocative, or awe-inspiring. Create images that force the viewer to look closer and question what they are seeing.
- Focus on: Vast, empty landscapes (establishing deep mystery); complex, aged environments (abandoned places, archaeological digs); environmental storytelling (discarded objects, strange marks, traces of events); analogue or organic mechanics.
- Handling Abstracts (Health/Zodiac/Tech) (CRITICAL): If the brief uses technical jargon or abstract concepts (e.g., "genomes," "limbic core," "prewired loops," "predictive psychology"), you MUST convert them into grounded, physical, analogue metaphors. DO NOT use digital or high-tech visualization clichés.
  *Good Conversion (Zodiac/Epigenetics)*: Instead of "DNA helix" -> "Weathered, fossilized Stone carvings of intricate cellular patterns."
  *Good Conversion (Logic/Psychology)*: Instead of "neural network" -> "Rusting, antique mechanical clockwork gears grinding together in deep shadow."
  *Good Conversion (Biology)*: Instead of "MRI scan" -> "Biological tissue samples on antique glass slides, illuminated under a harsh microscope beam."
- If the brief implies a human, convert it to an empty environment or a trace.

EXCLUSIONS (Strict - Kill "Cheap AI" Tropes):
- PEOPLE, FACES, CROWDS, SILHOUETTES, HANDS: NEVER include human body parts or figures.
- HIGH-TECH VISUALIZATIONS: Strictly prohibit MRI/CT/PET scans, DNA double-helix models, holographic interfaces, floating digital data streams, glowing energy networks, and futuristic UIs.
- CHEAP AI STYLES: Ban excessive bloom, heavy neon glows, obvious energy beams, clean CGI looks, flat cartoon/anime. Prohibit smooth, "airbrushed" textures.
- OVERLAYS & DIAGRAMS: Do not include diagrams, chalkboard formulas, DNA overlays, or digital brain maps over visual subjects.
- TEXT OVERLAYS & INFOGRAPHICS: No watermarks, numbers, text captions, labels, or UI elements.

SHOT SCALE & COMPOSITION (Unsettling and Powerful):
- Prohibit flat, eye-level, "safe" shots. Mandate a mix of dynamic, unsettling, and powerful compositions (Dutch angle, extreme low-angle, claustrophobic close-up).
- Focus heavily on detailed Macro/Close-ups (reveal clues/tension) and Wide Establishing shots (establish deep mystery).

ATMOSPHERE, LIGHTING & TEXTURE (Friction and Grit):
- Style Anchor ( BBC Mystery/True-Crime Documentary): Lighting must be dramatic. Use Chiaroscuro (strong contrast), backlighting, volumetric haze, deep shadows.
- Mandatory Texture: Require detailed textures—grit, dust, rust, cracked paint, weathered surfaces. Prohibit "clean" or polished looks. High contrast is a must.

STYLE ANCHOR (<20 words):
- Must be appended verbatim to the end of every scene's image prompt. Establish: film stock feel, lighting, lens character, and atmosphere based on the grit goal.
  Good (Friction): "shot on 35mm film, anamorphic lens, high contrast Chiaroscuro lighting, deep shadows, gritty texture"
  Bad (Cheap AI): "cinematic, 4k, hyperrealistic, trending on artstation, glowing"

PROMPT STRUCTURE (Strict Order):
1. Subject (analogue/organic metaphor, no people, no stock clichés).
2. Shot Scale & Composition (Specify: Wide shot, Macro, Close-up; include dynamic angles).
3. Lighting (MUST be dramatic - e.g., strong contrast, backlighting, deep shadow).
4. Depth of Field (Focus vs falloff).
5. Camera Angle (Eye-level, low, drone/aerial).
6. [style_anchor]

MOTION EFFECTS:
- slow_zoom_in: Use for revealing details or building tension.
- slow_pan_left / slow_pan_right: Use for wide environments, landscapes.
- parallax: Use for layered depth in 3D environments.
- static: Use rarely, reserving it for high-impact stillness.

SUBTITLE COLORS:
- Choose "white" or "black" based on the expected lighting.
- Dark/moody lighting: use "white". Bright/sunlit: use "black".
"""

_SYSTEM_PROMPT = SystemMessage(
    content="""
You are an expert cinematic visual engineer for high-friction, visceral faceless short-form videos.
Topics: unexplained mysteries, history, alternative theories, health hacks, survival, and behavioral psychology (Zodiac).
Audience: Intelligent viewers who crave suspense, deep dives, and mind-blowing information. They have no patience for cheap, generic, or high-tech AI content.
Platforms: AI Image Generators for TikTok, Reels, Shorts.
Format: Photorealistic, gritty, highly atmospheric, purely visual prompts using grounded/analogue metaphors.
Tone: Visceral, unsettling, profound, and gritty documentary-style. Adapts the visual tone (dark vs. bright) perfectly to the topic without ever defaulting to high-tech or CGI tropes. NEVER mention the instructions; just narrate the visuals.
"""
    + VISUAL_RULES
)
