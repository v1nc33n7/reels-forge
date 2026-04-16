# Reels Forge 🎞️

## Requirements

The requirements include Python 3.x along with pip and venv. You will also need ffmpeg for video and audio processing, which can be installed via Homebrew using the command `brew install ffmpeg-full`. Additionally, an OpenAI API key is required.

## Usage

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py --topic "Dinner, eat first vegetables, then meat, then carbs to keep your sugar low, fight the brain fog, and see results at first try"
```

## Flags

`--topic`
Sets the main topic for video generation.

`--start-from`
Starts the pipeline at a specific stage (useful if earlier outputs already exist).

```bash
python main.py --topic "Alliance between large technology and pharmaceutical corporations develops a hidden AI-driven health prediction system that raises ethical questions about privacy, consent, and data control" --start-from images
```

`--only`
Runs only one selected pipeline stage.

```bash
python main.py --topic "Astrology and zodiac signs" --only voice
```

## Notes

If `OPENAI_API_KEY` is not set, you will be prompted to enter it.
Output is saved in the `results/` directory.
Pipeline stages include: brief → visual → images → voice → subtitles → video.
