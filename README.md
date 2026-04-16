# reels-forge

## Description

Generates short videos automatically by creating a script, visuals, images, voiceover, subtitles, and final video using an AI-driven pipeline.

## Requirements

- Python 3.x
- pip
- venv
- ffmpeg (required for video/audio processing, install via Homebrew):

  ```bash
  brew install ffmpeg-full
  ```

- OpenAI API key

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py --topic "your topic"
```

## Flags

- `--topic` (required)
  Topic used to generate the video content.

- `--start-from`
  Start the pipeline from a specific stage. Useful if previous steps are already generated.
  Example:

  ```bash
  python main.py --topic "AI future" --start-from images
  ```

- `--only`
  Run only a single stage of the pipeline.
  Example:

  ```bash
  python main.py --topic "AI future" --only voice
  ```

## Notes

- If `OPENAI_API_KEY` is not set, you will be prompted to enter it.
- Output is saved in the `results/` directory.
- Pipeline stages include: brief → visual → images → voice → subtitles → video.
