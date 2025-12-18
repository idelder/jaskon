# jackson-wakeup

Listens to your microphone for the wake phrase **"hey jackson"**. When heard, it:

1) Uses OpenAI (ChatGPT) to generate an **image-generation prompt** for a **16:9** image that looks good on an **11\" digital photo frame**, based on whatever you say after the wake phrase.
2) Uses OpenAI image generation to create the image and saves it to `output/latest.png`.

The generated image is an **anthropomorphised cartoon** of your dog Jackson, using a **local reference photo** from `assets/`.

## Setup

### Directory expectations

At runtime the project expects these directories (created automatically where possible):

- `assets/` — put at least one reference photo of Jackson here
- `models/vosk-model/` — Vosk speech model directory (can be downloaded automatically on first run)
- `output/` — generated image output (`output/latest.png`)

### 1) Put a Jackson photo in assets/

Copy at least one image into `assets/` (supported: `.jpg`, `.png`, `.webp`).

Example:
- `assets/jackson.jpg`

### 2) Install Python deps

From the project root (the folder containing `pyproject.toml`):

- `py -m pip install -e .`

If you are *not* in the project root, pass the path explicitly:
- `py -m pip install -e "c:\\Users\\ianda\\OneDrive - University of Toronto\\Documents\\temoa_projects\\jackson"`

Note: on Windows, `pip` and `python` can point to the Microsoft Store alias. Using `py -m pip` ensures you install into the same Python that runs the app.

### 3) Get a Vosk speech model

This project uses Vosk for offline streaming speech-to-text.

Download an English model and unzip it into:
- `models/vosk-model/`

For example, you can use a Vosk “small” English model zip from the Vosk project releases, unzip it, and rename the extracted folder to `vosk-model`.

### 4) Set your OpenAI key

Set an environment variable.

PowerShell (current session):
- `$env:OPENAI_API_KEY = "..."`

PowerShell (persist for future shells):
- `setx OPENAI_API_KEY "..."`

### 4b) (Optional) Enable web search grounding

If you want the prompt to be accurate for things like *current weather / location / date-sensitive context*, enable web search.

This project supports:
- `serpapi` (default)
- `bing`

Set a search API key:
- `$env:WEB_SEARCH_API_KEY = "..."`

Optional (recommended) location hint for queries like "weather":
- set `default_location` in [src/jackson_wakeup/config.py](src/jackson_wakeup/config.py) (e.g. `"Toronto, ON"`)

### 5) Run

- `jackson-listen`

Verbose logging:
- `jackson-listen --verbose`

If the command isn't found (Windows PATH), run it via the module entrypoint:
- `py -m jackson_wakeup --verbose`

(`jackson-listen` is installed into your Python "Scripts" directory; it may not be on PATH depending on how Python was installed.)

Then say:
- "hey jackson make him a wizard in a library"

Output:
- `output/latest.png`

## Notes (Windows)

- Microphone capture uses `sounddevice`, which depends on PortAudio.
  - If `sounddevice` installation fails, install the Microsoft C++ Build Tools and try again.
- If your mic isn’t the default device, pass `--device N`.

## CLI options

- `--wake "hey jackson"` (change wake phrase)
- `--vosk-model path/to/model` (override model dir)
- `--device 2` (choose input device index)
