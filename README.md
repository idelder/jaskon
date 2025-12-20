# jackson-wakeup

Listens to your microphone for the wake phrase **"hey jackson"**. When heard, it:

1) Uses OpenAI (ChatGPT) to generate an **image-generation prompt** for a **16:9** image that looks good on an **11\" digital photo frame**, based on whatever you say after the wake phrase.
2) Uses OpenAI image generation to create the image and saves it to `output/latest.png`.

The generated image is an **anthropomorphised cartoon** of your dog Jackson, using a **local reference photo** from `assets/`.

## Setup

## Raspberry Pi (recommended)

This project runs well on a Raspberry Pi when you install the system audio libraries for `sounddevice`/PortAudio and run it as a `systemd` service.

### 0) Install OS + enable SSH

- OS: **Raspberry Pi OS Lite 64-bit (Bookworm)** is a good default.
- Use Raspberry Pi Imager → choose your OS → **enable SSH** in the Imager settings.
- Boot the Pi, then connect:
  - On your LAN: `ssh pi@raspberrypi.local` (or use the Pi’s IP)

### 1) Install system packages

On the Pi:

- `sudo apt update`
- `sudo apt install -y git python3 python3-venv python3-pip portaudio19-dev libasound2-dev libjpeg-dev zlib1g-dev`

### 2) Install this project

Clone and install in a venv:

- `git clone <your repo url> jackson`
- `cd jackson`
- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `python -m pip install -U pip`
- `python -m pip install -e .`

### 3) Add a Jackson photo + (optional) loading screen

- Put at least one photo in `assets/` (e.g. `assets/jackson.jpg`).
- Optional: `assets/loading.png` is displayed immediately on the frame while generation runs.

### 4) Vosk model

The app can attempt to download a Vosk model on first run. If you prefer manual setup, unzip an English model into:

- `models/vosk-model/`

### 5) Set environment variables

For interactive testing:

- `export OPENAI_API_KEY="..."`

Optional (web search fallback):

- `export WEB_SEARCH_API_KEY="..."`

### 6) Test run

- `python -m jackson_wakeup.cli --text-input "make jackson a wizard" --no-frameo`

To run continuously (wake listening + daily 5am weather):

- `python -m jackson_wakeup.cli --run-forever`

### Frameo sync (Pi/Linux)

Many Frameo devices expose **PTP/MTP-style USB** (not USB Mass Storage). In that case they:

- show up in `lsusb`
- do NOT show up in `lsblk`
- are not automatically mountable under `/media/...`

#### Diagnose USB mode

- `lsusb | grep -i -E 'frame|harmony|2207'`
- Check interface classes (example ID shown; adjust if needed):
  - `sudo lsusb -d 2207:0012 -v 2>/dev/null | sed -n '/Interface Descriptor/,+20p' | egrep -i "bInterfaceClass|bInterfaceSubClass|bInterfaceProtocol|MTP|PTP"`

If you see `Imaging` / `PTP`, use the `gphoto2` method below.

#### Recommended: sync via gphoto2 (PTP)

Install tools:

- `sudo apt update`
- `sudo apt install -y gphoto2`

Verify detection:

- `gphoto2 --auto-detect`

If `gphoto2 --auto-detect` shows nothing, try:

- `sudo gphoto2 --auto-detect`

If it works with `sudo` but not without, you likely need USB permissions (udev rule). A simple option:

- `sudo cp scripts/99-frameo.rules /etc/udev/rules.d/99-frameo.rules`
- `sudo udevadm control --reload-rules`
- unplug/replug the frame

Find the storage + folders:

- First capture the concrete port (example: `usb:001,005`):
  - `gphoto2 --auto-detect`
- Then use that port for folder discovery:
  - `gphoto2 --port usb:001,005 --storage-info`
  - `gphoto2 --port usb:001,005 --list-folders`

If you see: `The supplied vendor or product id (0x0,0x0) is not valid`, it usually means gphoto2 did not successfully detect a camera/device. Re-run the steps above (especially `--auto-detect`, try `sudo`, and check USB disconnects in `dmesg`).

Pick the folder that corresponds to DCIM (often something like `/store_00010001/DCIM`).

Recommended (no hard-coded store path):

- `python -m jackson_wakeup.cli --text-input "make jackson a wizard" --frameo-dir "gphoto2:auto"`

Or explicitly:

- `python -m jackson_wakeup.cli --text-input "make jackson a wizard" --frameo-dir "gphoto2:/store_00010001/DCIM"`

For the startup service, set `FRAMEO_DIR` in `/etc/jackson-wakeup.env` to `gphoto2:auto` (or the explicit `gphoto2:/...` value).

#### Troubleshooting: USB disconnect/reconnect loops

If `dmesg` shows repeated USB disconnect/reconnect:

- Try a different USB cable (many are power-only or unreliable for data).
- Use a powered USB hub.
- Consider disabling USB autosuspend (Pi OS): add `usbcore.autosuspend=-1` to `/boot/firmware/cmdline.txt` and reboot.

### Microphone notes (Pi)

- Check devices: `python -c "import sounddevice as sd; print(sd.query_devices())"`
- If needed, pick an input device index: `--device N`

### Remote access (optional): Tailscale

Tailscale makes it easy to SSH into the Pi from anywhere without opening router ports.

- Install: `curl -fsSL https://tailscale.com/install.sh | sh`
- Authenticate + enable Tailscale SSH: `sudo tailscale up --ssh`
- Then SSH from your laptop: `ssh pi@<pi-hostname>` (over your tailnet)

### Run on startup (systemd)

1) Create an env file:

- `sudo cp scripts/jackson-wakeup.env.example /etc/jackson-wakeup.env`
- `sudo nano /etc/jackson-wakeup.env`  (set `OPENAI_API_KEY=...` and `FRAMEO_DIR=...`)

2) Install the service:

- `sudo cp scripts/jackson-wakeup.service /etc/systemd/system/jackson-wakeup.service`
- Edit paths inside the unit if your repo isn’t at `/home/pi/jackson`.

3) Enable + start:

- `sudo systemctl daemon-reload`
- `sudo systemctl enable jackson-wakeup`
- `sudo systemctl start jackson-wakeup`

4) Logs:

- `journalctl -u jackson-wakeup -f`

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
