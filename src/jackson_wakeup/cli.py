from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from .config import AppConfig
from .pipeline import listen_and_generate, run_forever


def main() -> None:
    parser = argparse.ArgumentParser(description="Listen for 'jackson' and generate a frame-ready image")
    parser.add_argument(
        "--vosk-model",
        type=Path,
        default=AppConfig().vosk_model_dir,
        help="Path to Vosk model directory",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Input device index for sounddevice (default: system default)",
    )
    parser.add_argument(
        "--wake",
        type=str,
        default=AppConfig().wake_phrase,
        help="Wake phrase (default: 'jackson')",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    parser.add_argument(
        "--text-input",
        type=str,
        default=None,
        help=(
            "Skip microphone and run a single request provided on the command line. "
            "Example: --text-input \"what is the weather forecast today?\""
        ),
    )

    parser.add_argument(
        "--weather-forecast",
        action="store_true",
        help="Run the daily 5am weather forecast request immediately (test mode).",
    )

    parser.add_argument(
        "--run-forever",
        action="store_true",
        help="Run continuously: keep listening for wake phrase and also trigger a weather forecast every day at 5am local time.",
    )

    parser.add_argument(
        "--no-image-edit",
        action="store_true",
        help="Disable the image edit endpoint (forces text-only image generation)",
    )
    parser.add_argument(
        "--image-edit-size",
        type=str,
        default="none",
        help=(
            "When using image edit, override the edit output size. "
            "Use one of: 1024x1024, 1024x1536, 1536x1024, auto, none. "
            "Set to 'none' to avoid overriding generation_size. (default: 1024x1024)"
        ),
    )

    parser.add_argument(
        "--no-frameo",
        action="store_true",
        help="Disable copying the generated image to a connected Frameo device",
    )
    parser.add_argument(
        "--frameo-dir",
        type=str,
        default=os.environ.get("FRAMEO_DIR"),
        help=(
            "Destination folder for Frameo sync. Prefer a filesystem path like 'E:\\DCIM'. "
            "You may also pass a Windows Shell path like 'This PC\\Frame\\Internal storage\\DCIM' "
            "(requires pywin32). "
            "On Raspberry Pi/Linux PTP devices, you can use gphoto2 paths like 'gphoto2:/store_.../DCIM' "
            "or 'gphoto2:auto'. If unset, this defaults from the FRAMEO_DIR environment variable."
        ),
    )
    parser.add_argument(
        "--frameo-label",
        type=str,
        default=AppConfig().frameo_device_label,
        help="Windows volume label to auto-detect when --frameo-dir is not set (default: Frame)",
    )
    parser.add_argument(
        "--frameo-filename",
        type=str,
        default=AppConfig().frameo_dest_filename,
        help="Filename to overwrite on the frame (default: latest.png)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    edit_override: str | None
    if str(args.image_edit_size).strip().lower() == "none":
        edit_override = None
    else:
        edit_override = str(args.image_edit_size).strip()

    cfg = AppConfig(
        wake_phrase=args.wake,
        vosk_model_dir=args.vosk_model,
        device=args.device,
        verbose=args.verbose,
        image_use_edit_endpoint=(not args.no_image_edit),
        image_edit_output_size_override=edit_override,
        frameo_enabled=(not args.no_frameo),
        frameo_dest_dir=args.frameo_dir,
        frameo_device_label=args.frameo_label,
        frameo_dest_filename=args.frameo_filename,
    )

    if args.weather_forecast:
        # Immediate test of the scheduled feature.
        from .pipeline import _weather_forecast_request

        listen_and_generate(cfg, request_text_override=_weather_forecast_request(cfg))
        return

    if args.run_forever:
        run_forever(cfg)
        return

    if args.text_input is not None:
        listen_and_generate(cfg, request_text_override=args.text_input)
        return

    listen_and_generate(cfg)


if __name__ == "__main__":
    main()
