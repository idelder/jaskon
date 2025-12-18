from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import AppConfig
from .pipeline import listen_and_generate


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
    )

    if args.text_input is not None:
        listen_and_generate(cfg, request_text_override=args.text_input)
    else:
        listen_and_generate(cfg)


if __name__ == "__main__":
    main()
