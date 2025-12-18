from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from .config import AppConfig
from .openai_client import build_image_prompt, generate_image, generate_search_query
from .stt import WakeWordListener
from .vosk_setup import ensure_vosk_model
from .web_search import web_search


logger = logging.getLogger(__name__)


def _ensure_logging_configured(verbose: bool) -> None:
    root = logging.getLogger()
    if root.handlers:
        # CLI (or hosting app) already configured logging.
        return
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def listen_and_generate(cfg: AppConfig) -> Path:
    _ensure_logging_configured(cfg.verbose)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Optional convenience: if the model folder doesn't exist, attempt download.
    # You can override with your own URL by calling ensure_vosk_model yourself.
    if not cfg.vosk_model_dir.exists():
        try:
            logger.info("Vosk model not found; attempting download...")
            ensure_vosk_model(
                cfg.vosk_model_dir,
                url="https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
            )
        except Exception:
            # Fall back to the normal error from Vosk if it's still missing.
            logger.exception("Vosk model download failed")
            pass

    listener = WakeWordListener(
        vosk_model_path=str(cfg.vosk_model_dir),
        sample_rate=cfg.sample_rate,
        wake_phrase=cfg.wake_phrase,
        device=cfg.device,
        max_request_seconds=cfg.max_request_seconds,
        log_transcript=cfg.verbose,
    )

    logger.info("Listening for wake phrase: %r", cfg.wake_phrase)
    wake = listener.listen_once()
    logger.info("Heard request: %s", wake.request_text)
    now_local = datetime.now().astimezone().isoformat(timespec="seconds")

    search_results = []
    if cfg.enable_web_search:
        try:
            query_res = generate_search_query(
                api_key_env=cfg.openai_api_key_env,
                model=cfg.web_search_query_model,
                user_request=wake.request_text,
                now_local_iso=now_local,
                default_location=cfg.default_location,
            )
            logger.info("Generated web search query: %s", query_res.query)
            search_results = web_search(cfg, wake.request_text, query_override=query_res.query)
        except Exception:
            logger.exception("Failed to generate/search web context; continuing without search")

    prompt_res = build_image_prompt(
        api_key_env=cfg.openai_api_key_env,
        model=cfg.prompt_model,
        user_request=wake.request_text,
        assets_dir=cfg.assets_dir,
        frame_width=cfg.image_width,
        frame_height=cfg.image_height,
        now_local_iso=now_local,
        default_location=cfg.default_location,
        search_results=search_results,
    )

    logger.info("Generated image prompt. Calling image model...")
    out_path = cfg.output_dir / cfg.output_filename
    generate_image(
        api_key_env=cfg.openai_api_key_env,
        model=cfg.image_model,
        prompt=prompt_res.prompt,
        width=cfg.image_width,
        height=cfg.image_height,
        generation_size=cfg.image_generation_size,
        assets_dir=cfg.assets_dir,
        use_edit_endpoint=cfg.image_use_edit_endpoint,
        edit_output_size_override=cfg.image_edit_output_size_override,
        out_path=out_path,
    )

    logger.info("Saved image to: %s", out_path)
    return out_path
