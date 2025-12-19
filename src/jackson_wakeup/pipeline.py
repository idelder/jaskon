from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from .config import AppConfig
from .openai_client import build_image_prompt, generate_image, generate_search_query
from .stt import WakeWordListener
from .vosk_setup import ensure_vosk_model
from .web_search import web_search
from uuid import uuid4

from .frameo_sync import (
    copy_image_to_frameo,
    copy_image_to_frameo_shell_path,
    purge_images_in_dir,
    purge_images_in_shell_path,
    resolve_frameo_destination_dir,
)


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


def listen_and_generate(cfg: AppConfig, *, request_text_override: str | None = None) -> Path:
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

    if request_text_override is None:
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
        request_text = wake.request_text
        logger.info("Heard request: %s", request_text)
    else:
        request_text = request_text_override.strip()
        logger.info("Test run: skipping microphone. Using request: %s", request_text)

    # Show a loading screen on Frameo while the AI work runs.
    # This is best-effort and should never fail the main pipeline.
    if cfg.frameo_enabled:
        try:
            loading_path = cfg.assets_dir / "loading.png"
            if loading_path.exists() and loading_path.is_file():
                ts = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
                loading_name = f"loading_{ts}_{uuid4().hex[:8]}{loading_path.suffix or '.png'}"

                dest_dir = resolve_frameo_destination_dir(
                    explicit_dir=cfg.frameo_dest_dir,
                    device_label=cfg.frameo_device_label,
                )
                if dest_dir is not None:
                    purge_images_in_dir(dest_dir)
                    copy_image_to_frameo(
                        src_image=loading_path,
                        dest_dir=dest_dir,
                        dest_filename=loading_name,
                    )
                    logger.info("Sent loading screen to Frameo")
                else:
                    shell_path = (cfg.frameo_dest_dir or "").strip()
                    if shell_path.lower().startswith("this pc\\"):
                        purge_images_in_shell_path(shell_path=shell_path)
                        # MTP folders can take a moment to refresh after deletes.
                        import time

                        time.sleep(1.5)
                        ok = copy_image_to_frameo_shell_path(
                            src_image=loading_path,
                            shell_path=shell_path,
                            dest_filename=loading_name,
                            timeout_seconds=30.0,
                        )
                        if ok:
                            logger.info("Sent loading screen to Frameo (MTP)")
            else:
                logger.debug("No loading screen found at: %s", loading_path)
        except Exception:
            logger.debug("Failed to send loading screen to Frameo", exc_info=True)

    now_local = datetime.now().astimezone().isoformat(timespec="seconds")

    search_results = []
    if cfg.enable_web_search:
        try:
            query_res = generate_search_query(
                api_key_env=cfg.openai_api_key_env,
                model=cfg.web_search_query_model,
                user_request=request_text,
                now_local_iso=now_local,
                default_location=cfg.default_location,
            )
            logger.info("Generated web search query: %s", query_res.query)
            search_results = web_search(cfg, request_text, query_override=query_res.query)
        except Exception:
            logger.exception("Failed to generate/search web context; continuing without search")

    prompt_res = build_image_prompt(
        api_key_env=cfg.openai_api_key_env,
        model=cfg.prompt_model,
        user_request=request_text,
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

    if cfg.frameo_enabled:
        try:
            # Frameo caches by name; always use a unique filename.
            base = Path(cfg.frameo_dest_filename)
            prefix = base.stem or "jackson"
            ext = base.suffix or out_path.suffix or ".png"
            ts = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
            unique_name = f"{prefix}_{ts}_{uuid4().hex[:8]}{ext}"

            dest_dir = resolve_frameo_destination_dir(
                explicit_dir=cfg.frameo_dest_dir,
                device_label=cfg.frameo_device_label,
            )
            if dest_dir is not None:
                deleted = purge_images_in_dir(dest_dir)
                if deleted:
                    logger.info("Deleted %d image(s) from Frameo folder", deleted)
                copied = copy_image_to_frameo(
                    src_image=out_path,
                    dest_dir=dest_dir,
                    dest_filename=unique_name,
                )
                logger.info("Copied image to Frameo: %s", copied)
            else:
                shell_path = (cfg.frameo_dest_dir or "").strip()
                if shell_path.lower().startswith("this pc\\"):
                    deleted = purge_images_in_shell_path(shell_path=shell_path)
                    if deleted:
                        logger.info("Deleted %d image(s) from Frameo (MTP) folder", deleted)
                    # MTP folders can take a moment to refresh after deletes.
                    import time
                    time.sleep(1.5)
                    ok = copy_image_to_frameo_shell_path(
                        src_image=out_path,
                        shell_path=shell_path,
                        dest_filename=unique_name,
                    )
                    if ok:
                        logger.info("Copied image to Frameo via Windows Shell path: %s", shell_path)
                    else:
                        logger.warning(
                            "Frameo sync enabled but could not copy via Windows Shell path. "
                            "If this device supports a drive letter, prefer --frameo-dir 'E:\\DCIM'."
                        )
                else:
                    logger.warning(
                        "Frameo sync enabled but destination not found. "
                        "If Windows shows 'This PC\\Frame\\Internal storage\\DCIM', set --frameo-dir to that shell path (requires pywin32) "
                        "or use a drive-letter path like 'E:\\DCIM'."
                    )
        except Exception:
            logger.exception("Failed to copy image to Frameo")

    return out_path
