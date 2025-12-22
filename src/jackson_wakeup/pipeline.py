from __future__ import annotations

import os
import logging
from datetime import datetime
from pathlib import Path
import threading
import time
from datetime import timedelta
import textwrap
import tempfile
import shutil

from .config import AppConfig
from .openai_client import build_image_prompt, generate_image, generate_search_query
from .stt import WakeWordListener, probe_input_device
from .vosk_setup import ensure_vosk_model
from .web_search import web_search
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont

from .location import resolve_default_location
from .weather import build_weather_summary

from .frameo_sync import (
    copy_image_to_frameo,
    copy_image_to_frameo_gphoto2,
    copy_image_to_frameo_shell_path,
    purge_images_in_gphoto2_folder,
    purge_images_in_dir,
    purge_images_in_shell_path,
    resolve_frameo_destination_dir,
)


logger = logging.getLogger(__name__)


_FRAMEO_BG_LOCK = threading.Lock()


def _sync_frameo_final_image(*, cfg: AppConfig, src_image: Path) -> None:
    """Purge + copy the final image to Frameo (best-effort)."""

    try:
        # Frameo caches by name; always use a unique filename.
        base = Path(cfg.frameo_dest_filename)
        prefix = base.stem or "jackson"
        ext = base.suffix or src_image.suffix or ".png"
        ts = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        unique_name = f"{prefix}_{ts}_{uuid4().hex[:8]}{ext}"

        # Raspberry Pi/Linux: Frameo often exposes PTP/MTP (not mountable storage).
        # Support syncing via gphoto2 by passing --frameo-dir "gphoto2:/store_.../DCIM".
        if (cfg.frameo_dest_dir or "").strip().lower().startswith("gphoto2:"):
            purge_images_in_gphoto2_folder(gphoto2_path=str(cfg.frameo_dest_dir))
            ok = copy_image_to_frameo_gphoto2(
                src_image=src_image,
                gphoto2_path=str(cfg.frameo_dest_dir),
                dest_filename=unique_name,
            )
            if ok:
                logger.info("Copied image to Frameo via gphoto2: %s", cfg.frameo_dest_dir)
            else:
                logger.warning("Frameo gphoto2 sync failed. Verify gphoto2 detects the device and the folder path is correct.")
            return

        dest_dir = resolve_frameo_destination_dir(
            explicit_dir=cfg.frameo_dest_dir,
            device_label=cfg.frameo_device_label,
        )
        if dest_dir is not None:
            deleted = purge_images_in_dir(dest_dir)
            if deleted:
                logger.info("Deleted %d image(s) from Frameo folder", deleted)
            copied = copy_image_to_frameo(
                src_image=src_image,
                dest_dir=dest_dir,
                dest_filename=unique_name,
            )
            logger.info("Copied image to Frameo: %s", copied)
            return

        shell_path = (cfg.frameo_dest_dir or "").strip()
        if shell_path.lower().startswith("this pc\\"):
            deleted = purge_images_in_shell_path(shell_path=shell_path)
            if deleted:
                logger.info("Deleted %d image(s) from Frameo (MTP) folder", deleted)
            # MTP folders can take a moment to refresh after deletes.
            time.sleep(1.5)
            ok = copy_image_to_frameo_shell_path(
                src_image=src_image,
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
            return

        if os.name == "nt":
            logger.warning(
                "Frameo sync enabled but destination not found. "
                "If Windows shows 'This PC\\Frame\\Internal storage\\DCIM', set --frameo-dir to that shell path (requires pywin32) "
                "or use a drive-letter path like 'E:\\DCIM'."
            )
        else:
            logger.warning(
                "Frameo sync enabled but destination not found. "
                "On Linux/Raspberry Pi, many frames are PTP (not mountable). "
                "Use --frameo-dir 'gphoto2:auto' (recommended) or an explicit folder like 'gphoto2:/store_00010001/DCIM'."
            )
    except Exception:
        logger.exception("Failed to copy image to Frameo")


def _start_frameo_final_sync_async(*, cfg: AppConfig, src_image: Path) -> None:
    """Start Frameo sync in the background.

    Stages the image to a stable temp file so subsequent generations can't overwrite it.
    """

    try:
        staged_dir = cfg.output_dir / ".frameo_staging"
        staged_dir.mkdir(parents=True, exist_ok=True)
        staged_path = staged_dir / f"staged_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}{src_image.suffix or '.png'}"
        shutil.copyfile(src_image, staged_path)
    except Exception:
        logger.exception("Failed to stage image for async Frameo sync; falling back to synchronous")
        _sync_frameo_final_image(cfg=cfg, src_image=src_image)
        return

    def _worker() -> None:
        try:
            # Serialize Frameo operations to avoid purge/copy races.
            with _FRAMEO_BG_LOCK:
                _sync_frameo_final_image(cfg=cfg, src_image=staged_path)
        finally:
            try:
                staged_path.unlink(missing_ok=True)
            except Exception:
                pass

    threading.Thread(target=_worker, name="frameo-sync", daemon=True).start()


def _weather_forecast_request(cfg: AppConfig) -> str:
    loc_res = resolve_default_location(
        configured_default=cfg.default_location,
        auto_enabled=cfg.auto_default_location,
        timeout_seconds=cfg.auto_location_timeout_seconds,
        prefer_os=cfg.auto_location_prefer_os,
    )
    loc = (loc_res.location if loc_res else "").strip()
    if loc:
        return (
            f"Hey Jackson, make a nice infographic on today's weather forecast for {loc}, "
            "including numbers and graphics for temperature, humidity, conditions, etc. "
            "and stand outside so I can see the weather. Show me the day's weather, not "
            "the current weather."
        )
    return (
        "Make a nice infographic on today's weather forecast, "
        "including numbers and graphics for temperature, humidity, conditions, etc. "
        "and stand outside so I can see the weather. Show me the day's weather, not "
        "the current weather."
    )


def _build_weather_facts(*, location: str) -> str | None:
    ws = build_weather_summary(location=location)
    if not ws:
        return None
    return (
        f"Source: Open-Meteo\n"
        f"Location: {ws.location}\n"
        f"Date: {ws.as_of_date}\n"
        f"{ws.summary_text}\n"
    )


def _seconds_until_next_local_time(*, hour: int, minute: int) -> float:
    now = datetime.now().astimezone()
    nxt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if nxt <= now:
        nxt = nxt + timedelta(days=1)
    return max(0.0, (nxt - now).total_seconds())


def _render_loading_with_request(*, loading_path: Path, request_text: str, out_path: Path) -> None:
    """Render a loading image with the request text in the top-left.

    Uses Pillow's default font to avoid introducing new font dependencies.
    """

    request_text = (request_text or "").strip()
    if not request_text:
        # If there's no request, just copy the original.
        out_path.write_bytes(loading_path.read_bytes())
        return

    def _capitalize_first_word(s: str) -> str:
        parts = s.split(None, 1)
        if not parts:
            return s
        first = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        if first:
            first = first[:1].upper() + first[1:]
        return (first + (" " + rest if rest else "")).strip()

    # Keep it short-ish; this is a status overlay.
    max_chars = 160
    if len(request_text) > max_chars:
        request_text = request_text[: max_chars - 1].rstrip() + "…"

    request_text = _capitalize_first_word(request_text)

    with Image.open(loading_path) as im:
        im = im.convert("RGBA")
        draw = ImageDraw.Draw(im)

        # Force a scalable font. The tiny/purple-looking text reports typically happen
        # when we fall back to Pillow's bitmap default font.
        # Size tuned per user feedback.
        font_size = max(54, int(im.size[1] * 0.066))
        font = None
        # Try common locations in a best-effort order.
        for candidate in (
            "DejaVuSans.ttf",
            "arial.ttf",
            "Arial.ttf",
        ):
            try:
                font = ImageFont.truetype(candidate, font_size)
                break
            except Exception:
                font = None

        if font is None:
            # Last resort: still render (may be small), but keep color correct.
            font = ImageFont.load_default()

        # Place the request in a readable top-left box.
        margin = max(14, int(im.size[0] * 0.025))
        box_w = max(240, int(im.size[0] * 0.62 * 0.75))
        box_h = max(120, int(im.size[1] * 0.30))

        raw_text = request_text

        def _text_width(s: str) -> int:
            try:
                bbox = draw.textbbox((0, 0), s, font=font)
                return int(bbox[2] - bbox[0])
            except Exception:
                return len(s) * max(6, font_size // 2)

        # Word-wrap into lines within box_w.
        words = raw_text.split()
        lines: list[str] = []
        cur = ""
        for w in words:
            cand = (cur + " " + w).strip() if cur else w
            if _text_width(cand) <= box_w:
                cur = cand
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)

        if not lines:
            lines = textwrap.wrap(raw_text, width=55) or [raw_text]

        # Trim lines to fit height.
        try:
            line_h = int(draw.textbbox((0, 0), "Ag", font=font)[3]) + int(font_size * 0.35)
        except Exception:
            line_h = int(font_size * 1.25)
        max_lines = max(2, box_h // max(1, line_h))
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            # Ellipsize the last line to indicate truncation.
            last = lines[-1]
            if not last.endswith("…"):
                # Remove characters until it fits with an ellipsis.
                while last and _text_width(last + "…") > box_w:
                    last = last[:-1]
                lines[-1] = (last.rstrip() + "…") if last else "…"

        text = "\n".join(lines)

        # User request: black text only with no background.
        spacing = int(font_size * 0.25)
        x, y = margin, margin * 2
        draw.multiline_text((x, y), text, font=font, fill=(0, 0, 0, 255), spacing=spacing)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.convert("RGB").save(out_path, format="PNG", optimize=True)


def _ensure_logging_configured(verbose: bool) -> None:
    root = logging.getLogger()
    if root.handlers:
        # CLI (or hosting app) already configured logging.
        return
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _ensure_vosk_ready(cfg: AppConfig) -> None:
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


def generate_from_request(cfg: AppConfig, *, request_text: str, frameo_async: bool = False) -> Path:
    request_text = (request_text or "").strip()
    if not request_text:
        raise ValueError("request_text is empty")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Show a loading screen on Frameo while the AI work runs.
    # This is best-effort and should never fail the main pipeline.
    if cfg.frameo_enabled:
        try:
            loading_path = cfg.assets_dir / "loading.png"
            if loading_path.exists() and loading_path.is_file():
                ts = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
                loading_name = f"loading_{ts}_{uuid4().hex[:8]}.png"

                with tempfile.TemporaryDirectory(prefix="jackson_loading_") as td:
                    rendered = Path(td) / "loading_overlay.png"
                    _render_loading_with_request(
                        loading_path=loading_path,
                        request_text=request_text,
                        out_path=rendered,
                    )

                    dest_dir = resolve_frameo_destination_dir(
                        explicit_dir=cfg.frameo_dest_dir,
                        device_label=cfg.frameo_device_label,
                    )
                    if dest_dir is not None:
                        purge_images_in_dir(dest_dir)
                        copy_image_to_frameo(
                            src_image=rendered,
                            dest_dir=dest_dir,
                            dest_filename=loading_name,
                        )
                        logger.info("Sent loading screen to Frameo")
                    else:
                        shell_path = (cfg.frameo_dest_dir or "").strip()
                        if shell_path.lower().startswith("gphoto2:"):
                            purge_images_in_gphoto2_folder(gphoto2_path=shell_path)
                            ok = copy_image_to_frameo_gphoto2(
                                src_image=rendered,
                                gphoto2_path=shell_path,
                                dest_filename=loading_name,
                                timeout_seconds=60.0,
                            )
                            if ok:
                                logger.info("Sent loading screen to Frameo (gphoto2)")
                        elif shell_path.lower().startswith("this pc\\"):
                            purge_images_in_shell_path(shell_path=shell_path)
                            # MTP folders can take a moment to refresh after deletes.
                            import time

                            time.sleep(1.5)
                            ok = copy_image_to_frameo_shell_path(
                                src_image=rendered,
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

    loc_res = resolve_default_location(
        configured_default=cfg.default_location,
        auto_enabled=cfg.auto_default_location,
        timeout_seconds=cfg.auto_location_timeout_seconds,
        prefer_os=cfg.auto_location_prefer_os,
    )
    effective_default_location = loc_res.location if loc_res else None
    if loc_res and loc_res.source != "config":
        logger.info("Auto-detected default location: %s", loc_res.location)

    # If this looks like our scheduled weather infographic request, prefer the free
    # Open-Meteo API. If Open-Meteo is unavailable, fall back to web search.
    weather_facts: str | None = None
    is_weather_forecast = "weather forecast" in request_text.lower()
    if is_weather_forecast:
        if effective_default_location:
            try:
                weather_facts = _build_weather_facts(location=effective_default_location)
            except Exception:
                logger.debug("Failed building Open-Meteo weather facts", exc_info=True)
        if not weather_facts:
            logger.info("Weather forecast request detected, but Open-Meteo facts unavailable; falling back to web search")

    search_results = []
    if cfg.enable_web_search and (not weather_facts):
        try:
            query_res = generate_search_query(
                api_key_env=cfg.openai_api_key_env,
                model=cfg.web_search_query_model,
                user_request=request_text,
                now_local_iso=now_local,
                default_location=effective_default_location,
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
        default_location=effective_default_location,
        search_results=search_results,
        facts_context=weather_facts,
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
        if frameo_async:
            logger.info("Starting async Frameo sync")
            _start_frameo_final_sync_async(cfg=cfg, src_image=out_path)
        else:
            _sync_frameo_final_image(cfg=cfg, src_image=out_path)

    return out_path


def listen_and_generate(cfg: AppConfig, *, request_text_override: str | None = None) -> Path:
    _ensure_logging_configured(cfg.verbose)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_vosk_ready(cfg)

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

    return generate_from_request(cfg, request_text=request_text, frameo_async=False)


def run_forever(cfg: AppConfig) -> None:
    """Run continuously: listen for wake requests and also run a daily 5am forecast."""

    _ensure_logging_configured(cfg.verbose)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_vosk_ready(cfg)

    generation_lock = threading.Lock()
    stop_event = threading.Event()

    # Let systemd stop the service cleanly (SIGTERM).
    try:
        import signal

        def _handle_term(_signum: int, _frame) -> None:  # type: ignore[no-untyped-def]
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, _handle_term)
    except Exception:
        pass

    def _is_portaudio_error(exc: BaseException) -> bool:
        t = type(exc)
        if t.__name__ == "PortAudioError":
            return True
        return str(getattr(t, "__module__", "")).startswith("sounddevice")


    def daily_weather_loop() -> None:
        # Fire at 5:00am local time, once per day.
        while not stop_event.is_set():
            wait_s = _seconds_until_next_local_time(hour=5, minute=0)
            logger.info("Next scheduled weather forecast in %.1f minutes", wait_s / 60.0)
            # Sleep in chunks so Ctrl+C shuts down promptly.
            end = time.time() + wait_s
            while time.time() < end and not stop_event.is_set():
                time.sleep(min(5.0, max(0.0, end - time.time())))

            if stop_event.is_set():
                break

            try:
                with generation_lock:
                    logger.info("Running scheduled 5am weather forecast")
                    generate_from_request(cfg, request_text=_weather_forecast_request(cfg), frameo_async=True)
            except Exception:
                logger.exception("Scheduled weather forecast failed")

    t = threading.Thread(target=daily_weather_loop, name="daily-weather", daemon=True)
    t.start()

    try:
        ok, dev, sr = probe_input_device(device=cfg.device, preferred_sample_rate=cfg.sample_rate)
        if not ok or sr is None:
            logger.warning(
                "Microphone/PortAudio unavailable at startup. Continuing in weather-only mode."
            )
            # Keep the daily weather thread alive; periodically retry the mic.
            retry_s = 30.0
            while not stop_event.is_set():
                time.sleep(retry_s)
                ok, dev, sr = probe_input_device(device=cfg.device, preferred_sample_rate=cfg.sample_rate)
                if ok and sr is not None:
                    logger.info(
                        "Microphone available; starting wake listening (device=%s, sample_rate=%s)",
                        str(dev) if dev is not None else "default",
                        sr,
                    )
                    break
            if stop_event.is_set() or not ok or sr is None:
                raise KeyboardInterrupt

        listener = WakeWordListener(
            vosk_model_path=str(cfg.vosk_model_dir),
            sample_rate=int(sr),
            wake_phrase=cfg.wake_phrase,
            device=dev,
            max_request_seconds=cfg.max_request_seconds,
            log_transcript=cfg.verbose,
        )

        while True:
            try:
                logger.info("Listening for wake phrase: %r", cfg.wake_phrase)
                wake = listener.listen_once()
                request_text = wake.request_text
                logger.info("Heard request: %s", request_text)
                with generation_lock:
                    generate_from_request(cfg, request_text=request_text, frameo_async=True)
            except Exception as exc:
                if _is_portaudio_error(exc):
                    logger.warning(
                        "Microphone/PortAudio unavailable (%s). Continuing in weather-only mode.",
                        str(exc).strip() or type(exc).__name__,
                    )
                    # Keep the daily weather thread alive; periodically retry the mic.
                    retry_s = 30.0
                    while not stop_event.is_set():
                        time.sleep(retry_s)
                        try:
                            ok, dev, sr = probe_input_device(device=cfg.device, preferred_sample_rate=cfg.sample_rate)
                            if not ok or sr is None:
                                logger.info("Mic still unavailable; will retry in %.0fs", retry_s)
                                continue

                            listener = WakeWordListener(
                                vosk_model_path=str(cfg.vosk_model_dir),
                                sample_rate=int(sr),
                                wake_phrase=cfg.wake_phrase,
                                device=dev,
                                max_request_seconds=cfg.max_request_seconds,
                                log_transcript=cfg.verbose,
                            )
                            logger.info(
                                "Microphone available; resuming wake listening (device=%s, sample_rate=%s)",
                                str(dev) if dev is not None else "default",
                                sr,
                            )
                            break
                        except Exception as retry_exc:
                            if _is_portaudio_error(retry_exc):
                                logger.info("Mic still unavailable; will retry in %.0fs", retry_s)
                                continue
                            raise
                    if stop_event.is_set():
                        raise KeyboardInterrupt
                    continue
                raise
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        stop_event.set()
