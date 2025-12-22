from __future__ import annotations

import logging
import json
import queue
import re
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from vosk import KaldiRecognizer, Model


logger = logging.getLogger(__name__)


def _reset_portaudio() -> None:
    """Best-effort reset of PortAudio via sounddevice.

    On some Linux/ALSA setups, unplugging/replugging a USB mic can leave PortAudio
    with a stale device list until it is re-initialized.
    """

    try:
        # These are internal APIs, but are widely used as a pragmatic recovery.
        term = getattr(sd, "_terminate", None)
        init = getattr(sd, "_initialize", None)
        if callable(term):
            term()
        if callable(init):
            init()
    except Exception:
        logger.debug("PortAudio reset attempt failed", exc_info=True)


def probe_input_device(
    *,
    device: int | None,
    preferred_sample_rate: int,
    candidate_sample_rates: list[int] | None = None,
) -> tuple[bool, int | None, int | None]:
    """Return (ok, device_to_use, sample_rate_to_use) for an input stream.

    Common failure mode on Raspberry Pi USB mics: 16kHz is rejected with
    `Invalid sample rate [PaErrorCode -9997]`. In that case we try alternate
    rates (typically 48000 or 44100) and return the first one that can be opened.
    """

    # Build sample-rate candidates with a sensible ordering.
    rates: list[int] = []
    if preferred_sample_rate:
        rates.append(int(preferred_sample_rate))
    if candidate_sample_rates:
        rates.extend(int(r) for r in candidate_sample_rates if r)
    # Common microphone capture rates.
    rates.extend([48000, 44100, 16000])
    # Deduplicate while preserving order.
    seen: set[int] = set()
    rates = [r for r in rates if not (r in seen or seen.add(r))]

    def _try_open(dev: int | None, sr: int) -> bool:
        try:
            with sd.InputStream(
                samplerate=sr,
                channels=1,
                dtype="int16",
                callback=lambda *_a, **_k: None,
                device=dev,
                blocksize=8000,
            ):
                time.sleep(0.05)
            return True
        except Exception:
            return False

    def _probe_once() -> tuple[bool, int | None, int | None]:
        if device is not None:
            for sr in rates:
                if _try_open(device, sr):
                    return (True, device, sr)
            return (False, device, None)

        # First try the default input device.
        for sr in rates:
            if _try_open(None, sr):
                return (True, None, sr)

        # If default is invalid (often shows up as device -1), search for a working input.
        try:
            devices = sd.query_devices()
        except Exception:
            return (False, None, None)

        for idx, info in enumerate(devices):
            try:
                if int(info.get("max_input_channels", 0) or 0) <= 0:
                    continue
            except Exception:
                continue
            for sr in rates:
                if _try_open(idx, sr):
                    return (True, idx, sr)

        return (False, None, None)

    ok, dev, sr = _probe_once()
    if ok:
        return (ok, dev, sr)

    # If we didn't find anything, try a PortAudio reset (helps after USB replug), then probe again.
    _reset_portaudio()
    ok2, dev2, sr2 = _probe_once()
    return (ok2, dev2, sr2)


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


@dataclass(frozen=True)
class WakeResult:
    request_text: str


class WakeWordListener:
    def __init__(
        self,
        *,
        vosk_model_path: str,
        sample_rate: int,
        wake_phrase: str,
        device: int | None = None,
        max_request_seconds: float = 10.0,
        log_transcript: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.wake_phrase = _normalize_text(wake_phrase)
        self._wake_variants = self._build_wake_variants(self.wake_phrase)
        self.device = device
        self.max_request_seconds = max_request_seconds
        self.log_transcript = log_transcript

        self._model = Model(vosk_model_path)
        self._rec = KaldiRecognizer(self._model, self.sample_rate)
        self._rec.SetWords(True)

    @staticmethod
    def _build_wake_variants(wake_phrase: str) -> list[str]:
        """Return acceptable wake phrase variants.

        Vosk frequently mishears "hey" as "he"/"hi"/"hay".
        We keep the matching intentionally simple and conservative.
        """

        tokens = wake_phrase.split()
        if len(tokens) == 2 and tokens[1] == "jackson":
            first = tokens[0]
            variants = {first, "hey", "he", "hi", "hay"}
            return [f"{v} jackson" for v in sorted(variants)]
        return [wake_phrase]

    def _matches_wake(self, text: str) -> bool:
        text = _normalize_text(text)
        return any(phrase in text for phrase in self._wake_variants)

    def listen_once(self) -> WakeResult:
        """Blocks until it hears the wake phrase, then returns the subsequent request text."""

        audio_queue: queue.Queue[bytes] = queue.Queue()

        # If the USB mic is unplugged, PortAudio/ALSA sometimes stops delivering audio
        # without raising promptly. We treat a prolonged lack of audio as a disconnect.
        last_audio_time = time.time()
        stall_timeout_s = 5.0

        def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                logger.debug("Audio input status: %s", status)
            nonlocal last_audio_time
            last_audio_time = time.time()
            audio_queue.put(indata.tobytes())

        waiting_for_wake = True
        waiting_for_request = False
        request_deadline = 0.0

        logger.debug(
            "Starting mic stream (sr=%s, device=%s). Wake phrase=%r",
            self.sample_rate,
            self.device,
            self.wake_phrase,
        )

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            callback=callback,
            device=self.device,
            blocksize=8000,
        ):
            while True:
                try:
                    data = audio_queue.get(timeout=1.0)
                except queue.Empty:
                    # No audio frames delivered recently -> likely device unplug or ALSA stall.
                    if time.time() - last_audio_time > stall_timeout_s:
                        raise sd.PortAudioError(
                            f"Audio input stalled for {stall_timeout_s:.0f}s (mic disconnected?)"
                        )
                    continue
                if self._rec.AcceptWaveform(data):
                    res = json.loads(self._rec.Result())
                    text = _normalize_text(res.get("text", ""))

                    if self.log_transcript and text:
                        # Show everything the recognizer finalizes so you can debug mic vs wake phrase.
                        logger.info("STT(final): %s", text)

                    if not text:
                        if waiting_for_request and time.time() > request_deadline:
                            logger.debug("Request window expired; returning to wake listening")
                            waiting_for_request = False
                            waiting_for_wake = True
                        continue

                    if waiting_for_wake:
                        idx = -1
                        matched_phrase = None
                        for phrase in self._wake_variants:
                            idx = text.find(phrase)
                            if idx >= 0:
                                matched_phrase = phrase
                                break

                        if idx >= 0 and matched_phrase is not None:
                            logger.debug("Wake phrase detected (%r) in: %r", matched_phrase, text)
                            after = text[idx + len(matched_phrase) :].strip()
                            if after:
                                return WakeResult(request_text=after)
                            waiting_for_wake = False
                            waiting_for_request = True
                            request_deadline = time.time() + self.max_request_seconds
                            logger.debug(
                                "Wake phrase heard; waiting up to %.1fs for request",
                                self.max_request_seconds,
                            )
                        elif self.log_transcript and "jackson" in text:
                            logger.info(
                                "Heard 'jackson' but wake phrase didn't match. Expected one of: %s",
                                ", ".join(self._wake_variants),
                            )
                        continue

                    if waiting_for_request:
                        # The next finalized utterance becomes the request.
                        return WakeResult(request_text=text)

                else:
                    # partial results are less reliable, but are useful for debugging.
                    if self.log_transcript:
                        try:
                            pres = json.loads(self._rec.PartialResult())
                            ptext = _normalize_text(pres.get("partial", ""))
                            if ptext:
                                logger.debug("STT(partial): %s", ptext)
                        except Exception:
                            pass

                    if waiting_for_request and time.time() > request_deadline:
                        logger.debug("Request window expired; returning to wake listening")
                        waiting_for_request = False
                        waiting_for_wake = True
