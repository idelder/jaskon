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


def probe_input_device(*, device: int | None, sample_rate: int) -> tuple[bool, int | None]:
    """Return (ok, device_to_use) for an input stream.

    - If `device` is provided, we only test that device.
    - If `device` is None, we try the default input device and, if that fails,
      we scan for the first input-capable device that can actually be opened.

    This is used by the systemd run-forever loop to decide whether wake listening
    can resume after a USB mic reconnect.
    """

    def _try_open(dev: int | None) -> bool:
        try:
            with sd.InputStream(
                samplerate=sample_rate,
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

    if device is not None:
        return (_try_open(device), device)

    # First try the default input device.
    if _try_open(None):
        return (True, None)

    # If default is invalid (often shows up as device -1), search for a working input.
    try:
        devices = sd.query_devices()
    except Exception:
        return (False, None)

    for idx, info in enumerate(devices):
        try:
            if int(info.get("max_input_channels", 0) or 0) <= 0:
                continue
        except Exception:
            continue
        if _try_open(idx):
            return (True, idx)

    return (False, None)


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

        def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                logger.debug("Audio input status: %s", status)
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
                data = audio_queue.get()
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
