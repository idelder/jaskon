from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    wake_phrase: str = "jackson"

    # Logging
    verbose: bool = False

    # Speech-to-text (Vosk)
    vosk_model_dir: Path = Path("models") / "vosk-model"
    sample_rate: int = 16000
    device: int | None = None

    # Wake behavior
    max_request_seconds: float = 10.0

    # OpenAI
    openai_api_key_env: str = "OPENAI_API_KEY"
    prompt_model: str = "gpt-4o-mini"
    image_model: str = "gpt-image-1-mini"

    # Output
    assets_dir: Path = Path("assets")
    output_dir: Path = Path("output")
    output_filename: str = "latest.png"

    # Image target for 16:9 photo frame
    image_width: int = 1729
    image_height: int = 1024

    # OpenAI image size (must be supported by the image endpoint; output will be cropped/resized to image_width x image_height)
    # Supported values commonly include: "1024x1024", "1024x1536", "1536x1024", "auto"
    image_generation_size: str = "1536x1024"

    # If supported by the OpenAI image model, use an image-edit call with a reference photo from assets_dir.
    # This allows the image model to "see" what Jackson looks like.
    image_use_edit_endpoint: bool = True

    # When using the edit endpoint, optionally override the output size.
    # Many edit endpoints work best with square sizes. Set to None to avoid overriding.
    image_edit_output_size_override: str | None = None #"1024x1024"

    # Web search grounding (optional)
    enable_web_search: bool = True
    # Supported: "serpapi" (default), "bing"
    web_search_provider: str = "serpapi"
    web_search_api_key_env: str = "WEB_SEARCH_API_KEY"
    # Model to turn the spoken request into an effective web search query.
    web_search_query_model: str = "gpt-4o-mini"
    web_search_results_k: int = 5
    web_search_timeout_seconds: float = 10.0

    # Optional: helps queries like "weather" or "sunset" be location-accurate.
    default_location: str | None = "Boston, MA"

    # Only used when web_search_provider == "bing"
    bing_search_endpoint: str | None = None
