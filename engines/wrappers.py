from qdrant_client import QdrantClient
from .horizon_scanning_engine import (
    HorizonScanningEngine,
    TopicDetectionConfig,
    SynthesisConfig
)

LOOKBACK_UI = {
    "Days":   (7, 30, 1, "days"),
    "Months": (1, 11, 1, "months"),
    "Years":  (1, 1, 1, "years"),
}

HORIZON_UI = {
    # "Days":   (7, 30, 1, "days"),
    "Months": (3, 11, 1, "months"),
    "Years":  (1, 10, 1, "years"),
}

def get_horizon_engine(client: QdrantClient, google_api_key: str, topic_cfg: TopicDetectionConfig, synthesis_cfg: SynthesisConfig):
    return HorizonScanningEngine(
        qdrant_client=client,
        google_api_key=google_api_key,
        topic_cfg=topic_cfg,
        synthesis_cfg=synthesis_cfg
    )

def convert_to_days(length: int, unit: str) -> int:
    return length if unit == "days" else length * 30 if unit == "months" else length * 365