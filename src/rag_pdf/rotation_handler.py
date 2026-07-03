from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def is_rotated(rotation: int) -> bool:
    # 90° and 270° pages often contain full-page tables that need minimal boilerplate stripping.
    normalized_rotation = rotation % 360
    return normalized_rotation in (90, 270)


def is_landscape(page_width: float, page_height: float) -> bool:
    if page_height == 0:
        return False
    ratio = page_width / page_height
    return ratio > 1.2


def get_strip_fractions_for_rotation(
        rotation: int,
        page_width: float,
        page_height: float,
        default_fractions: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    if default_fractions is None:
        default_fractions = {
            'left': 0.08,
            'right': 0.08,
            'top': 0.08,
            'bottom': 0.08
        }

    needs_minimal_strip = (
            is_rotated(rotation) or
            is_landscape(page_width, page_height)
    )

    if needs_minimal_strip:
        # Minimal stripping for rotated/landscape pages to preserve full-page tables.
        return {
            'left': 0.02,
            'right': 0.02,
            'top': 0.02,
            'bottom': 0.02
        }
    else:
        return default_fractions.copy()


def should_use_alternative_extractor(
        rotation: int,
        text_length: int,
        page_width: float,
        page_height: float
) -> Tuple[bool, str]:
    if is_rotated(rotation) and text_length < 100:
        return True, "rotated_page_low_yield"

    if is_landscape(page_width, page_height) and text_length < 100:
        return True, "landscape_page_low_yield"

    return False, "normal_extraction_ok"


def get_rotation_metadata(
        rotation: int,
        page_width: float,
        page_height: float,
        text_length: int
) -> Dict[str, any]:
    rotated = is_rotated(rotation)
    landscape = is_landscape(page_width, page_height)
    use_alt, reason = should_use_alternative_extractor(
        rotation, text_length, page_width, page_height
    )

    strip_mode = "minimal" if (rotated or landscape) else "standard"

    return {
        'rotation_degrees': rotation,
        'is_rotated': rotated,
        'is_landscape': landscape,
        'strip_mode': strip_mode,
        'needs_alternative_extractor': use_alt,
        'alternative_reason': reason,
        'aspect_ratio': page_width / page_height if page_height > 0 else 0,
    }


def log_rotation_handling(
        page_number: int,
        rotation: int,
        text_length_before: int,
        text_length_after: int,
        extraction_method: str
):
    if is_rotated(rotation):
        reduction_pct = (
            (1 - text_length_after / text_length_before) * 100
            if text_length_before > 0 else 0
        )

        logger.debug(
            f"Page {page_number}: Rotated {rotation}°, "
            f"extracted {text_length_after} chars "
            f"(reduced {reduction_pct:.1f}% by minimal strip), "
            f"method={extraction_method}"
        )


# Configuration constants
ROTATION_CONFIG = {
    'minimal_strip_fraction': 0.02,
    'standard_strip_fraction': 0.08,
    'landscape_threshold_ratio': 1.2,
    'low_yield_threshold_chars': 100,
}


def get_rotation_config() -> Dict[str, float]:
    return ROTATION_CONFIG.copy()
