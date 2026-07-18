import pytest
from pydantic import ValidationError

from groundmark.models import BBox


def test_bbox_validation_and_union() -> None:
    left = BBox(x0=0, y0=0, x1=0.5, y1=0.5)
    right = BBox(x0=0.4, y0=0.4, x1=1, y1=1)
    assert left.union(right) == BBox(x0=0, y0=0, x1=1, y1=1)
    with pytest.raises(ValidationError):
        BBox(x0=0.8, y0=0, x1=0.2, y1=1)
