from datetime import date

import pytest

from fire_cli.query import BBox, DateRange


def test_bbox_parse_and_formats():
    b = BBox.parse("12, 48.5, 19, 51")
    assert b.as_firms() == "12,48.5,19,51"
    assert b.as_cds_area() == [51, 12, 48.5, 19]


@pytest.mark.parametrize("text", ["1,2,3", "10,0,5,1", "0,10,1,5", "0,0,200,1"])
def test_bbox_rejects_invalid(text):
    with pytest.raises(ValueError):
        BBox.parse(text)


def test_bbox_buffered_only_grows_small_sides():
    b = BBox(14.4, 50.0, 14.5, 52.0).buffered(0.5)
    assert b.east - b.west == pytest.approx(0.5)
    assert (b.south, b.north) == (50.0, 52.0)


def test_daterange_chunks_cover_range_exactly():
    r = DateRange.parse("2023-07-01", "2023-07-12")
    chunks = list(r.chunks(5))
    assert [(c.start.day, c.end.day) for c in chunks] == [(1, 5), (6, 10), (11, 12)]


def test_daterange_months_spans_boundary():
    r = DateRange.parse("2023-01-30", "2023-02-02")
    assert list(r.months()) == [(2023, 1, [30, 31]), (2023, 2, [1, 2])]


def test_daterange_clip():
    r = DateRange.parse("2023-01-01", "2023-01-31")
    assert r.clip(date(2023, 1, 10), None) == DateRange(date(2023, 1, 10), date(2023, 1, 31))
    assert r.clip(date(2024, 1, 1), None) is None
