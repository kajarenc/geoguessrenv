import types

import pytest

from geoguess_env.geometry_utils import GeometryUtils


class Dummy:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon


def test_coerce_lat_lon_from_mapping():
    lat, lon = GeometryUtils.coerce_lat_lon({"lat": 12.5, "lon": -7.25}, "err")
    assert lat == pytest.approx(12.5)
    assert lon == pytest.approx(-7.25)


def test_coerce_lat_lon_from_sequence():
    lat, lon = GeometryUtils.coerce_lat_lon([1, 2], "err")
    assert lat == 1.0
    assert lon == 2.0


def test_coerce_lat_lon_from_object_attrs():
    obj = Dummy(47.6, -122.3)
    lat, lon = GeometryUtils.coerce_lat_lon(obj, "err")
    assert lat == pytest.approx(47.6)
    assert lon == pytest.approx(-122.3)


def test_coerce_lat_lon_errors_on_missing_values():
    with pytest.raises(ValueError):
        GeometryUtils.coerce_lat_lon({}, "missing lat/lon")
    with pytest.raises(ValueError):
        GeometryUtils.coerce_lat_lon([1], "missing lat/lon")
    with pytest.raises(ValueError):
        GeometryUtils.coerce_lat_lon(types.SimpleNamespace(lat=1), "missing lat/lon")


def test_coerce_lat_lon_errors_on_non_numeric():
    with pytest.raises(ValueError):
        GeometryUtils.coerce_lat_lon({"lat": "x", "lon": 2}, "bad numbers")
    with pytest.raises(ValueError):
        GeometryUtils.coerce_lat_lon(["a", "b"], "bad numbers")
