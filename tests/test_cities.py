"""Every city in xnet.cities.CITIES builds from its bundled map, is a usable network, and registers with av_world (a few seconds)."""
import os

from pgh import mine
from xnet import cities


def test_every_city_builds_with_more_than_100_links():
    for name, c in cities.CITIES.items():
        assert os.path.exists(os.path.join(os.path.dirname(cities.__file__), "..", "av_world", "_data", f"{name}_segments.parquet")), name
        assert c["release"] in (cities.OLD, cities.NEW) and c["name"]
        net, geo, info = cities.build(name)
        assert net["link_len"].shape[0] > 100, name
        assert int(net["node_signal"].sum()) > 0 and len(net["mov_len"]) > net["link_len"].shape[0], name
        assert info["city"] == c["name"]
        assert (mine.LAT0, mine.LON0) == (40.44, -79.99)  # Pittsburgh's grid restored


def test_city_names_are_known_to_av_world():
    from av_world import easy
    assert set(cities.CITIES) <= set(easy.cities())
