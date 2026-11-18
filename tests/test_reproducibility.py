"""The client frame must be identical across runs, or tree models drift between rebuilds."""

import duckdb
import pandas as pd
import pytest

from couture_atelier.client_frames import window_frame
from couture_atelier.paths import WAREHOUSE


@pytest.mark.skipif(not WAREHOUSE.exists(), reason="needs the local warehouse (raw data is not published)")
def test_client_frame_is_bit_identical_across_builds() -> None:
    with duckdb.connect(str(WAREHOUSE), read_only=True) as con:
        first, second = window_frame(con, "2017-06-30"), window_frame(con, "2017-06-30")
    pd.testing.assert_frame_equal(first, second, check_exact=True)
