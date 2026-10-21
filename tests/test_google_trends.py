import pandas as pd
from scripts.ingest_google_trends import market_pulse, read_trends_csv, rescale_to_anchor

DOWNLOAD = """Category: All categories

Week,Louis Vuitton Speedy: (United States),Coach Tabby: (United States)
2021-10-17,10,<1
2021-10-24,20,4
"""


def write(tmp_path, name: str, text: str):
    path = tmp_path / f"{name}.csv"
    path.write_text(text)
    return path


def test_parser_reads_headers_and_below_one(tmp_path) -> None:
    frame = read_trends_csv(write(tmp_path, "luxury_bags", DOWNLOAD))

    tabby = frame[frame["term"] == "Coach Tabby"].sort_values("week")
    assert set(frame["region"]) == {"United States"}
    assert tabby["interest"].tolist() == [0.5, 4.0]
    assert tabby["below_one"].tolist() == [True, False]


def test_rescaling_uses_anchor_only_when_it_is_precise(tmp_path) -> None:
    reference = read_trends_csv(write(tmp_path, "luxury_bags", DOWNLOAD))
    doubled = read_trends_csv(write(tmp_path, "aspirational_bags", DOWNLOAD.replace("10,<1", "5,<1").replace("20,4", "10,4")))
    coarse = read_trends_csv(write(tmp_path, "premium_brands", DOWNLOAD.replace("10,<1", "<1,7").replace("20,4", "<1,9")))

    scaled = rescale_to_anchor({"luxury_bags": reference, "aspirational_bags": doubled, "premium_brands": coarse})

    aspirational = scaled[scaled["source_file"] == "aspirational_bags"]
    premium = scaled[scaled["source_file"] == "premium_brands"]
    assert aspirational["scale"].iloc[0] == 2.0 and aspirational["common_scale"].all()
    assert premium["scale"].iloc[0] == 1.0 and not premium["common_scale"].any()


def test_pulse_skips_percent_change_from_near_zero_base() -> None:
    weeks = pd.date_range("2021-10-17", periods=104, freq="W")
    trends = pd.DataFrame({
        "week": list(weeks) * 2, "term": ["rising"] * 104 + ["tiny"] * 104, "region": "United States",
        "interest_common": [10.0] * 52 + [20.0] * 52 + [0.2] * 52 + [3.0] * 52,
        "below_one": False, "source_file": "f", "tier": "luxury", "common_scale": True,
    })

    pulse = market_pulse(trends).set_index("term")

    assert pulse.loc["rising", "change_pct"] == 100.0
    assert pd.isna(pulse.loc["tiny", "change_pct"])
