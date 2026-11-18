from couture_atelier.advisor_notes import check_note, facts_key, published_note, template_note

FACTS = {"tier": "core", "weeks_since_last_visit": 9, "spend_last_12m_eur": 14200.0,
         "favourite_category": "long dress", "champion_action": "contact", "challenger_action": "no action"}
CATEGORIES = ("long dress", "culottes", "medium coat")


def test_template_always_passes_its_own_check() -> None:
    assert check_note(template_note(FACTS), FACTS, CATEGORIES) == []
    quiet = {**FACTS, "champion_action": "no action"}
    assert check_note(template_note(quiet), quiet, CATEGORIES) == []


def test_invented_numbers_and_discount_language_are_rejected() -> None:
    assert check_note("She spent 18,000 last year; invite her to the private sale.", FACTS)
    assert check_note("9 weeks since her visit; 20% off the new coats.", FACTS)


def test_wrong_tier_or_category_is_rejected() -> None:
    assert check_note("A VIC who loves long dress pieces.", FACTS)
    assert check_note("Show her the new culottes.", FACTS, CATEGORIES)
    assert check_note("Nine weeks on; show her the new long dress pieces.", FACTS, CATEGORIES) == []


def test_failed_drafts_fall_back_to_the_template() -> None:
    cache = {facts_key(FACTS): "Offer 30% off culottes."}
    note, source = published_note(FACTS, cache, CATEGORIES)
    assert source == "template" and note == template_note(FACTS)
    cache = {facts_key(FACTS): "Nine weeks since her last visit; share the new long dress pieces first."}
    assert published_note(FACTS, cache, CATEGORIES)[1] == "model draft, checked"


def test_cache_key_survives_a_csv_round_trip() -> None:
    import numpy as np

    from couture_atelier.advisor_notes import facts_of

    as_built = {**FACTS, "weeks_since_last_visit": np.int64(9), "spend_last_12m_eur": np.float64(14200.0)}
    as_read = {**FACTS, "weeks_since_last_visit": 9, "spend_last_12m_eur": 14200}
    assert facts_key(facts_of(as_built)) == facts_key(facts_of(as_read))
