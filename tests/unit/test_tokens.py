from reachy_demo.ops.tokens import est_tokens


def test_est_tokens_rounds_up_chars_over_4() -> None:
    assert est_tokens("") == 0
    assert est_tokens("abcd") == 1
    assert est_tokens("abcde") == 2
