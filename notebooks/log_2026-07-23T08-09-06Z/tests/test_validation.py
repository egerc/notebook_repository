from collections.abc import Sequence

import pytest

from log_2026_07_23t08_09_06z.types import result_is_ok, unwrap_result
from log_2026_07_23t08_09_06z.utils import validate_tokens


@pytest.mark.parametrize(
    "tokens_1, tokens_2, validation",
    [
        # Positive cases
        ([1, 2], [1, 2, 3], True),
        (["A", "B"], ["A", "B", "C"], True),
        # Duplicate Tokens check
        ([1, 1], [1, 2, 3], False),
        ([1, 1], [1, 2, 3, 3], False),
        # Token length check
        ([1, 2, 3], [1, 2, 3], True),
        ([1, 2, 3, 4], [1, 2, 3], True),
        # Validation failure cases
        ([1, 2], [1, 2, 1], False),
        
    ],
)
def test_token_validation[A](
    tokens_1: Sequence[A], tokens_2: Sequence[A], validation: bool
) -> None:
    if (
        validation_passed := result_is_ok(validate_tokens(tokens_1, tokens_2))
        == validation
    ):
        pass
    elif validation_passed:
        raise AssertionError(
            f"Expected {validation}, got {validation_passed} on input {tokens_1} vs {tokens_2}"
        )
    else:
        unwrap_result(validate_tokens(tokens_1, tokens_2))
