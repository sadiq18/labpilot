import pytest

from labpilot.accessor.kaggle.urls import parse_kernel_ref


def test_parse_kernel_ref_accepts_code_prefix():
    owner, slug = parse_kernel_ref("code/sadiq18/aerial-cactus-labpilot-baseline")
    assert owner == "sadiq18"
    assert slug == "aerial-cactus-labpilot-baseline"


def test_parse_kernel_ref_rejects_invalid_ref():
    with pytest.raises(ValueError, match="owner/slug"):
        parse_kernel_ref("only-one-part")
