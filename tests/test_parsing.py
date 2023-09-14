from pathlib import Path
import pytest
from pybinsim.parsing import parse_boolean, parse_soundfile_list


def test_parse_boolean_regular():
    inputs = [True, False, "True", "FALSE", None, "Something Strange", 12, 0]
    expected_outputs = [True, False, True, False, None, None, True, False]

    for i, test_value in enumerate(inputs):
        output = parse_boolean(test_value)
        assert output == expected_outputs[i]


def test_parse_soundfile_list():
    assert parse_soundfile_list("a/b/c#d") == list((Path("a/b/c"), Path("d")))

    with pytest.raises(Exception):
        parse_soundfile_list(42)
