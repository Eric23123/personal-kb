from __future__ import annotations

from scripts.ops.ingestion_readback import (
    select_readable_source_leaf,
    source_readback_matches,
)


def test_readback_skips_generated_overview_and_abstract_leaves():
    nodes = [
        {"isDir": False, "uri": "viking://x/source/.overview.md", "name": ".overview.md"},
        {"isDir": False, "uri": "viking://x/source/.abstract.md", "name": ".abstract.md"},
        {"isDir": False, "uri": "viking://x/source/content.md", "name": "content.md"},
    ]

    assert select_readable_source_leaf(nodes) == "viking://x/source/content.md"


def test_readback_returns_none_when_only_generated_leaves_exist():
    nodes = [
        {"isDir": False, "uri": "viking://x/source/.overview.md", "name": ".overview.md"},
        {"isDir": False, "uri": "viking://x/source/.abstract.md", "name": ".abstract.md"},
    ]

    assert select_readable_source_leaf(nodes) is None


def test_readback_accepts_json_escape_and_line_ending_normalization():
    expected = '{\n  "source_path": "C:\\course_material\\SE_424_HW1_Solution.pdf",\n  "pages": 7\n}'
    actual = '{\r\n  "source_path": "C:\\\\course_material\\\\SE_424_HW1_Solution.pdf",\r\n  "pages": 7\r\n}'

    assert source_readback_matches(expected, actual)


def test_readback_rejects_unrelated_content():
    assert not source_readback_matches(
        "inverted pendulum Jacobian state-space linearization",
        "unrelated supply-chain inventory content",
    )
