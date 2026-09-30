import json

from bidsgate.segcard.core import build, render


def test_card(tmp_path):
    sc = {"dice": 0.8, "sensitivity": 0.75, "lesions": 4, "detected": 3, "false_positive_components": 2,
          "by_size": {"0-100mm3": {"n": 2, "sensitivity": 0.5}}, "by_height": {"lower": {"n": 4, "sensitivity": 0.75}}}
    (tmp_path / "b.json").write_text(json.dumps({"pipeline": "X", "results": [{"subject": "s", "score": sc}]}))
    (tmp_path / "r.json").write_text(json.dumps({"pairs": 3, "invented_new_per_pair": 2.0, "invented_resolved_per_pair": 1.0, "mean_dice": 0.9}))
    c = build("X", tmp_path / "b.json", [tmp_path / "r.json"])
    assert c["evidence"]["injection"]["sensitivity"] == 0.75 and c["missing"] == ["agreement with expert masks"]
    h = render(c)
    assert "3 of 4" in h and "2.0 per pair" in h
