import pytest

from reachy_demo.robot.motion import AXES, LIMITS, NEUTRAL, dance

pytest.importorskip("reachy_mini_dances_library")  # the optional `robot` extra

from reachy_demo.robot.dances import load_dances  # noqa: E402


def test_loads_every_library_move() -> None:
    dances = load_dances()
    assert len(dances) >= 20
    assert {"simple_nod", "side_to_side_sway", "dizzy_spin"} <= set(dances)
    assert dances["simple_nod"].pass_s == pytest.approx(4 * 60 / 114)
    assert dances["simple_nod"].description


def test_library_moves_play_inside_the_safety_limits_unclipped() -> None:
    for name, d in load_dances().items():
        move = dance(name, d.pass_s, d.offset)
        for i in range(201):
            pose = move.pose_at(i * move.duration_s / 200, NEUTRAL)
            assert set(pose) == set(AXES)
            for k in AXES:
                assert abs(pose[k]) <= LIMITS[k] + 1e-6, (name, k, pose[k])


def test_offsets_use_mm_and_degrees() -> None:
    sway = load_dances()["side_to_side_sway"]
    peak = max(abs(sway.offset(i * sway.pass_s / 100)["y"]) for i in range(101))
    assert peak == pytest.approx(40, abs=1)  # library amplitude 0.04 m
    nod = load_dances()["simple_nod"]
    assert max(abs(nod.offset(i * nod.pass_s / 100)["pitch"]) for i in range(101)) == pytest.approx(
        20, abs=1
    )
