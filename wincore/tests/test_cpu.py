from WinCore.cpu import recommended_threads


def test_small_machine_reserves_one():
    plan = recommended_threads(total=4)
    assert plan.total_logical == 4
    assert plan.reserved == 1
    assert plan.recommended == 3


def test_six_thread_reserves_one():
    plan = recommended_threads(total=6)
    assert plan.reserved == 1
    assert plan.recommended == 5


def test_twelve_thread_reserves_two():
    plan = recommended_threads(total=12)
    assert plan.reserved == 2
    assert plan.recommended == 10


def test_sixteen_thread_reserves_up_to_three():
    plan = recommended_threads(total=16)
    assert plan.reserved == 3
    assert plan.recommended == 13


def test_explicit_threads_overrides_heuristic():
    plan = recommended_threads(total=16, threads=8)
    assert plan.recommended == 8
    assert plan.reserved == 8


def test_explicit_reserve_overrides_heuristic():
    plan = recommended_threads(total=8, reserve=4)
    assert plan.recommended == 4
    assert plan.reserved == 4


def test_never_recommends_zero():
    plan = recommended_threads(total=1)
    assert plan.recommended == 1
