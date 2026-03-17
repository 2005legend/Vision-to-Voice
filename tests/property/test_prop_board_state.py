# Feature: intelliagent-board-reader, Property 12: Board_State serialisation round-trip

from hypothesis import given, settings, strategies as st

from board_reader.models import BoardState, BoardStep


@given(
    topic=st.text(),
    steps=st.lists(st.builds(BoardStep, id=st.integers(), text=st.text())),
    equations=st.lists(st.text()),
)
@settings(max_examples=100)
def test_board_state_round_trip(topic, steps, equations):
    """Validates: Requirements 10.1, 10.2, 10.3"""
    state = BoardState(topic=topic, board_steps=steps, equations=equations)
    assert BoardState.from_json(state.to_json()) == state
