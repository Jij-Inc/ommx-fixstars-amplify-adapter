from unittest.mock import Mock, patch

import amplify
from ommx import Instance, State

from ommx_fixstars_amplify_adapter import OMMXFixstarsAmplifyAdapter


def test_decode_evaluates_the_instance_used_to_build_the_model():
    instance = Instance.maximize()
    x = instance.new_binary("x")
    instance.objective = x
    adapter = OMMXFixstarsAmplifyAdapter(instance)

    state = State(entries={x.id: 1})
    instance.objective = 1 - x
    instance.add_constraint(x <= 0)

    with patch.object(adapter, "decode_to_state", return_value=state):
        solution = adapter.decode(Mock(spec=amplify.Result))

    assert solution.state.entries == state.entries
    assert solution.objective == 1
    assert solution.feasible
