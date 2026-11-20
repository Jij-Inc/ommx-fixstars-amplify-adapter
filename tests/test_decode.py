import copy
from unittest.mock import Mock, patch

import amplify
import pytest
from ommx import Instance, State
from ommx.adapter import NoSolutionReturned

from ommx_fixstars_amplify_adapter import OMMXFixstarsAmplifyAdapter


def test_decode_empty_result_with_active_variables():
    instance = Instance.minimize()
    instance.objective = instance.new_binary("x")
    adapter = OMMXFixstarsAmplifyAdapter(instance)
    result = amplify.solve(
        adapter.solver_input, amplify.AmplifyAEClient(), dry_run=True
    )
    assert len(result) == 0

    with pytest.raises(NoSolutionReturned):
        adapter.decode(result)


def test_decode_constant_objective_without_reading_a_sample():
    instance = Instance.minimize()
    instance.objective = 7
    adapter = OMMXFixstarsAmplifyAdapter(instance)

    solution = adapter.decode(Mock(spec_set=[]))

    assert solution.state.entries == {}
    assert solution.objective == 7


def test_decode_uses_the_copy_supplied_by_the_caller():
    instance = Instance.maximize()
    x = instance.new_binary("x")
    instance.objective = x
    supplied = copy.copy(instance)
    adapter = OMMXFixstarsAmplifyAdapter(supplied)
    assert adapter.instance is supplied

    state = State(entries={x.id: 1})
    instance.objective = 1 - x
    instance.add_constraint(x <= 0)

    with patch.object(adapter, "decode_to_state", return_value=state):
        solution = adapter.decode(Mock(spec=amplify.Result))

    assert solution.state.entries == state.entries
    assert solution.objective == 1
    assert solution.feasible
