# OMMX adapter for Fixstars Amplify

[![PyPI - Version](https://img.shields.io/pypi/v/ommx-fixstars-amplify-adapter)](https://pypi.org/project/ommx-fixstars-amplify-adapter/) 
[![main](https://img.shields.io/badge/API_Reference-blue)](https://jij-inc.github.io/ommx-fixstars-amplify-adapter/)


This package provides an adapter for the [Fixstars Amplify SDK](https://amplify.fixstars.com/ja/docs/amplify/v1/index.html) from/to [OMMX](https://github.com/Jij-Inc/ommx).

## How to install

The `ommx-fixstars-amplify-adapter` can be installed from PyPI as follows:

```bash
pip install ommx-fixstars-amplify-adapter
```

## Solve problems formulated in OMMX with Fixstars Amplify

The `ommx-fixstars-amplify-adapter` allows problems formulated in OMMX to be solved in Fixstars Amplify.

```mermaid
sequenceDiagram
    participant U as User
    participant A as Adapter
    participant P as Fixstars Amplify
    U->>A: ommx.Instance
    A->>U: amplify.Model
    U->>P: amplify.Model and parameters for solvers;
    P->>P: Solve with Fixstars Amplify
    P->>U: amplify.Result
    U->>A: amplify.Result and variable_map
    A->>U: ommx:State
```

For example, the following problem formulated in OMMX can be solved using Fixstars Amplify.

```python
from ommx import Instance, DecisionVariable
from ommx_fixstars_amplify_adapter import OMMXFixstarsAmplifyAdapter

q_0 = DecisionVariable.binary(id=0, name="q_0")
q_1 = DecisionVariable.binary(id=1, name="q_1")

ommx_instance = Instance.from_components(
    decision_variables=[q_0, q_1],
    objective=q_0 * q_1 + q_0 - q_1 + 1,
    constraints={0: q_0 + q_1 == 1},
    sense=Instance.MAXIMIZE,
)

token = "***FIXSTARS AMPLIFY TOKEN***"
solution = OMMXFixstarsAmplifyAdapter.solve(ommx_instance, amplify_token=token)
print(solution.decision_variables_df())
```

### Preparing an Instance

The `solve()` API does not modify the input instance. It prepares an isolated
copy with the adapter's recommended policy before solving. Fixstars Amplify
accepts OneHot constraints directly, so the recommended policy preserves
OneHot constraints and lowers only Indicator and SOS1 constraints to regular
constraints.

When an application needs a custom preparation policy, prepare the instance
explicitly and use the preparation-free API:

```python
input_class = OMMXFixstarsAmplifyAdapter.INPUT_CLASS
policy = OMMXFixstarsAmplifyAdapter.recommended_preparation_policy()
# Edit the policy here if needed.

ommx_instance.prepare(input_class, policy)
solution = OMMXFixstarsAmplifyAdapter.solve_without_preparation(
    ommx_instance,
    amplify_token=token,
)
```

`solve_without_preparation()` is strict: it does not prepare or otherwise
modify the input and rejects instances outside the adapter's `INPUT_CLASS`.

## Solve problems formulated in Fixstars Amplify SDK with other solvers

The `ommx-fixstars-amplify-adapter` allows problems formulated in Fixstars Amplify SDK to be solved in other solvers.

```mermaid
sequenceDiagram
    participant U as User
    participant A as Adapter
    participant O as Other OMMX toolchain
    U->>A: amplify.Model
    A->>U: ommx.Instance
    U->>O: ommx.Instance and parameters for other solver
    O->>O: Solve the instance with other solver using other adapter
    O->>U: ommx.State
```

For example, the following mixed integer programming problem formulated in Fixstars Amplify SDK can be solved using PythonMIP.

```python
import amplify
from ommx_fixstars_amplify_adapter import model_to_instance
from ommx_python_mip_adapter import OMMXPythonMIPAdapter

UPPER = float("inf")
LOWER = 0.0

gen = amplify.VariableGenerator()
x = gen.scalar("Integer", bounds=(LOWER, UPPER), name="x")
y = gen.scalar("Real", bounds=(LOWER, UPPER), name="y")

model = amplify.Model()
model += -10 * x - y
model += amplify.less_equal(x, 1)
model += amplify.less_equal(20 * x + y, 100)

ommx_instance = model_to_instance(model)
solution = OMMXPythonMIPAdapter.solve(ommx_instance)
print(solution.decision_variables_df())
```

> [!NOTE]
> Currently, the `model_to_instance` function does not support Ising variables. Therefore, if your model contains Ising variables, you must convert them to binary variables. The following function can be used for this conversion.
> ```python
> import typing
> import amplify
> 
> def ising_to_binary(
>     model: amplify.Model
> ) -> typing.Tuple[amplify.Model, amplify.Result.ModelConversion.IntermediateMapping]:
>     ising_to_binary_settings = {
>         amplify.VariableType.Binary: amplify.Degree.HighOrder,
>         amplify.VariableType.Ising: amplify.Degree.Zero,
>         amplify.VariableType.Integer: amplify.Degree.HighOrder,
>         amplify.VariableType.Real: amplify.Degree.HighOrder,
>     }
>     return model.to_intermediate_model(
>         amplify.AcceptableDegrees(
>             objective=ising_to_binary_settings,  # type: ignore
>             equality_constraints=ising_to_binary_settings,  # type: ignore
>             inequality_constraints=ising_to_binary_settings,  # type: ignore
>         )
>     )
> ```

## For Developer

The packages required for development can be installed as follows:

```bash
uv sync --all-extras
```

Use the following commands to test, lint and format.

```bash
uv run pytest
uv run ruff check
uv run ruff format
```
