# e7-prompt

The prompt contract shared by the E7 training pipeline and the Jetson inference
stack. Pure Python, zero dependencies.

**Do not copy these files onto the Jetson.** Install the package on both sides
and pin the version. A copy drifts, and the drift is invisible: the robot keeps
moving, it just moves to the wrong shelf, and the regression gets blamed on the
policy.

```bash
pip install /path/to/openpi/examples/e7/e7_prompt
```

## Rendering

One function turns enum fields into the prompt string. Both sides call it.

```python
from e7_prompt import ContextSpec, render_prompt

spec = ContextSpec(
    category="humanities",
    target="right",
    rule_version="v03",
    rule_table={"engineering": "center", "humanities": "right", "other": "left"},
    prompt_style="rule_table",
)
render_prompt(spec)
# 'category=humanities. rules: engineering->center, humanities->right, other->left.
#  insert the humanities book into the correct shelf'
```

`rule_table` is the **full mapping**, not a resolved destination. Handing the
renderer only the answer means the rule was applied outside the policy — which
is precisely what the experiment is trying to measure.

Rows are emitted sorted by category name, never with the episode's own category
first, so "read row 1" is not a shortcut.

## Validation

The converter reports every violation at once; the Jetson client aborts on any.

```python
from e7_prompt import validate_all

validate_all({"v01": {"engineering": "left", "humanities": "left"}})
# {'v01': ['INCOMPLETE_RULE_TABLE: missing [\'other\']',
#          "NOT_INJECTIVE: destination(s) ['left'] used twice"]}
```

- `INCOMPLETE_RULE_TABLE` — a category has no row.
- `NOT_INJECTIVE` — two categories share a destination, so the destination no
  longer identifies the category and a counterfactual between them changes
  nothing.
- `MULTISET_MISMATCH` — two `rule_version`s use different destination sets, so
  token presence alone leaks which version is active.

## Contract hash

```python
from e7_prompt import TokenizerSpec, build_manifest

build_manifest(
    prompt_style="rule_table",
    rule_tables=tables,
    tokenizer=TokenizerSpec(max_token_len=200, discrete_state_input=False),
    image_keys=("base_0_rgb", "left_wrist_0_rgb"),
    action_dim=32,
    action_horizon=16,
)
```

Ships next to the checkpoint. Jetson recomputes the hash at startup and refuses
to serve on a mismatch.

The hash covers the templates themselves, the category map, the rule tables, the
prompt style, and the tokenizer settings — including **`discrete_state_input`**.
That flag defaults to `False`, and flipping it makes the tokenizer wrap the text
as `"Task: … , State: <7 discretised values>;\nAction: "`, roughly 35 extra
tokens, with no error and no visible symptom. It is the exact drift this hash
exists to catch.

Editing a template changes the hash and invalidates earlier checkpoints. That is
intended.
