"""A reflected replica must enter the audit even if every primary transmits."""
import ast
from pathlib import Path
import numpy as np


def test_cartesian_replay_selection_covers_reflected_replica_without_duplicate_nodes():
    source = Path(__file__).resolve().parents[1]/'examples/40_refinement_audit.py'
    function = next(n for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name=='select_cartesian_replays')
    namespace = {'np':np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    nodes = np.array([[1.2, .7, .8, .1], [1.3, .8, .9, .2],
        [1.4, .9, 1., .3], [1.2, .7, .8, .1], [1.5, 1., 1.1, .4]])
    states = {'nodes':nodes, 'status':np.array(['passed']*4+['unresolved']),
        'exit':np.array(['transmitted', 'transmitted', 'reflected', 'transmitted', 'missing']),
        'min_separation':np.array([.1, .2, .15, .09, .01])}
    values = np.zeros((5, 4)); values[:, 1] = [1., 2., 3., 1., 10.]
    # State 3 is a tighter copy of primary state 0; state 2 belongs to a replica.
    selected = namespace['select_cartesian_replays'](states, [0, 1, 3], values)
    assert 2 in selected
    assert 4 not in selected
    assert len(selected)==3
    assert len({tuple(nodes[i]) for i in selected})==3
