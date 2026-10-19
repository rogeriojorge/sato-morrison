"""Independent action derivative and conditional-mean checks of the control."""
import ast
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import roots_legendre


@lru_cache(None)
def control_namespace():
    source = Path(__file__).resolve().parents[1]/'examples/43_encounter_born_control.py'
    tree = ast.parse(source.read_text())
    constants = {'MASS', 'CHARGE', 'FIELD', 'STRENGTH', 'SCREENING', 'THETA',
                 'COEFFICIENT', 'MIN_GAP', 'CONTROL_DISTANCE', 'ORDERS', 'BATCH',
                 'OMEGA', 'MU_REF'}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {item.id for target in node.targets for item in ast.walk(target)
                     if isinstance(item, ast.Name)}
            if names and names <= constants:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == 'born_control':
            selected.append(node)
    namespace = {'np': np, 'roots_legendre': roots_legendre}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def test_born_control_matches_independent_potential_action_derivative():
    ns = control_namespace()
    nodes = np.array([[1.9, 1.1, 1.3, .2], [2.2, 1.5, 1.8, 2.4],
                      [1.6, .8, 1.4, 5.4]])
    expected = []
    for b, u, p, phase in nodes:
        def action(angle):
            def potential(z):
                psi = angle-ns['OMEGA']*z/u
                position = np.array([b-p*np.sin(psi)/ns['OMEGA'],
                                     p*np.cos(psi)/ns['OMEGA'], z])
                radius = np.linalg.norm(position)
                return ns['STRENGTH']*np.exp(-radius/ns['SCREENING'])/radius/u
            return quad(potential, -ns['CONTROL_DISTANCE'], ns['CONTROL_DISTANCE'],
                        epsabs=1e-12, epsrel=1e-11, limit=400)[0]
        step = 2e-5
        derivative = (action(phase+step)-action(phase-step))/(2*step)
        expected.append(ns['OMEGA']/(2*ns['FIELD'])*derivative)
    for order in ns['ORDERS']:
        np.testing.assert_allclose(ns['born_control'](nodes, order), expected,
                                   atol=2e-11, rtol=2e-7)


def test_born_control_is_zero_in_each_phase_cell_and_respects_gap_mask():
    ns = control_namespace()
    x, weights = roots_legendre(24)
    for lower, upper in ((np.pi/2-.25, np.pi/2+.25),
                         (np.pi/2+.25, np.pi/2-.25+2*np.pi)):
        phase = (lower+(upper-lower)*(x+1)/2) % (2*np.pi)
        nodes = np.column_stack((np.full(len(x), 1.8), np.full(len(x), .9),
                                 np.full(len(x), 1.2), phase))
        reflected = nodes.copy()
        reflected[:, 3] = (np.pi-phase) % (2*np.pi)
        for order in ns['ORDERS']:
            values = ns['born_control'](nodes, order)
            opposite = ns['born_control'](reflected, order)
            np.testing.assert_allclose(values+opposite, 0., atol=2e-13, rtol=0)
            assert abs(weights@values/2) < 2e-13
    excluded = np.array([[1.2, .4, 2.4, .1], [1.3, 1.2, 2.2, 2.7]])
    for order in ns['ORDERS']:
        np.testing.assert_array_equal(ns['born_control'](excluded, order), np.zeros(2))
