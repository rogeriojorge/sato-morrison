"""Local-action near-uniform quadrature regression; no evolution claim."""
import ast
from types import SimpleNamespace
from pathlib import Path
import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'examples'/'18_near_uniform.py'
constants={'POSITION','MASS','CHARGE','STRENGTH','D','UMAX','MUMAX','NMU',
    'ORDERS','AMPLITUDES'}
body=[]
for node in ast.parse(SCRIPT.read_text()).body:
    if isinstance(node,(ast.Import,ast.ImportFrom,ast.FunctionDef)):
        body.append(node)
    elif isinstance(node,ast.Assign):
        targets=[name for target in node.targets for name in
            (target.elts if isinstance(target,ast.Tuple) else [target])]
        if all(isinstance(name,ast.Name) and name.id in constants for name in targets):
            body.append(node)
namespace={}
exec(compile(ast.Module(body=body,type_ignores=[]),str(SCRIPT),'exec'),namespace)
namespace['jax'].config.update('jax_enable_x64',True)
example=SimpleNamespace(**namespace)


@pytest.mark.parametrize('nu',[8,16])
def test_positive_amplitude_projector_limit_differs_only_on_equal_u_pairs(nu):
    forms=example.analytic_forms(nu);direct=example.direct_form(nu,1e-7)
    assert direct['ad_relative_error']<1e-12
    assert abs(direct['gram']/forms['positive_amplitude_limit']-1)<2e-6
    assert abs(direct['equal_u_gram']/forms['equal_u_limit']-1)<2e-6
    uniform_unequal=forms['uniform']-forms['equal_u_uniform']
    assert abs(direct['unequal_u_gram']/uniform_unequal-1)<2e-6
    assert forms['gap']>0
    assert abs(forms['equal_u_uniform']-forms['equal_u_limit']-forms['gap'])<1e-13


def test_positive_quadrature_gap_bound_and_parallel_refinement():
    rows=[example.analytic_forms(nu) for nu in (8,16,32,64,128)]
    assert all(0<r['gap']<=r['gap_bound'] for r in rows)
    assert all(b['relative_gap']<a['relative_gap'] for a,b in zip(rows,rows[1:]))
    assert rows[-1]['relative_gap']<.025
    assert rows[-1]['relative_bound']<rows[0]['relative_bound']/10


@pytest.mark.parametrize('amplitude',[0.,-1e-6])
def test_singular_point_is_not_used_to_choose_a_projector(amplitude):
    with pytest.raises(ValueError,match='strictly positive'):
        example.direct_form(8,amplitude)
