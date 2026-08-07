"""PNDiode2D contact validation — the failure that depended on dictionary order.

A node listed in two contacts is not a harmless duplicate. ``_contact_state``
writes psi into a dict keyed by node, so the last contact visited wins and the
answer depends on the order the contacts happen to be declared in. Measured on
the ten-node strip below, node 0 came out at -13.82 or +5.50 thermal volts — a
19-volt swing — purely by swapping two dictionary keys. Nothing raised, and the
solve converged either way.

Found in review by Codex against the published main; these tests are the guard.
"""
import numpy as np
import pytest

from tarhan.models.pn2d import PNDiode2D

CELL = 1e-5          # cm


def _strip():
    """Four square cells; doping p on the left, n on the right."""
    pts = []
    for k in range(5):
        pts += [(k * CELL, 0.0), (k * CELL, CELL)]
    tris = []
    for k in range(4):
        a, b = 2 * k, 2 * (k + 1)
        c, d = b + 1, a + 1
        tris += [(a, b, c), (a, c, d)]
    doping = np.array([-1e16, -1e16, -1e16, -1e16, 0.0, 0.0,
                       1e16, 1e16, 1e16, 1e16])
    return pts, tris, doping


def _device(contacts, biased="top"):
    pts, tris, doping = _strip()
    return PNDiode2D(points=pts, triangles=tris, net_doping=doping,
                     contacts=contacts, biased_contact=biased)


def test_a_node_in_two_contacts_is_refused():
    """The order-dependent case, named in the error rather than resolved quietly.

    Silently keeping one of the two would be worse than either choice: it makes
    the result a function of dictionary insertion order, which is not a physical
    input.
    """
    with pytest.raises(ValueError, match="disjoint"):
        _device({"top": [0, 1], "bot": [0, 1, 8, 9]})


def test_the_refusal_does_not_depend_on_declaration_order():
    """Both orderings must fail. If only one did, the guard would be the bug.

    This is the assertion that would have caught the original defect, because
    the defect WAS the asymmetry between these two.
    """
    for contacts in ({"top": [0, 1], "bot": [0, 1, 8, 9]},
                     {"bot": [0, 1, 8, 9], "top": [0, 1]}):
        with pytest.raises(ValueError, match="disjoint"):
            _device(contacts)


def test_an_empty_contact_is_refused():
    """An empty contact constrains nothing and says nothing about it.

    Quieter than the overlap and just as wrong: the device solves with one
    terminal simply absent.
    """
    with pytest.raises(ValueError, match="no nodes"):
        _device({"top": [], "bot": [8, 9]})


def test_a_contact_node_outside_the_mesh_is_refused():
    with pytest.raises(ValueError, match="outside 0"):
        _device({"top": [0, 1], "bot": [8, 99]})


def test_disjoint_contacts_are_accepted():
    """The guard must not reject the ordinary case it exists to protect."""
    dev = _device({"top": [0, 1], "bot": [8, 9]})
    assert dev.mesh.n_nodes == 10
    psi_bc, n_bc, p_bc = dev._contact_state(0.5)
    assert set(psi_bc) == {0, 1, 8, 9}
    assert set(n_bc) == set(p_bc) == set(psi_bc)
