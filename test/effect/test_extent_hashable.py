"""hashability for the Extenter classes."""

import pytest

from glow.effect import ExtenterMinVar, ExtenterSphere


class TestExtenterSphereHash:
    def test_equal_same_params(self):
        a = ExtenterSphere(radius=2)
        b = ExtenterSphere(radius=2)
        assert a == b
        assert hash(a) == hash(b)

    def test_int_cast_equality(self):
        # constructor casts ints, so float-ish inputs that round to the
        # same int should still compare equal
        assert ExtenterSphere(radius=2) == ExtenterSphere(radius=2.0)
        assert (hash(ExtenterSphere(radius=2))
                == hash(ExtenterSphere(radius=2.0)))

    @pytest.mark.parametrize('a, b', [
        (ExtenterSphere(radius=2), ExtenterSphere(radius=3)),
        (ExtenterSphere(n_vox=10), ExtenterSphere(n_vox=11)),
        (ExtenterSphere(radius=2), ExtenterSphere(n_vox=10)),
        (ExtenterSphere(n_vox=10),
         ExtenterSphere(n_vox=10, connected=True)),
    ])
    def test_each_attribute_changes_hash(self, a, b):
        assert a != b
        assert hash(a) != hash(b)

    def test_dict_key(self):
        a = ExtenterSphere(radius=2)
        d = {a: 'sphere'}
        assert d[ExtenterSphere(radius=2)] == 'sphere'

    def test_set_member(self):
        s = {ExtenterSphere(radius=2), ExtenterSphere(radius=2)}
        assert len(s) == 1


class TestExtenterMinVarHash:
    def test_equal_same_params(self):
        a = ExtenterMinVar(n_vox=10)
        b = ExtenterMinVar(n_vox=10)
        assert a == b
        assert hash(a) == hash(b)

    def test_int_cast_equality(self):
        assert ExtenterMinVar(n_vox=10) == ExtenterMinVar(n_vox=10.0)
        assert (hash(ExtenterMinVar(n_vox=10))
                == hash(ExtenterMinVar(n_vox=10.0)))

    def test_n_vox_change(self):
        a = ExtenterMinVar(n_vox=10)
        b = ExtenterMinVar(n_vox=11)
        assert a != b
        assert hash(a) != hash(b)

    def test_dict_key(self):
        a = ExtenterMinVar(n_vox=10)
        d = {a: 'minvar'}
        assert d[ExtenterMinVar(n_vox=10)] == 'minvar'


class TestCrossClass:
    def test_sphere_minvar_distinct_when_n_vox_matches(self):
        sphere = ExtenterSphere(n_vox=10)
        minvar = ExtenterMinVar(n_vox=10)
        assert sphere != minvar
        assert hash(sphere) != hash(minvar)


class TestNotEqualToOtherTypes:
    def test_sphere_neq_dict(self):
        # __eq__ returns False (not NotImplemented) for foreign types
        assert ExtenterSphere(radius=2) != {'radius': 2}

    def test_minvar_neq_int(self):
        assert ExtenterMinVar(n_vox=10) != 10
