from hrba.vec_stat import *


class TestVecStat:
    def test_from_array(self):
        # generate random array
        rng = np.random.default_rng(seed=0)
        b, n = 2, 100
        x = rng.standard_normal(size=(b, n))

        # build from array
        vec_stat = VecStat.from_array(x)

        assert vec_stat.n == 100
        assert np.allclose(vec_stat.mu, x.mean(axis=1))
        assert np.allclose(vec_stat.cov, np.cov(x, ddof=0))
        assert np.allclose(vec_stat.mean_outer, x @ x.T / n)

    def test_init(self):
        # generate random array
        rng = np.random.default_rng(seed=0)
        b, n = 2, 100
        x = rng.standard_normal(size=(b, n))

        mu = x.mean(axis=1)
        cov = np.cov(x, ddof=0)
        mean_outer = x @ x.T / n

        for vec_stat in (VecStat(n=n, mu=mu, cov=cov),
                         VecStat(n=n, mu=mu, mean_outer=mean_outer)):
            assert vec_stat.n == n
            assert np.allclose(vec_stat.mu, mu)
            assert np.allclose(vec_stat.cov, cov)
            assert np.allclose(vec_stat.mean_outer, mean_outer)

    def test_union(self):
        # generate random array
        rng = np.random.default_rng(seed=0)
        b, n = 2, 100
        x = rng.standard_normal(size=(b, n))

        # build vec_stat from raw vectors
        vec_stat = VecStat.from_array(x)

        # split raw vectors into two sets, build vec stats and sum them
        vec_stat0 = VecStat.from_array(x[:, 40:])
        vec_stat1 = VecStat.from_array(x[:, :40])
        vec_stat_sum = vec_stat0 | vec_stat1

        # approaches above should yield same result
        assert vec_stat.n == vec_stat_sum.n
        assert np.allclose(vec_stat.mu, vec_stat_sum.mu)
        assert np.allclose(vec_stat.cov, vec_stat_sum.cov)
        assert np.allclose(vec_stat.mean_outer, vec_stat_sum.mean_outer)
