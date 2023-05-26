from hrba.extent import *


class TestExtenterSphere:
    def test_call(self):
        mask_idx = np.arange(64).reshape((8, 8))
        mask_idx[:2, :] = -1

        mask_expect_tup = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [1, 0, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 0, 0, 0, 0]]), \
            np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1, 1, 0, 0],
                      [1, 1, 1, 1, 1, 1, 1, 0],
                      [1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1]])

        for radius, mask_expect in zip((3, 10), mask_expect_tup):
            extenter = ExtenterSphere(radius=radius)
            mask = extenter(mask_idx=mask_idx, seed=0)

            assert np.allclose(mask, mask_expect)
