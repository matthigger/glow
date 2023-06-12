from hrba.experiment import *
from hrba.graph import get_miss_hits
from hrba.sample_effect import ExtenterMinVar
from .make_test_image import folder_test_data


class TestAnalysisHRBA:
    def test_cluster(self):
        # in a population of identical test images, clustering should
        # segment based on color

        # load single image, bootstrap a few more (no noise), sample rand x
        exp = Experiment.from_search(folder=folder_test_data,
                                     sbj_regex='sbj\d',
                                     img_glob_dict={'color': '*test.png'})
        exp.bootstrap_img(n=10, noise_scale=0)
        exp.sample_x(a=2)

        # cluster (should collect all areas of consistent color)
        ana_hrba = AnalysisHRBA(exp=exp, n_permute=0)
        ana_hrba.cluster()

        # assumptions: test image has 1 color per greyscale value and each
        # color is contiguous
        y_sbj0_grey = np.linalg.norm(exp.y[:, 0, :], axis=0)
        img_grey = np.zeros(exp.mask_idx.shape)
        img_grey[exp.mask_idx >= 0] = y_sbj0_grey
        for grey_val in set(y_sbj0_grey):
            mask = img_grey == grey_val
            dendro = ana_hrba.perm_dendro_dict[0]
            miss_hit_dict = get_miss_hits(mask=mask, mask_idx=exp.mask_idx,
                                          dendro=dendro)
            perfect_miss_hit = (0, mask.sum())
            for miss_hit in miss_hit_dict.values():
                if perfect_miss_hit == tuple(miss_hit):
                    break
            else:
                raise AssertionError('some color not segmented perfectly')
