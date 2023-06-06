from itertools import product

import hrba
from helper import generate_dummy_data
from hrba.experiment.experiment import *
from hrba.sample_effect import ExtenterSphere

data_folder = pathlib.Path(hrba.__file__).parents[2] / 'data'

# build some dummy data to load
folder = pathlib.Path('.') / 'data'
folder.mkdir(exist_ok=True)

shape = (10, 11)
affine = np.eye(4)
img_feat_intensity = defaultdict(dict)
for intensity, (img_idx, feat_idx) in enumerate(product(range(3),
                                                        range(2))):
    # add 1 to intensity (zero values are considered background)
    intensity += 1
    file = folder / f'img{img_idx}_feat{feat_idx}.nii.gz'
    x = np.full((shape), fill_value=intensity).astype(float)
    if not file.exists():
        img = nib.Nifti2Image(dataobj=x, affine=affine)
        img.to_filename(file)

    file = folder / f'img{img_idx}_feat{feat_idx}.jpg'
    if not file.exists():
        Image.fromarray(x.astype(np.uint8)).save(file)

    img_feat_intensity[img_idx][feat_idx] = intensity


def get_experiment(**kwargs):
    reg_size = 1000
    mask_idx = np.arange(reg_size).reshape((10, 10, 10))

    x, y, contrast = generate_dummy_data(reg_size=reg_size, **kwargs)
    return Experiment(x=x, y=y, contrast=contrast, mask_idx=mask_idx)


class TestExperiment:
    def test_from_search(self):
        # nii
        exp0 = Experiment.from_search(folder=folder,
                                      sbj_regex='img\d',
                                      img_glob_dict={'feat0': '*feat0.nii.gz',
                                                     'feat1': '*feat1.nii.gz'})
        # jpg
        exp1 = Experiment.from_search(folder=folder,
                                      sbj_regex='img\d',
                                      img_glob_dict={'feat0': '*feat0.jpg',
                                                     'feat1': '*feat1.jpg'})

        for exp in (exp0, exp1):
            # ensure values arrived at their proper place in y
            for img_idx, feat_intense in img_feat_intensity.items():
                for feat_idx, intense in feat_intense.items():
                    assert (exp.y[feat_idx, img_idx, :] == intense).all()

    def test_impose_effect(self):
        seed = 0
        exp = get_experiment(seed=seed)
        extenter = ExtenterSphere(radius=3)

        for p_val in np.logspace(-3, -.0001, 4):
            _exp, effect = exp.impose_effect(seed=seed, extenter=extenter,
                                             p_val=p_val)

            assert np.isclose(effect.p_val, p_val)

    def test_sample_x(self):
        seed = 0
        exp = get_experiment(seed=seed)
        exp.sample_x(a=exp.x.shape[0])
        exp.sample_x(contrast=exp.contrast)
        exp.sample_x(a=4)