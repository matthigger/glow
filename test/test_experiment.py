from itertools import product

import hrba
from hrba.experiment import *

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
