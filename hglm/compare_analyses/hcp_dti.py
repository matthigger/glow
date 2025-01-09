import pathlib
from tqdm import tqdm

import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.dti import TensorModel, fractional_anisotropy, \
    mean_diffusivity

dti_fnc_dict = {'fa': fractional_anisotropy,
                'md': mean_diffusivity}

folder = pathlib.Path('/home/matt/Dropbox/pnl_hglm/data/HCP_100unrelated')
for file in tqdm(folder.glob('**/data.nii.gz'), desc='DTI per img'):

    _folder = file.parent
    for label in dti_fnc_dict.keys():
        _file = _folder / f'{label}.nii.gz'
        if not _file.exists():
            # some output file needed, process all
            break
    else:
        # all outputs already made, skip this one
        print(f'already processed, skipping: {file}')
        continue

    # load
    dwi_img = nib.load(str(file))
    dwi_data = dwi_img.get_fdata()
    bvals, bvecs = read_bvals_bvecs(str(_folder / 'bvals'),
                                    str(_folder / 'bvecs'))

    # compute DTI model
    gtab = gradient_table(bvals, bvecs)
    dti_model = TensorModel(gtab)
    dti_fit = dti_model.fit(dwi_data)

    # compute and save outputs
    for label, fnc in dti_fnc_dict.items():
        img = nib.Nifti1Image(fnc(dti_fit.evals),
                              dwi_img.affine)
        nib.save(img, _folder / f'{label}.nii.gz')
