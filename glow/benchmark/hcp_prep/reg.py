import re
from pathlib import Path
import ants
import numpy as np


def build_template(fa_files):
    '''Build a simple average FA template.'''
    images = [ants.image_read(str(f)) for f in fa_files]
    arrs = [img.numpy() for img in images]
    avg_arr = np.mean(arrs, axis=0)
    template = ants.from_numpy(
        avg_arr,
        origin=images[0].origin,
        spacing=images[0].spacing,
        direction=images[0].direction,
    )
    return template


def collect_subject_files(path_in, sbj_regex, img_regex_list):
    '''
    Scan folder for images matching patterns, group by subject ID.
    Returns: dict {sbj: {modality: Path}}
    '''
    path_in = Path(path_in)
    sbj_dict = {}

    for regex in img_regex_list:
        files = path_in.glob(f'**/*{regex}.nii.gz')
        for f in files:
            sbj_match = re.search(sbj_regex, str(f))
            if sbj_match is None:
                print(f'[WARN] No subject ID found for {f}')
                continue
            sbj = sbj_match.group(0)
            if sbj not in sbj_dict:
                sbj_dict[sbj] = {}
            sbj_dict[sbj][regex] = f

    return sbj_dict


def register_and_warp(sbj_dict, path_out, register_on='fa'):
    '''
    Build template (average), then register and warp subject images.
    Additionally, warp nodif_brain_mask.nii.gz for each subject and
    create a group mask that is applied to all warped outputs.
    '''
    path_out = Path(path_out)
    path_out.mkdir(parents=True, exist_ok=True)

    # Build template from FA images
    fa_files = [d[register_on] for d in sbj_dict.values() if register_on in d]
    if not fa_files:
        raise RuntimeError(f'No {register_on} images found in subject dict')
    template = build_template(fa_files)
    template.to_file(str(path_out / f'{register_on}_template.nii.gz'))

    warped_masks = []

    # Register and warp each subject
    for sbj, imgs in sbj_dict.items():
        if register_on not in imgs:
            print(f'[WARN] Subject {sbj} has no {register_on} image. Skipping.')
            continue

        moving_img = ants.image_read(str(imgs[register_on]))
        reg = ants.registration(
            fixed=template,
            moving=moving_img,
            type_of_transform='SyN',
            metric='CC',
            reg_iterations=[100, 70, 50, 20]
        )

        # Warp register_on modality
        warped_reg_img = reg['warpedmovout']
        warped_reg_img.to_file(str(path_out / f'{sbj}_{register_on}.nii.gz'))

        # Warp other modalities
        for modality, path in imgs.items():
            if modality == register_on or modality == 'nodif_brain_mask':
                continue
            img = ants.image_read(str(path))
            warped_img = ants.apply_transforms(
                fixed=template,
                moving=img,
                transformlist=reg['fwdtransforms']
            )
            warped_img.to_file(str(path_out / f'{sbj}_{modality}.nii.gz'))

        # Warp subject brain mask
        if 'nodif_brain_mask' in imgs:
            mask_img = ants.image_read(str(imgs['nodif_brain_mask']))
            warped_mask = ants.apply_transforms(
                fixed=template,
                moving=mask_img,
                transformlist=reg['fwdtransforms'],
                interpolator='nearestNeighbor'
            )
            warped_masks.append(warped_mask.numpy())

        print(f'[DONE] Subject {sbj} registered and warped.')

    # Compute group mask (intersection of all warped masks)
    if warped_masks:
        group_mask_arr = np.logical_and.reduce(warped_masks).astype(np.uint8)
        group_mask = ants.from_numpy(
            group_mask_arr,
            origin=template.origin,
            spacing=template.spacing,
            direction=template.direction,
        )
        group_mask.to_file(str(path_out / 'group_mask.nii.gz'))
        print(f'[INFO] Group mask saved at {path_out / "group_mask.nii.gz"}')

        # Apply group mask to all warped FA/MD images
        for f in path_out.glob('*.nii.gz'):
            if f.name.endswith('_fa.nii.gz') or f.name.endswith('_md.nii.gz'):
                img = ants.image_read(str(f))
                masked_img = img * group_mask
                masked_img.to_file(str(f))


if __name__ == '__main__':
    path_in = '/home/matt/data/hcp100_aug25'
    path_out = '/home/matt/data/hcp100_aug25_registered'
    sbj_regex = r'[\d]{6}'
    img_regex_list = ['fa', 'md', 'nodif_brain_mask']
    register_on = 'fa'

    sbj_dict = collect_subject_files(path_in, sbj_regex, img_regex_list)

    register_and_warp(sbj_dict, path_out, register_on=register_on)
