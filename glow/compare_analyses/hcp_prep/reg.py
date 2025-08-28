register_cmd = '''{ants_path}antsRegistration --dimensionality 3 --float 0
--output [{folder_out}{sub}_,{folder_out}{sub}_Warped.nii.gz]
--interpolation Linear
--winsorize-image-intensities [0.005,0.995]
--use-histogram-matching 0
--initial-moving-transform [{img_fix},{img_move},1]
--transform Rigid[0.1]
--metric MI[{img_fix},{img_move},1,32,Regular,0.25]
--convergence [1000x500x250x100,1e-6,10]
--shrink-factors 8x4x2x1
--smoothing-sigmas 3x2x1x0vox
--transform Affine[0.1]
--metric MI[{img_fix},{img_move},1,32,Regular,0.25]
--convergence [1000x500x250x100,1e-6,10]
--shrink-factors 8x4x2x1
--smoothing-sigmas 3x2x1x0vox
--transform SyN[0.1,3,0]
--metric CC[{img_fix},{img_move},1,4]
--convergence [100x70x50x20,1e-6,10]
--shrink-factors 8x4x2x1
--smoothing-sigmas 3x2x1x0vox'''

# remove new lines
register_cmd = ' '.join(register_cmd.split('\n'))

# todo: build template

# todo: extract sbj

# todo: register FA to template

fmt_dict = dict(ants_path='/home/matt/Downloads/install/bin/',
                img_move='/home/matt/Downloads/register_play/fa_move.nii.gz',
                img_fix='/home/matt/Downloads/register_play/fa_fix.nii.gz',
                folder_out='/home/matt/Downloads/register_play/',
                sub='123')
print(register_cmd.format(**fmt_dict))