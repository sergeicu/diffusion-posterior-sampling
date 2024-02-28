cd /fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/diffusion/experiments/s20240209_oecorr/diffusion-posterior-sampling
conda activate diffusers 
source venv/bin/activate 


# https://github.com/DPS2022/diffusion-posterior-sampling?tab=readme-ov-file

python3 sample_condition.py \
--model_config=configs/model_config.yaml \
--diffusion_config=configs/diffusion_config.yaml \
--task_config=configs/super_resolution_config.yaml



python3 sample_condition.py \
--model_config=configs/model_config.yaml \
--diffusion_config=configs/diffusion_config.yaml \
--task_config=configs/oddeven_config.yaml



python3 sample_condition.py \
--model_config=configs/model_config_3d.yaml \
--diffusion_config=configs/diffusion_config.yaml \
--task_config=configs/oddeven_config_cardiac.yaml



# TODO: 

- may need to change "save_root" -> since it is writing to the same thing as odd_even .. 
- create training routine for guided_diffusion... (this one will take some time... )


# NEXT STEPS: 
- implement DPS with DDPM_only trained DDM 
    - first need to implement standard inferrence lol 
    - 

- [less priority] train voxelmorph - aka DiffuseMorph without DDM (just direct image )

