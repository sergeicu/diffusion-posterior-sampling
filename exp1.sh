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
--task_config=configs/oddeven_config.yaml

# TODO: 
- build 3D dataloader... 
- reduce unet size


# NEXT STEPS: 
- integrate DPS into DDPM in diffusemorph 
- at first we need to do it without 

