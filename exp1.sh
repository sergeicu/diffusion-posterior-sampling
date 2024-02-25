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
- 
- built 3D dataloader 
- built 3D unet for which we can run inferrence 


- reduce unet size
- may need to change this: def get_dataloader(dataset: VisionDataset, to handle Dataset and VisionDataset in sample_condition.py - depending on the case 
- may need to change "save_root" -> since it is writing to the same thing as odd_even .. 


# NEXT STEPS: 
- integrate DPS into DDPM in diffusemorph 
- at first we need to do it without 

