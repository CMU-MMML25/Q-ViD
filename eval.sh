result_dir="result/"

exp_name='star_infer'
ckpt='qvid_checkpoints/instruct_blip_flanxl_trimmed.pth'
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 evaluate.py \
--cfg-path lavis/projects/qvid/star_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.model_type=flant5xl \
model.cap_prompt="Provide a detailed description of the image related to the" \
model.qa_prompt="Considering the information presented in the captions, select the correct answer in one letter (A,B,C,D) from the options." \
datasets.star.vis_processor.eval.n_frms=8 \
run.batch_size_eval=4 \
model.task='qvh_freeze_loc_freeze_qa_vid' \
model.finetuned=${ckpt} \
run.task='videoqa' \
--subset-ratio 1
