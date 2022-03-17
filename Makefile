train:
	#python -m torch.distributed.launch --nproc_per_node=4 --use_env Demo.py
	python Demo.py --distributed=False
