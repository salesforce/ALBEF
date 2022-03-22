train:
	python -m torch.distributed.launch --nproc_per_node=4 --use_env Yelp.py
	#python Yelp.py --distributed=False
test:
	python Yelp.py --distributed=False --evaluate --checkpoint='output/Yelp/checkpoint_best.pth'
