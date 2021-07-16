## Align before Fuse: Vision and Language Representation Learning with Momentum Distillation (Salesforce Research)

This is the official PyTorch implementation of the <a href="">ALBEF paper</a> <a href="">[Blog]</a>. 
This repository supports finetuning ALBEF on VQA, SNLI-VE, NLVR2, Image-Text Retrieval on MSCOCO and Flickr30k,
and visual grounding on RefCOCO+. Pre-trained and Fine-tuned checkpoints are released.
<img src="img.png" width="600">


### Requirements:
* pytorch 1.8.0
* transformers 4.8.1

### Download:

* <a href="https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF.pth"> Pre-trained checkpoint </a>
* <a href="https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/data.tar.gz"> Dataset json files </a>
* <a href="https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/mscoco.pth"> Finetuned checkpoint for retrieval on MSCOCO </a>
* <a href="https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/vqa.pth"> Finetuned checkpoint for VQA </a>
* <a href="https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/refcoco.pth"> Finetuned checkpoint for visual grounding on RefCOCO+ </a>

### Visualization:
We provide code in visualize.ipynb to visualize the important areas in an image for each word in a text. 
Here is an example visualization using the visual grounding checkpoint.

<img src="examples/visualization.png" width="700">



### Image-Text Retrieval:

1. Download MSCOCO or Flickr30k datasets from original websites.
2. Download and extract the provided dataset json files.
3. In configs/Retrieval_coco.yaml or configs/Retrieval_flickr.yaml, set the paths for the json files and the image path.
4. Finetune the pre-trained checkpoint using 8 A100 GPUs:
<pre>python -m torch.distributed.launch --nproc_per_node=8 --use_env Retrieval.py \
--config ./configs/Retrieval_flickr.yaml \
--output_dir output/Retrieval_flickr \
--checkpoint [Pretrained checkpoint]</pre> 
