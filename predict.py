import re
import tempfile
from functools import partial
import cv2
from PIL import Image
import numpy as np
from cog import BasePredictor, Path, Input

from skimage import transform as skimage_transform
from scipy.ndimage import filters
from matplotlib import pyplot as plt

import torch
from torch import nn
from torchvision import transforms

from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel
from models.tokenization_bert import BertTokenizer


class Predictor(BasePredictor):
    def setup(self):
        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize((384, 384), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.tokenizer = BertTokenizer.from_pretrained("bert/bert-base-uncased")

        bert_config_path = "configs/config_bert.json"
        self.model = VL_Transformer_ITM(
            text_encoder="bert/bert-base-uncased", config_bert=bert_config_path
        )

        checkpoint = torch.load("refcoco.pth", map_location="cpu")
        msg = self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()

        self.block_num = 8
        self.model.text_encoder.base_model.base_model.encoder.layer[
            self.block_num
        ].crossattention.self.save_attention = True

        self.model.cuda()

    def predict(
        self,
        image: Path = Input(description="Input image."),
        caption: str = Input(
            description="Caption for the image. Grad-CAM visualization will be generated "
            "for each word in the cation."
        ),
    ) -> Path:

        image_pil = Image.open(str(image)).convert("RGB")
        img = self.transform(image_pil).unsqueeze(0)

        text = pre_caption(caption)
        text_input = self.tokenizer(text, return_tensors="pt")

        img = img.cuda()
        text_input = text_input.to(img.device)

        # Compute GradCAM
        output = self.model(img, text_input)
        loss = output[:, 1].sum()

        self.model.zero_grad()
        loss.backward()

        with torch.no_grad():
            mask = text_input.attention_mask.view(
                text_input.attention_mask.size(0), 1, -1, 1, 1
            )

            grads = self.model.text_encoder.base_model.base_model.encoder.layer[
                self.block_num
            ].crossattention.self.get_attn_gradients()
            cams = self.model.text_encoder.base_model.base_model.encoder.layer[
                self.block_num
            ].crossattention.self.get_attention_map()

            cams = cams[:, :, :, 1:].reshape(img.size(0), 12, -1, 24, 24) * mask
            grads = (
                grads[:, :, :, 1:].clamp(0).reshape(img.size(0), 12, -1, 24, 24) * mask
            )

            gradcam = cams * grads
            gradcam = gradcam[0].mean(0).cpu().detach()

        num_image = len(text_input.input_ids[0])
        fig, ax = plt.subplots(num_image, 1, figsize=(20, 8 * num_image))

        rgb_image = cv2.imread(str(image))[:, :, ::-1]
        rgb_image = np.float32(rgb_image) / 255

        ax[0].imshow(rgb_image)
        ax[0].set_yticks([])
        ax[0].set_xticks([])
        ax[0].set_xlabel("Image")

        for i, token_id in enumerate(text_input.input_ids[0][1:]):
            word = self.tokenizer.decode([token_id])
            gradcam_image = getAttMap(rgb_image, gradcam[i + 1])
            ax[i + 1].imshow(gradcam_image)
            ax[i + 1].set_yticks([])
            ax[i + 1].set_xticks([])
            ax[i + 1].set_xlabel(word)

        out_path = Path(tempfile.mkdtemp()) / "output.png"
        fig.savefig(str(out_path))
        return out_path


class VL_Transformer_ITM(nn.Module):
    def __init__(self, text_encoder=None, config_bert=""):
        super().__init__()

        bert_config = BertConfig.from_json_file(config_bert)

        self.visual_encoder = VisionTransformer(
            img_size=384,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        self.text_encoder = BertModel.from_pretrained(
            text_encoder, config=bert_config, add_pooling_layer=False
        )

        self.itm_head = nn.Linear(768, 2)

    def forward(self, image, text):
        image_embeds = self.visual_encoder(image)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        output = self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        vl_embeddings = output.last_hidden_state[:, 0, :]
        vl_output = self.itm_head(vl_embeddings)
        return vl_output


def pre_caption(caption, max_words=30):
    caption = (
        re.sub(
            r"([,.'!?\"()*#:;~])",
            "",
            caption.lower(),
        )
        .replace("-", " ")
        .replace("/", " ")
    )

    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    # truncate caption
    caption_words = caption.split(" ")
    if len(caption_words) > max_words:
        caption = " ".join(caption_words[:max_words])
    return caption


def getAttMap(img, attMap, blur=True, overlap=True):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order=3, mode="constant")
    if blur:
        attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    cmap = plt.get_cmap("jet")
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = (
            1 * (1 - attMap ** 0.7).reshape(attMap.shape + (1,)) * img
            + (attMap ** 0.7).reshape(attMap.shape + (1,)) * attMapV
        )
    return attMap
