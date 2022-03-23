from functools import partial

from numpy import dtype
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.distill = config['distill']

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], 
            patch_size=16, 
            embed_dim=768, 
            depth=12, 
            num_heads=12, 
            mlp_ratio=4, 
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_rate=0.5,
        )    

        bert_config = BertConfig.from_json_file(config['bert_config'])

        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)

        self.cls_head = nn.Sequential(
                  nn.Linear(
                    self.text_encoder.config.hidden_size, 
                    self.text_encoder.config.hidden_size
                  ),
                  nn.Dropout(0.5),
                  nn.ReLU(),
                  nn.LayerNorm(self.text_encoder.config.hidden_size, eps=1e-6),
                  nn.Linear(
                        self.text_encoder.config.hidden_size, 
                        config['num_label']
                    )
                )

        if self.distill:
            self.visual_encoder_m = VisionTransformer(
                img_size=config['image_res'], 
                patch_size=16, 
                embed_dim=768, 
                depth=12, 
                num_heads=12, 
                mlp_ratio=4, 
                qkv_bias=True, 
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                drop_rate=0.5,
            )
            self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)      
            self.cls_head_m = nn.Sequential(
                    nn.Linear(
                        self.text_encoder.config.hidden_size, 
                        self.text_encoder.config.hidden_size
                    ),
                    nn.Dropout(0.5),
                    nn.ReLU(),
                    nn.LayerNorm(self.text_encoder.config.hidden_size, eps=1e-6),
                    nn.Linear(
                        self.text_encoder.config.hidden_size, 
                        config['num_label']
                    )
            )

            self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                                [self.text_encoder,self.text_encoder_m],
                                [self.cls_head,self.cls_head_m],
                               ]
            self.copy_params()        
            self.momentum = 0.995
         

    def get_t_feat(self, inputs, device):
        bs = len(inputs)
        b = []
        num_max_sent = 0
        for i in range(bs):
            sents = torch.tensor(inputs[i], dtype=torch.long).to(device)[:20]
            # print(sents.size())
            att_mask = torch.ones(sents.shape, dtype=torch.long).to(device)
            output = self.text_encoder(
                sents,
                attention_mask=att_mask,
                return_dict=True,
                mode='text'
            )
            # sent_feat = output.last_hidden_state.max(dim=0).values
            sent_feat = output.last_hidden_state[:,0,:]
            # doc_feat = sent_feat.mean(dim=0)
            b.append(sent_feat)
            num_max_sent = max(num_max_sent, sent_feat.size(0))
        ret = torch.zeros(
            bs, 
            num_max_sent, 
            self.text_encoder.config.hidden_size, 
            dtype=torch.float
        ).to(device)

        for i in range(bs):
            ret[i][:b[i].size(0)] = b[i]
        return ret

    def get_v_feat(self, inputs, device):
        bs = inputs.size(0)
        num_max_img = 0
        b = []
        for i in range(bs):
            img = inputs[i]
            output = self.visual_encoder(img)
            img_feat = output[:,0,:]
            b.append(img_feat)
            num_max_img = max(num_max_img, img_feat.size(0))

        ret = torch.zeros(
            bs, 
            num_max_img,
            768,
            dtype=torch.float
        ).to(device)

        for i in range(bs):
            ret[i][:b[i].size(0)] = b[i]
        return ret

    def forward(self, image, text, label, device, alpha=0, train=True):
        output_v = self.get_v_feat(image, device)
        output_t = self.get_t_feat(text, device)
        output_fuse = self.text_encoder(
            encoder_embeds = output_t,
            encoder_hidden_states = output_v,
            mode='fusion',
            return_dict = True
        )
        logit = self.cls_head(output_fuse.last_hidden_state[:,0,:])
        if train:
            if self.distill:                
                with torch.no_grad():
                    self._momentum_update()
                    output_t_m = self.get_t_feat(text, device)
                    output_v_m = self.get_v_feat(image, device)
                    output_fuse_m = self.text_encoder_m(
                        encoder_embeds = output_t_m,
                        encoder_hidden_states = output_v_m,
                        mode='fusion',
                        return_dict = True
                    )
                    prediction_m = self.cls_head_m(output_fuse_m.last_hidden_state[:,0,:])

                loss = (1-alpha)*F.cross_entropy(logit, label) - alpha*torch.sum(
                    F.log_softmax(logit, dim=1)*F.softmax(prediction_m, dim=1),dim=1).mean()
            else:
                loss = F.cross_entropy(logit, label)                
            return logit, loss 
            
        else:
            loss = F.cross_entropy(logit, label)                
            return logit, loss
 


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                


