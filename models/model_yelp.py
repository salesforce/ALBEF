from functools import partial

from numpy import dtype
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel

import torch
from torch import nn
import torch.nn.functional as F

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
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    

        bert_config = BertConfig.from_json_file(config['bert_config'])

        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)          

        self.cls_head = nn.Sequential(
                  nn.Linear(
                    self.text_encoder.config.hidden_size, 
                    self.text_encoder.config.hidden_size
                  ),
                  nn.ReLU(),
                  nn.Linear(
                        self.text_encoder.config.hidden_size, 
                        config['num_label']
                    )
                )

        if self.distill:
            self.visual_encoder_m = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))               
            self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)      
            self.cls_head_m = nn.Sequential(
                    nn.Linear(
                        self.text_encoder.config.hidden_size, 
                        self.text_encoder.config.hidden_size
                    ),
                    nn.ReLU(),
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
         
    @torch.no_grad()
    def split_words(self, input_ids):
        '''
        input_ids: bs * num_max_words 
        '''
        bs = input_ids.size(0)
        input_ids = input_ids.clone().detach()
        ret = []
        for b_id in range(bs):
            max_num_words = 0
            sent = input_ids[b_id][1:]
            if sent[-1].item() == 0:
                sent = sent[:(sent==0).nonzero().min().item()]
            idx = (sent == 102).nonzero().squeeze(1)
            idx = idx + 1
            idx_bak = idx.clone().detach()
            for i in range(1, len(idx)):
                idx[i] = idx_bak[i] - idx_bak[i - 1]
            idx = idx.tolist()
            # print(sent, idx, idx_bak)
            sents = list(sent.split(idx))
            for i in range(len(sents)):
                sents[i] = [101] + sents[i].tolist()
                max_num_words = max(max_num_words, len(sents[i]))

            for i in range(len(sents)):
                sents[i] = sents[i] + [0] * (max_num_words - len(sents[i]))
            ret.append(torch.tensor(sents).to(input_ids.device))
        return ret

    def get_feat(self, inputs):
        bs = len(inputs)
        b = torch.zeros(bs, self.text_encoder.config.hidden_size).to(inputs[0][0].device)
        for i in range(bs):
            # sent_num = len(inputs[i])
            sents = inputs[i]
            att_mask = torch.ones_like(sents).to(sents.device)
            output = self.text_encoder(
                sents,
                attention_mask=att_mask,
                return_dict=True,
                mode='text'
            )
            a = output.last_hidden_state[:, 0, :].mean(dim=0)
        b[i] = a
        return b
            
    def forward(self, image, text, label, alpha=0, train=True):
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        
        if train:
            output = self.get_feat(self.split_words(text.input_ids))
            # output = self.text_encoder(text.input_ids, 
            #                            attention_mask = text.attention_mask, 
            #                            encoder_hidden_states = image_embeds,
            #                            encoder_attention_mask = image_atts,        
            #                            return_dict = True
            #                           )         
            # prediction = self.cls_head(output.last_hidden_state.mean(dim=1))
            prediction = self.cls_head(output)
            # prediction = self.cls_head(output.last_hidden_state[:,0,:])                
            if self.distill:                
                with torch.no_grad():
                    self._momentum_update()
                    image_embeds_m = self.visual_encoder_m(image) 
                    output_m = self.text_encoder_m(text.input_ids, 
                                               attention_mask = text.attention_mask, 
                                               encoder_hidden_states = image_embeds_m,
                                               encoder_attention_mask = image_atts,        
                                               return_dict = True
                                              )           
                    prediction_m = self.cls_head_m(output_m.last_hidden_state[:,0,:])   

                loss = (1-alpha)*F.cross_entropy(prediction, label) - alpha*torch.sum(
                    F.log_softmax(prediction, dim=1)*F.softmax(prediction_m, dim=1),dim=1).mean()
            else:
                loss = F.cross_entropy(prediction, label)                
            return loss 
            
        else:
            output = self.get_feat(self.split_words(text.input_ids))
            # output = self.text_encoder(text.input_ids, 
            #                            attention_mask = text.attention_mask, 
            #                            encoder_hidden_states = image_embeds,
            #                            encoder_attention_mask = image_atts,        
            #                            return_dict = True
            #                           )         
            prediction = self.cls_head(output)                        
            # prediction = self.cls_head(output.last_hidden_state.mean(dim=1))                        
            # prediction = self.cls_head(output.last_hidden_state[:,0,:])                        
            return prediction
 


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
                


