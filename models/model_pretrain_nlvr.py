from functools import partial
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
        vision_width = config['vision_width']  
        embed_dim = config['embed_dim']

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    

        bert_config = BertConfig.from_json_file(config['bert_config'])
        bert_config.num_hidden_layers = 18
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)      

        #share the cross-attention layers for two images
        self.share_cross_attention(self.text_encoder.encoder)            

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)  
        self.temp = nn.Parameter(torch.ones([]) * 0.07)   
        self.ta_head = nn.Linear(self.text_encoder.config.hidden_size, 3)   
            
            
    def forward(self, image, text):
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        with torch.no_grad():            
            image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  
            sim = image_feat @ image_feat.t() / 0.07
            weights = F.softmax(sim,dim=1)
            weights.fill_diagonal_(0)

        image_inputs = [[],[]]    
        labels = []
        for b in range(image.size(0)):
            if torch.rand(1)>1/3:
                idx = torch.multinomial(weights[b], 1).item()
                if torch.rand(1)>0.5:
                    image_inputs[0].append(image_embeds[b])
                    image_inputs[1].append(image_embeds[idx])
                    labels.append(0)
                else:
                    image_inputs[1].append(image_embeds[b])
                    image_inputs[0].append(image_embeds[idx])        
                    labels.append(1)
            else:
                idx = torch.multinomial(weights[b], 2)
                image_inputs[0].append(image_embeds[idx[0]])
                image_inputs[1].append(image_embeds[idx[1]])
                labels.append(2)                    

        image_inputs[0] = torch.stack(image_inputs[0],dim=0)         
        image_inputs[1] = torch.stack(image_inputs[1],dim=0)      
        labels = torch.LongTensor(labels).to(image.device)

        output = self.text_encoder(text.input_ids,
                                   attention_mask = text.attention_mask, 
                                   encoder_hidden_states = image_inputs, 
                                   encoder_attention_mask = [image_atts,image_atts],        
                                   return_dict = True,
                                  )  

        pred = self.ta_head(output.last_hidden_state[:,0,:])
        loss = F.cross_entropy(pred, labels)     

        return loss      
 


    def share_cross_attention(self, model):
            
        for i in range(6):
            layer_num = 6+i*2
            modules_0 = model.layer[layer_num].crossattention.self._modules
            modules_1 = model.layer[layer_num+1].crossattention.self._modules

            for name in modules_0.keys():
                if 'key' in name or 'value' in name:
                    module_0 = modules_0[name]
                    module_1 = modules_1[name]
                    if hasattr(module_0, "weight"):
                        module_0.weight = module_1.weight
                        if hasattr(module_0, "bias"):
                            module_0.bias = module_1.bias    