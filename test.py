from transformers import BertModel, BertTokenizer
import torch
from torch import nn
import re
import torch.nn.functional as F

def pre_ac(text):
    ret = []
    texts = text.split('|||')
    for text in texts:
        url_pattern = '(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
        text = re.sub(url_pattern, '', text)
        # tag_pattern = '#[a-zA-Z0-9]*'
        # text = re.sub(tag_pattern, '', text)
        at_pattern = '@[a-zA-Z0-9]*'
        text = re.sub(at_pattern, '', text)
        not_ascii_pattern = '[^a-zA-Z0-9]'
        text = re.sub(not_ascii_pattern, ' ', text)
        text = re.sub(' +', ' ', text)
        text = text.strip()
        ret.append(text)
    return ret

target = 4 - 1

text = \
'''
pokinometry is as easy as 1-2-3 !|||just follow these simple steps : # 1 : decide if you want white rice , brown rice , salad , chips or any two for half and half # 2 : choose 2 scoops of fish for small , 3 scoops medium or 5 scoops for a large bowl # 3 : would you like some sauce ?|||if yes , spicy or non-spicy ?|||spicy mayo is new .|||get it !|||# 4 : how about some toppings ?|||smelt eggs , onions , and sesame seeds are the basic # 5 : pay up and enjoy your creation !|||fish choices include : tuna , salmon , yellow tail , albacore , shrimp , octopus , scallop , mashed spicy tuna ( all raw except for the shrimp and octopus ) i for one went with the large bowl with half white and half brown rice .|||slices of cucumbers , some avocado , onions , and imitation crab are automatic .|||do n't skip those .|||i got two scoops of salmon , a scoop of mashed spicy tuna , octopus , and yellow tail .|||everything was harmoniously delicious that i 'll pick exactly the same fish all over again in a heartbeat .|||i love a good kick for some excitement so i asked for medium spice level plus their spicy mayo sauce to be added .|||boy was my tongue on fire .|||love it !|||of course , all toppings added but with ginger on the side .|||i have to say , i outdid myself with my own genius creation .|||bravo !|||the concept of customizing your own poke bowl is utterly brilliant .|||it 's one of those things where you 'd ask yourself : `` now , why did n't i think of that ? ''|||my thoughts exactly !|||overhead expense is low with it being a self-service establishment and no hot kitchen needed .|||it 's affordable so the target market is pretty much everyone who enjoys eating raw fish .|||with disposable bowls and utensils , you can easily decide to eat in or have it to-go .|||the ambiance is clean and simplistic with no point for guests to linger and take their time , thus making the turnover pretty quick .|||only caveat is the parking .|||otherwise , this business is the shiznit !'
'''


text2 = \
    '''
    谈下使用f4一周的感受，是第二轮在天猫抢购到的，过了五六分钟还能抢到，说明备货还是很足的，不像某些大厂一样玩饥饿营销。说下主要有点：1、599的指纹机，不知道算不算最低的，不过绝对良心了；2、整体手机握感很舒服，虽然只有边框是金属的，但是还挺有质感额；3、360OS的系统点个赞，财产隔离、冷藏室之类的黑科技不错。。不足就是内存有点不够用，不过599能这配置也算良心了，要是早点出高配版就好了~·。599元的价格，有金属、有指纹、有双微信，还是4G，还要啥自行车呢。 
    '''

# text = text2
texts = pre_ac(text)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained(
    'bert-base-uncased', 
    config='configs/config_demo.json', 
    add_pooling_layer=True
)
cls_head = nn.Sequential(
            nn.Linear(
                bert.config.hidden_size, 
                bert.config.hidden_size
            ),
            nn.ReLU(),
            nn.Linear(
                bert.config.hidden_size, 
                5
            )
        )


@torch.no_grad()
def split_words(input_ids):
    '''
    input_ids: bs * num_max_words 
    '''
    bs = input_ids.size(0)
    input_ids = input_ids.clone().detach()
    ret = []
    for b_id in range(bs):
        # max_num_words = 0
        sent = input_ids[b_id][1:]
        if sent[-1].item() == 0:
            sent = sent[:(sent==0).nonzero().min().item()]
        idx = (sent == 102).nonzero().squeeze(1)
        idx = idx + 1
        for i in range(1, len(idx)):
            idx[i] = idx[i] - idx[i - 1]
        idx = idx.tolist()
        sents = list(sent.split(idx))
        for i in range(len(sents)):
            sents[i] = torch.tensor([101] + sents[i].tolist()).to(input_ids.device)
            # max_num_words = max(max_num_words, len(sents[i]))

        # for i in range(len(sents)):
        #     sents[i] = sents[i] + [0] * (max_num_words - len(sents[i]))
        ret.append(sents)
    return ret 

def get_feat(inputs):
    bs = len(inputs)
    a = torch.zeros(bert.config.hidden_size)
    for i in range(bs):
        sent_num = len(inputs[i])
        for sent in inputs[i]:
            att_mask = torch.ones_like(sent).to(sent.device)
            output = bert(
                sent.unsqueeze(0),
                attention_mask=att_mask.unsqueeze(0),
                return_dict=True,
                # mode='text'
            )
        a = a + output.last_hidden_state[:, 0, :]
    a = a / sent_num
    return a

# n = len(texts)
# a = torch.zeros(bert.config.hidden_size)
# for text in texts:
#     tokens = tokenizer(text, truncation=True, return_tensors="pt")
#     output = bert(
#         tokens.input_ids,
#         attention_mask = tokens.attention_mask, 
#         return_dict = True
#     )
#     a = a + output.last_hidden_state[:, 0, :]
# a = a / n
# prediction = cls_head(a)
# _, pred_class = prediction.max(1)
batch = ['Hello world.[SEP]Hello python and pytorch.', 'Are you happy?']
tokens = tokenizer(
    batch, 
    padding=True, 
    return_tensors='pt'
)
inputs = split_words(tokens.input_ids)
feat = get_feat(inputs)
prediction = cls_head(feat)
_, pred_class = prediction.max(1)
print(prediction, pred_class)
# print(tokens.input_ids)
# print(split_words(tokens.input_ids))
# 101 [CLS] 102 [SEP]