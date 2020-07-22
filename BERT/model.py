import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from transformers.modeling_bert import BertLayerNorm
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

        
class Model(nn.Module):   
    def __init__(self, encoder,config,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.lm_head=[]
        self.text_embeddings=nn.Embedding(args.vocab_size_v1,args.vocab_dim_v1)
        self.text_embeddings.apply(self._init_weights)
        self.text_linear=nn.Linear(args.text_dim+args.vocab_dim_v1*len(args.text_features), config.hidden_size) 
        self.text_linear.apply(self._init_weights)
        for x in args.vocab_size:
            self.lm_head.append(nn.Linear(config.hidden_size, x, bias=False))
        self.lm_head=nn.ModuleList(self.lm_head)
        self.config=config
        self.args=args
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            
    def forward(self, inputs,inputs_ids,masks,labels):     
        inputs_embedding=self.text_embeddings(inputs_ids).view(inputs.size(0),inputs.size(1),-1)
        inputs=torch.cat((inputs.float(),inputs_embedding),-1)
        inputs=torch.relu(self.text_linear(inputs))
        outputs = self.encoder(inputs_embeds=inputs,attention_mask=masks.float())[0]
        loss=0
        for idx,(x,y) in enumerate(zip(self.lm_head,self.args.text_features)):
            if y[3] is True:
                outputs_tmp=outputs[labels[:,:,idx].ne(-100)]
                labels_tmp=labels[:,:,idx]
                labels_tmp=labels_tmp[labels_tmp.ne(-100)].long()
                prediction_scores = x(outputs_tmp)
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(prediction_scores, labels_tmp)  
                loss=loss+masked_lm_loss
        return loss
         
 