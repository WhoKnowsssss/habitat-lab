import math
import logging
from turtle import forward

import torch
import torch.nn as nn
from torch.nn import functional as F
from .focal_loss import FocalLoss, ActionNorm

logger = logging.getLogger(__name__)

import numpy as np

class LSTMBC(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model_type = config.model_type

        self.num_inputs = 4

        config.block_size = config.block_size * self.num_inputs
        
        self.block_size = config.block_size

        self.n_embd = config.n_embd
        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.LSTM(self.n_embd, self.n_embd, num_layers=2, batch_first=True, dropout=0.1)
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, 2, bias=False)
        # self.head = nn.Linear(config.n_embd, 18, bias=False)
        # self.head_2 = nn.Linear(config.n_embd, 8, bias=False)
        self.head_2 = nn.Linear(config.n_embd, 7*11, bias=False)

        self.head_3 = nn.Linear(config.n_embd, 3, bias=False)
        self.head_4 = nn.Linear(config.n_embd, 1, bias=False)
        self.focal_loss = FocalLoss(
            alpha=torch.tensor([0.05,0.0125,0.0125,0.0125,0.0125,0.8,0.0125,0.0125,0.0125,0.0125,0.05]), gamma=5).cuda()

        self.action_normalization = ActionNorm(mean=torch.tensor([0.4667095,  0.00209379]), std=torch.tensor([0.61708325, 0.9862876])).cuda()
        
        self.apply(self._init_weights)

        self.loss_vars = nn.parameter.Parameter(torch.zeros((3,)))
        # self.loss1_var = nn.parameter.Parameter(torch.zeros((1,)))
        # self.loss2_var = nn.parameter.Parameter(torch.zeros((1,)))
        # self.loss3_var = nn.parameter.Parameter(torch.zeros((1,)))

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))


        if config.num_states[0] == 0:
            self.state_encoder = nn.Sequential(nn.Linear(config.num_states[1], config.n_embd), nn.Tanh())
        else:
            self.state_encoder = nn.ModuleList([nn.Sequential(nn.Linear(i, config.n_embd // 2), nn.Tanh()) for i in [config.num_states[1]]])

        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())

        self.action_embeddings = nn.Sequential(nn.Linear(config.vocab_size, config.n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

        self.boundaries = torch.tensor([-1.1, -0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1]).cuda()

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # state, action, and return
    def forward(self, states, actions, targets=None, rtgs=None, timesteps=None):
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 8)
        # targets: (batch, block_size, 8)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, 1, 1)
        # attention_mask: (batch, block_size)

        assert states.shape[1] == actions.shape[1] and actions.shape[1] == rtgs.shape[1], "Dimension must match, {}, {}, {}".format(states.shape[1], actions.shape[1], rtgs.shape[1])

        state_inputs = list(torch.split(states,[self.n_embd, self.n_embd//2,self.config.num_states[1]], -1))
        # vision_embeddings = self.vision_encoder(visual_input.reshape(-1, 1, 128, 128).type(torch.float32).contiguous())
        # vision_embeddings = vision_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd//2) # (batch, block_size, n_embd)

        for i in range(2, len(state_inputs)):
            state_inputs[i] = self.state_encoder[i-2](state_inputs[i].type(torch.float32))
        
        if actions is not None and self.model_type == 'reward_conditioned': 
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            actions[:,:,:7] = (torch.bucketize(actions[:,:,:7], self.boundaries) - 1) / 10
            actions[:,:,[8,9]] = self.action_normalization(actions[:,:,[8,9]])
            actions = actions.type(torch.float32)
            # targets = torch.bucketize(targets[:,:,:], self.boundaries) - 1
            if actions.shape[-1] == 12:
                actions = torch.cat([actions[:,:,:10], actions[:,:,11:]], dim=-1)
            action_embeddings = self.action_embeddings(actions) # (batch, block_size, n_embd)

            token_embeddings = torch.zeros((states.shape[0], (self.num_inputs -1) * states.shape[1], self.config.n_embd), dtype=torch.float32, device=action_embeddings.device)
            
            # token_embeddings[:,::self.num_inputs,:] = rtg_embeddings

            # for i in range(len(state_inputs)):
            #     token_embeddings[:,(i+1)::self.num_inputs,:] = state_inputs[i]
            token_embeddings[:,::(self.num_inputs-1),:] = state_inputs[0]
            token_embeddings[:,1::(self.num_inputs-1),:] = torch.cat([state_inputs[1], state_inputs[-1]], dim=-1)
            
            token_embeddings[:,(self.num_inputs-2)::(self.num_inputs-1),:] = action_embeddings

        x, _ = self.blocks(token_embeddings)
            
        x = self.ln_f(x)

        if actions is not None and self.model_type == 'reward_conditioned':
            logits_loc = self.head(x[:, (self.num_inputs-3)::(self.num_inputs-1), :]) # only keep predictions from state_embeddings
            logits_arm = self.head_2(x[:, (self.num_inputs-3)::(self.num_inputs-1), :])
            logits_pick = self.head_3(x[:, (self.num_inputs-3)::(self.num_inputs-1), :])
            logits_stop = self.head_4(x[:, (self.num_inputs-3)::(self.num_inputs-1), :])
        elif actions is None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1:, :]
        elif actions is not None and self.model_type == 'naive':
            logits = logits[:, ::2, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'naive':
            logits = logits # for completeness
        else:
            raise NotImplementedError()

        # if we are given some desired targets also calculate the loss
        loss = None
        loss_dict = None
        # a = (targets[:,:,9].long() + 1 + 2*torch.all(targets[:,:,8:-1].detach()==0,dim=-1))
        # print(a[0])
        # logits_loc = torch.argmax(logits_loc,dim=-1)
        # print(logits_loc[0])
        if targets is not None:
            # loss1 = F.cross_entropy(logits_loc.permute(0,2,1), (targets[:,:,9].long() + 1 + 2*torch.all(targets[:,:,8:-1].detach()==0,dim=-1)), label_smoothing=0.05)
            
            # boundaries = torch.tensor([-1.1, -0.9, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.9, 1.1]).cuda()
            # temp_target = torch.bucketize(targets[:,:,[8,9]], boundaries) - 1
            # loss1 = F.cross_entropy(logits_loc[:,:,:9].permute(0,2,1), temp_target[:,:,0], label_smoothing=0.05) + F.cross_entropy(logits_loc[:,:,9:].permute(0,2,1), temp_target[:,:,1], label_smoothing=0.05)
            targets[:,:,[8,9]] = self.action_normalization(targets[:,:,[8,9]])
            loss1 = F.mse_loss(logits_loc, targets[:,:,[8,9]], reduction='none')

            temp_target = torch.bucketize(targets[:,:,:7], self.boundaries) - 1
            logits_arm = logits_arm.view(*logits_arm.shape[:2], 7, 11)
            # loss2 = F.cross_entropy(logits_arm[:,:,:,:].permute(0,3,1,2), temp_target[:,:,:7], label_smoothing=0.00)
            loss2 = self.focal_loss(logits_arm[:,:,:,:].permute(0,3,1,2), temp_target[:,:,:7])
            # loss2 = F.mse_loss(torch.tanh(logits_arm[:,:,:8]), targets[:,:,:8], reduction='none')
            # mask1 = torch.all(targets[:,:,[8,9]] == 0, dim=-1)
            # mask2 = torch.all(targets[:,:,:7] == 0, dim=-1)
            # loss1[~mask2] = 0.
            # loss2[~mask1] = 0.
            loss1 = torch.mean(loss1)
            # loss2 = torch.mean(loss2)

            # temp_target = mask1 + 2 * mask2 - 1
            # loss4 = F.cross_entropy(logits_stop.permute(0,2,1), temp_target.long(), label_smoothing=0.05)

            # loss3 = F.binary_cross_entropy(torch.sigmoid(logits_arm[:,:,7]), (1-0.2) * (targets[:,:,7] >= 0).to(torch.float32) + 0.2 * 0.5)
            loss3 = F.cross_entropy(logits_pick.permute(0,2,1), torch.round(targets[:,:,7]).long() + 1, label_smoothing=0.1)
            loss_dict = {"locomotion": loss1.detach().item(), "arm": loss2.detach().item(), "pick": loss3.detach().item()} #, "phase": loss4.detach().item()
            loss1 = torch.exp(-self.loss_vars[0]) * loss1 + self.loss_vars[0]
            loss2 = torch.exp(-self.loss_vars[1]) * loss2 + self.loss_vars[1]
            loss3 = torch.exp(-self.loss_vars[2]) * loss3 + self.loss_vars[2]
            # loss4 = torch.exp(-self.loss_vars[3]) * loss4 + self.loss_vars[3]
            loss = loss1 + loss2 + loss3
            # loss = F.cross_entropy(logits_pick.permute(0,2,1), 3 * (targets[:,:,7] > 0) + (targets[:,:,7] < 0) + 2 * (targets[:,:,7] == 0) - 1)
            # loss += F.binary_cross_entropy_with_logits(logits_pick, F.sigmoid(targets[:,:,7]).unsqueeze(-1))

        # logits_loc = torch.argmax(logits_loc,dim=-1)
        
        # logits = torch.zeros((*logits_loc.shape[:2], 11),device=logits_loc.device)
        # logits[:,:,:7] = torch.tanh(logits_arm[:,:,:7])
        # logits[:,:,7] = 2 * torch.sigmoid(logits_arm[:,:,7]) - 1
        # # logits[:,:,7:8] = logits_pick
        # # logits[:,:,7] = torch.argmax(logits_pick, dim=-1) - 1
        # logits[:,:,8] = logits_loc == 1
        # logits[:,:,9] = logits_loc - 1
        # logits[:,:,9:-1][logits[:,:,9:-1]==2] = 0
        # logits[:,:,-1:] = 0 
        logits_loc = self.action_normalization.unnormalize(logits_loc)
        return (logits_loc, logits_arm, logits_pick), loss, loss_dict
