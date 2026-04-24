from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit


def update_ema_variables(ema_model, model, alpha_teacher):#, iteration):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

class OUR(nn.Module):
    """CoTTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, config,steps=1, episodic=False): #3
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.eps = 1e-8
        self.batch_idx = 0  # 记录当前batch索引

        self.trainable_dict = {k: v for k, v in self.model.named_parameters() if v.requires_grad}
        self.exp_sens = {np: 0.0 for np in self.trainable_dict.keys()}
        self.grad_weight = {np: 0.0 for np in self.trainable_dict.keys()}

        self.beta_3 = 0.5
        self.threshold =0.001  #h3d isprs 0.001
        self.temp = 50
        self.lambd = 0.01
        self.params = []

        self.ingnorlabel = 11  # 9 #4
        if config.datasetClass=='H3D':
            self.ingnorlabel=11 #9 #4
        elif config.datasetClass=='ISPRS':
            self.ingnorlabel=9

        self.celoss=nn.CrossEntropyLoss(ignore_index= self.ingnorlabel)

        self.OptimizerLayer = []

        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x,config,aa):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x,config, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # use this line if you want to reset the teacher model as well. Maybe you also 
        # want to del self.model_ema first to save gpu memory.
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)                         


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x,config, model, optimizer):
        self.batch_idx+=1
        logits,_ = self.model(x.weakbatch,config,None)
        logits_aug,_ = self.model(x.strongbatch,config,None)
        del _
       
        self.model.zero_grad()
        logsoftmax = nn.LogSoftmax(dim=-1)
        uniform_dist = (torch.ones((x.weakbatch['labels'].shape[0], config.num_classes))/config.num_classes).to(
            'cuda')
        logits_copy = logits / self.temp # temperature scaling
        loss = torch.mean(torch.sum(-uniform_dist * logsoftmax(logits_copy), dim=-1))  # KL divergence
        loss.backward(retain_graph=True)

        # params = []
        layer_grad_norms = {}  # 存储每层的梯度范数总和
        layer_param_counts = {}  # 存储每层的参数数量

        # 按层汇总梯度和参数数量
        for n, p in self.trainable_dict.items():
            # 提取最后一个 '.' 之前的所有部分作为唯一层名
            unique_layer_name = '.'.join(n.split('.')[:-1])  # 例如 'encoder.blocks.0.KPCnn'

            # 计算当前参数的梯度范数
            layer_grad = p.grad.data
            grad_norm = torch.norm(layer_grad, p=1).cpu().numpy()

            # 累加到对应层的总梯度范数
            if unique_layer_name not in layer_grad_norms:
                layer_grad_norms[unique_layer_name] = 0
                layer_param_counts[unique_layer_name] = 0
            layer_grad_norms[unique_layer_name] += grad_norm
            layer_param_counts[unique_layer_name] += 1  # 记录参数数量


        self.model.zero_grad()

        params = []
        OptimizerLayer=[]
        # 根据每层的梯度均值决定学习率，并统计
        for unique_layer_name, total_grad_norm in layer_grad_norms.items():
            # 计算梯度均值
            param_count = layer_param_counts[unique_layer_name]
            grad_mean = total_grad_norm / param_count if param_count > 0 else 0  # 避免除以 0
            
            if grad_mean <= self.threshold:
                # below_threshold_count += 1  # 计数加 1
                # 如果梯度均值小于阈值，保持正常学习率
                for n, p in self.trainable_dict.items():
                    if unique_layer_name == '.'.join(n.split('.')[:-1]):  # 匹配唯一层名
                        params.append({
                            "params": p,
                            "lr": config.learning_rate,
                            "momentum": config.momentum,
                            "weight_decay": config.weight_decay
                        })
                        OptimizerLayer.append(n)
            else:
                # 如果梯度均值大于阈值，将该层学习率设为 0
                for n, p in self.trainable_dict.items():
                    if unique_layer_name == '.'.join(n.split('.')[:-1]):  # 匹配唯一层名
                        params.append({
                            "params": p,
                            "lr": config.learning_rate * 0,
                            "momentum": config.momentum,
                            "weight_decay": config.weight_decay
                        })

        self.optimizer1 = torch.optim.SGD(params)
        self.optimizer1.zero_grad()
        aa=entropy_(logits)
        maskaa=aa<0.8
        loss1=consistency(logits, logits_aug)
        
        loss=loss1[maskaa].mean()
        loss.backward()
        self.optimizer1.step()

        alpha_teacher=0.999
        if True:
             for nm, m in self.model.named_modules():
                 for npp, p in m.named_parameters():
                    if npp in OptimizerLayer and p.requires_grad:
                          mask = (torch.rand(p.shape) < 0.01).float().cuda().to(p.device)
                          with torch.no_grad():
                            p.data = self.model_state[f"{npp}"] * mask*alpha_teacher+p*mask*(1-alpha_teacher) + p * (1. - mask)

        torch.cuda.empty_cache()

        return logits,None

@torch.jit.script
def consistency(x: torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    """Consistency loss between two softmax distributions."""
    # return -(x.softmax(1) * y.log_softmax(1)).sum(1).mean()
    return -(x.softmax(1) * y.log_softmax(1)).sum(1)

def entropy_(outputs, e_margin = 0.4):
    """Calculate entropy of the output of a batch of images.
    """
    entropys = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
    return entropys

def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:#isinstance(m, nn.BatchNorm2d): collect all 
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    print(nm, np)
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy

    model.eval()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.train()
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.train()
            m.requires_grad_(True)
        else:
            m.requires_grad_(True)
    return model

def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
