import torch
import torch.nn as nn

class PGD_v2:
    def __init__(self, eps=8/255, alpha=2/255, steps=10, random_start=True, loss_fn=None):
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

    def __call__(self, model, x, y):
        x_adv = x.clone().detach()

        if self.random_start:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(
                -self.eps, self.eps
            )
            x_adv = torch.clamp(x_adv, min=0, max=1).detach()

        for i in range(self.steps):
            x_adv.requires_grad = True

            with torch.enable_grad():
                logits = model(x_adv)
                loss = self.loss_fn(logits, y)
            if i == 0:
                # 第 0 步（第一次迭代），保存模型对初始扰动样本的输出 logit
                logits_orig = logits.detach()

            grad = torch.autograd.grad(loss, x_adv, create_graph=False)[0]
            x_adv = x_adv.detach() + self.alpha * torch.sign(grad)
            # torch.clamp = 把张量里的所有数值，强行限制在 [min, max] 范围内
            # 小于 min 的数 → 变成 min
            # 大于 max 的数 → 变成 max
            # 在中间的数 → 保持不变
            x_adv = torch.clamp(x_adv, min=x - self.eps, max=x + self.eps)
            x_adv = torch.clamp(x_adv, min=0, max=1)

        return x_adv.detach(), logits_orig


class PGD:
    def __init__(self, eps=8/255, alpha=2/255, steps=10, random_start=True, loss_fn=None):
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

    def __call__(self, model, x, y):
        x_adv = x.clone().detach()

        if self.random_start:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(
                -self.eps, self.eps
            )
            x_adv = torch.clamp(x_adv, min=0, max=1).detach()

        for _ in range(self.steps):
            x_adv.requires_grad = True

            with torch.enable_grad():
                logits = model(x_adv)
                loss = self.loss_fn(logits, y)

            grad = torch.autograd.grad(loss, x_adv, create_graph=False)[0]
            x_adv = x_adv.detach() + self.alpha * torch.sign(grad)
            x_adv = torch.clamp(x_adv, min=x - self.eps, max=x + self.eps)
            x_adv = torch.clamp(x_adv, min=0, max=1)

        return x_adv.detach()
