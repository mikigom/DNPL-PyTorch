import torch
from torch import nn as nn
from torch.autograd import Variable


def to_torch_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    def set_param(self, curr_mod=None, name=None, param=None):
        if curr_mod is None:
            curr_mod = self

        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def named_params(self, current_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if current_module is None:
            current_module = self

        if hasattr(current_module, 'named_leaves'):
            for name, p, in current_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in current_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in current_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_torch_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(name=name_t, param=tmp)
        else:
            for name, param in self.named_params():
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_torch_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(name=name, param=tmp)
                else:
                    param = param.detach_()
                    self.set_param(name=name, param=param)

    def params(self):
        for name, param in self.named_params():
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_torch_var(param.data.clone(), requires_grad=True)
            self.set_param(name=name, param=param)