from torch.func import grad, jvp, vmap
from .model_wrapper import TransformerWrapper
import torch


def jvp_single(input_ids, attention_mask, labels, loss_mask, model: TransformerWrapper, lam_param, params, buffers):
    loss_single_func_wrapper = lambda params: \
        model.compute_loss_func_single(
            params, buffers, model,
            input_ids, attention_mask, labels, loss_mask)
    _, ct = jvp(loss_single_func_wrapper, (params,), (lam_param,))

    return ct


def jvp_single_grad(input_ids, attention_mask, labels, loss_mask, model: TransformerWrapper, lam_param, params, buffers, grad_checkpoint):
    loss_single_func_wrapper = lambda params: \
        model.compute_loss_func_single(
            params, buffers, model,
            input_ids, attention_mask, labels, loss_mask)

    grad_params = grad(loss_single_func_wrapper)(params)
    grad_flat = torch.cat([g.view(-1) for g in grad_params.values()])
    lam_flat = torch.cat([v.view(-1) for v in lam_param.values()])

    ct_manual = torch.dot(grad_flat, lam_flat)

    grad_norm = torch.norm(grad_flat)
    lam_norm = torch.norm(lam_flat)
    cos_theta = ct_manual / (grad_norm * lam_norm)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  
    # theta = torch.acos(cos_theta) 

    # total_ct_cos = lam_norm * (grad_checkpoint / grad_norm) * cos_theta
    total_ct_cos = lam_norm * (grad_norm / grad_checkpoint) * cos_theta

    return total_ct_cos, grad_norm


def jvp_batch(model: TransformerWrapper, batch, lam_param, params, buffers, chunk_size=None, grad_checkpoint=None):
    return vmap(jvp_single_grad, in_dims=(0, 0, 0, 0, None, None, None, None, None), chunk_size=chunk_size)(
        batch["input_ids"], batch["attention_mask"], batch["label"], batch["loss_mask"], model, lam_param, params, buffers, grad_checkpoint)


def hvp_fwdrev(model: TransformerWrapper, batch, lam_param, params, buffers, bs, gacc, ws):
    f = model.compute_loss_func
    def grad_wrapper(pr):
        g = {n: 0 for n in params}
        for i in range(gacc):
            mini_batch = {k: v[i*bs:(i+1)*bs] for k, v in batch.items()}
            _g = grad(f)(pr, buffers, model, **mini_batch)
            for n in g:
                g[n] += _g[n]
        return g
    _, hvp_res = jvp(grad_wrapper, (params,), (lam_param,))
    hvp_res = model.params_to_vector(hvp_res) / (ws * gacc)
    return hvp_res
