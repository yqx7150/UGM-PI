import torch
import torch.autograd as autograd
import random


def dsm(energy_net, samples, sigma=1):
    samples.requires_grad_(True)
    vector = torch.randn_like(samples) * sigma
    perturbed_inputs = samples + vector
    logp = -energy_net(perturbed_inputs)
    dlogp = sigma ** 2 * autograd.grad(logp.sum(), perturbed_inputs, create_graph=True)[0]
    kernel = vector
    loss = torch.norm(dlogp + kernel, dim=-1) ** 2
    loss = loss.mean() / 2.

    return loss


def dsm_score_estimation(scorenet, samples, sigma=0.01):
    perturbed_samples = samples + torch.randn_like(samples) * sigma
    target = - 1 / (sigma ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)

    return loss


def anneal_dsm_score_estimation_constraint(scorenet, samples, labels, sigmas, anneal_power=2.):
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    target1 = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
    u = []
    u.append(random.randint(0,10))
    u.append(u[0]+1)
    
    for vv,i in enumerate(u):
        scores = scorenet(perturbed_samples[...,i], labels)
        target = target1[...,i]
        target = target.view(target[...,i].shape[0], -1)
        scores = scores.view(scores.shape[0], -1)
        if vv==0:
            loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
        else:
            u = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
            loss = 0.5 * loss + (1-0.5)*u
        # if i == 0:
        #     scores = scorenet(perturbed_samples[...,i], labels)
        #     target = target1[...,i]
        #     target = target.view(target[...,i].shape[0], -1)
        #     scores = scores.view(scores.shape[0], -1)
        #     if i==0:
        #         loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
        #     else:
        #         u = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
        #         loss = 0.5 * loss + u
        # else:
        #     with torch.no_grad():
        #         scores = scorenet(perturbed_samples[...,i], labels)
        #         target = target1[...,i]
        #         target = target.view(target[...,i].shape[0], -1)
        #         scores = scores.view(scores.shape[0], -1)
        #         if i==0:
        #             loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
        #         else:
        #             u = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
        #             loss = 0.5 * loss + u

    return loss.mean(dim=0)


def anneal_dsm_score_estimation_modify(scorenet, samples, labels, sigmas, anneal_power=2.):
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    target1 = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
    u = random.randint(0,10)

    scores = scorenet(perturbed_samples[...,u], labels)
    target = target1[...,u]
    target = target.view(target[...,u].shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
        
    return loss.mean(dim=0)


def anneal_dsm_score_estimation(scorenet, samples, labels, sigmas, anneal_power=2.):
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    return loss.mean(dim=0)
