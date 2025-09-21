import torch

def EMD2_loss(y, p):

    loss = torch.cumsum(y, dim=-1) - torch.cumsum(p, dim=-1)
    loss = torch.norm(loss, p=2, dim=-1)
    loss = torch.mean(loss**2)

    return loss


def EMD1_loss(y, p):

    cdf_y = torch.cumsum(y, dim=1)
    cdf_p = torch.cumsum(p, dim=1)
    loss = cdf_y - cdf_p
    loss = torch.abs(loss) / cdf_y
    loss = loss.sum(dim=1)
    loss = torch.mean(loss)

    return loss


def UCE_loss(y, p):

    loss = -torch.sum(y * torch.log(p), dim=1)
    return loss.mean()


def UEMD2_loss(y, p, beta):

    cs_p = torch.cumsum(p, dim=1)
    cs_y = torch.cumsum(y, dim=1)
    cs_beta = torch.cumsum(beta, dim=1)
    beta_prior = torch.ones_like(cs_beta)
    beta_sum = beta.sum(dim=1)

    beta_part = (cs_beta + beta_prior) / (beta_sum + 1).unsqueeze(1)

    eta = -2 * torch.bmm(cs_p.unsqueeze(1), cs_y.unsqueeze(2)) + torch.bmm(cs_y.unsqueeze(1), cs_y.unsqueeze(2))

    loss = torch.bmm(cs_p.unsqueeze(1), beta_part.unsqueeze(2)) + eta
    
    return loss.mean()