import torch


class Config:
    height = 512
    width = 512
    train_num = 1300

config = Config()


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def tensor_normalize(feat, mean, std):
    '''
    :param feat:  N x C x H x W
    :param mean:  1 dimension as C
    :param std: 1 dimension as C
    :return: normalized N x C x H x W
    '''
    mean = torch.as_tensor(mean, dtype=torch.float32, device=feat.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=feat.device)
    feat = feat.sub(mean[None, :, None, None]).div(std[None, :, None, None])
    return feat


if __name__ == '__main__':
    feat = torch.arange(1, 97).reshape((2, 3, 4, 4))
    mean = (0.5, 0.5, 0.3)
    std = (0.1, 4, 6)
    print(tensor_normalize(feat, mean, std))
