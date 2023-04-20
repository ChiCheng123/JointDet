import torch

def accuracy(output, target, topk=(1, ), ignore_index=-1):
    """Computes the precision@k for the specified values of k"""
    keep = torch.nonzero(target != ignore_index).squeeze()
    if keep.dim() <= 0:
        return [torch.cuda.FloatTensor([1]).zero_()]
    assert (keep.dim() == 1)
    target = target[keep]
    output = output[keep]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res