import torch

def SamplePairing(label_a, label_b, encoder_output):
    achor = []
    positive = []
    negative = []

    for i in torch.unique(torch.unique(label_a)):
        for j in torch.unique(label_b.data):
            index_a = ((label_a == i) * (label_b == j)).nonzero().squeeze(dim=-1)
            index_p = ((label_a == i) * (label_b != j)).nonzero().squeeze(dim=-1)
            index_n = ((label_a != i)).nonzero().squeeze(dim=-1)

            if index_a.dim() > 0:
                pair_no = min(len(index_a), len(index_p), len(index_n))
                if pair_no == 0:
                    continue

                index_a = index_a[0:pair_no].squeeze(dim=-1)
                index_p = index_p[0:pair_no].squeeze(dim=-1)
                index_n = index_n[0:pair_no].squeeze(dim=-1)

                if pair_no == 1:
                    a = encoder_output[index_a].unsqueeze(dim=0)
                    p = encoder_output[index_p].unsqueeze(dim=0)
                    n = encoder_output[index_n].unsqueeze(dim=0)
                else:
                    a = encoder_output[index_a]
                    p = encoder_output[index_p]
                    n = encoder_output[index_n]

                achor.append(a)
                positive.append(p)
                negative.append(n)

    achor = torch.cat(achor, dim=0)
    positive = torch.cat(positive, dim=0)
    negative = torch.cat(negative, dim=0)

    return achor, positive, negative

def CrossSamplePairing(label_p, label_s, origin_input):
    target = []
    component_p = []
    component_s = []

    for i in torch.unique(torch.unique(label_p)):
        for j in torch.unique(label_s.data):
            index_t = ((label_p == i) * (label_s == j)).nonzero().squeeze(dim=-1)
            index_p = ((label_p == i) * (label_s != j)).nonzero().squeeze(dim=-1)
            index_s = ((label_p != i) * (label_s == j)).nonzero().squeeze(dim=-1)

            if index_t.dim() > 0:

                index_p = index_p[torch.randperm(len(index_p))]
                index_s = index_s[torch.randperm(len(index_s))]

                pair_no = min(len(index_t), len(index_p), len(index_s))
                if pair_no < 2:
                    continue

                index_t = index_t[0:pair_no]
                index_p = index_p[0:pair_no]
                index_s = index_s[0:pair_no]

                t = origin_input[index_t]
                p = origin_input[index_p]
                s = origin_input[index_s]

                target.append(t)
                component_p.append(p)
                component_s.append(s)

    target = torch.cat(target, dim=0)
    component_p = torch.cat(component_p, dim=0)
    component_s = torch.cat(component_s, dim=0)

    return target, component_p, component_s