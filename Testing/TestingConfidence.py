import torch
from tqdm import tqdm
import torch.nn as nn
from Utility.SamplePairing import SamplePairing, CrossSamplePairing

def test_epoch(model, device, dataloader, loss_fn):
    test_loss = 0.0
    test_correct = 0
    model.eval()

    with torch.no_grad():
        for i, (inputs, label, id) in enumerate(tqdm(dataloader)):
            inputs, label = inputs.to(device), label.to(device)
            label = label.to(torch.int64)
            outputs = model(inputs)
            loss = loss_fn(outputs, label)
            test_loss += loss.item() * inputs.size(0)
            _, prediction = torch.max(outputs.data, dim=-1)
            test_correct += (prediction == label).sum().item()
            softmax_probs = nn.functional.softmax(outputs.data, dim=-1)
            max_prob, max_class = torch.max(softmax_probs, dim=-1)
            softmax_probs[torch.arange(dataloader.batch_size),max_class] = float('-inf')
            second_max_prob, second_max_class = torch.max(softmax_probs, dim=-1)
            prob_difference = max_prob - second_max_prob
    return test_loss, test_correct, prob_difference.sum().item()

def test_epoch_dem(model, device, dataloader, loss_fn):
    test_loss = 0.0
    test_correct = 0
    model.eval()

    with torch.no_grad():
        for i, (inputs, label, id) in enumerate(tqdm(dataloader)):
            inputs, label, id = inputs.to(device), label.to(device), id.to(device)
            label = label.to(torch.int64)
            output_d, output_p, output_s, output_c = model(inputs)
            # loss_recon = loss_fn["recon_criterion"](output_d, inputs.flatten(1,2))

            # achor_p, positive_p, negative_p = SamplePairing(label, id, output_p)
            # loss_trip_p = loss_fn["trip_criterion"](achor_p, positive_p, negative_p)
            #
            # achor_s, positive_s, negative_s = SamplePairing(id, label, output_s)
            # loss_trip_s = loss_fn["trip_criterion"](achor_s, positive_s, negative_s)
            #
            # target, component_p, component_s = CrossSamplePairing(label, id, inputs)
            # intermediates_p = model.encoderp(component_p)
            # intermediates_s = model.encoders(component_s)
            # combination = model.decoder(torch.concatenate((intermediates_p, intermediates_s), axis=1))
            # loss_cross_recon = loss_fn["recon_criterion"](combination, target)
            #
            loss_clf = loss_fn["clf_criterion"](output_c, label)

            loss = loss_clf #+ 0.5 * loss_trip_p + 0.5 * loss_trip_s + loss_cross_recon + loss_recon

            test_loss += loss.item() * inputs.size(0)
            _, prediction = torch.max(output_c.data, -1)
            test_correct += (prediction == label).sum().item()

    return test_loss, test_correct

