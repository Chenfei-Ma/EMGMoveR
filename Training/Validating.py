import torch
from tqdm import tqdm
from Utility.SamplePairing import SamplePairing, CrossSamplePairing

def valid_epoch(model, device, dataloader, loss_fn):
    valid_loss = 0.0
    valid_correct = 0
    model.eval()

    with torch.no_grad():
        for i, (inputs, label) in enumerate(tqdm(dataloader)):
            inputs, label = inputs.to(device), label.to(device)
            label = label.to(torch.int64)

            outputs = model(inputs)
            label = torch.squeeze(label)
            loss = loss_fn(outputs, label)
            valid_loss += loss.item() * inputs.size(0)
            _, prediction = torch.max(outputs.data, 1)
            valid_correct += (prediction == label).sum().item()

    return valid_loss, valid_correct

def valid_epoch_dem(model, device, dataloader, loss_fn):
    valid_loss = 0.0
    valid_correct = 0
    model.eval()

    with torch.no_grad():
        for i, (inputs, label, id) in enumerate(tqdm(dataloader)):
            inputs, label, id = inputs.to(device), label.to(device), id.to(device)
            label = label.to(torch.int64)

            output_d, output_p, output_s, output_c = model(inputs)
            loss_recon = loss_fn["recon_criterion"](output_d, inputs)
            #
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
            # loss_clf = loss_fn["clf_criterion"](output_c, label)

            loss = loss_recon # + 0.5 * loss_trip_p + 0.5 * loss_trip_s + loss_cross_recon + loss_clf

            valid_loss += loss.item() * inputs.size(0)
            _, prediction = torch.max(output_c.data, 1)
            valid_correct += (prediction == label).sum().item()

    return valid_loss, valid_correct
