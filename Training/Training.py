import torch
from tqdm import tqdm
from Utility.SamplePairing import SamplePairing, CrossSamplePairing

def train_epoch(model, device, dataloader, loss_fn, optimiser):
    train_loss = 0.0
    train_correct = 0
    model.train()

    for i, (inputs, label, id) in enumerate(tqdm(dataloader)):
        inputs, label = inputs.to(device), label.to(device)
        label = label.to(torch.int64)

        optimiser.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, label)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        train_loss += loss.item() * inputs.size(0)
        _, prediction = torch.max(outputs.data, -1)
        train_correct += (prediction == label).sum().item()

    return train_loss, train_correct

def train_epoch_cl(model, device, dataloader, loss_fn, optimiser):
    train_loss = 0.0
    train_correct = 0
    model.train()

    for i, (inputs, label, id) in enumerate(tqdm(dataloader)):
        inputs, label, id = inputs.to(device), label.to(device), id.to(device)
        label = label.to(torch.int64)

        optimiser.zero_grad()
        output_d, output_p, output_s, output_c = model(inputs)

        loss_recon = loss_fn["recon_criterion"](output_d, inputs.flatten(1,2))
        achor_p, positive_p, negative_p = SamplePairing(label, id, output_p)
        loss_trip_p = loss_fn["trip_criterion"](achor_p, positive_p, negative_p)
        achor_s, positive_s, negative_s = SamplePairing(id, label, output_s)
        loss_trip_s = loss_fn["trip_criterion"](achor_s, positive_s, negative_s)
        target, component_p, component_s = CrossSamplePairing(label, id, inputs.flatten(1,2))
        intermediates_p = model.module.encoderp(component_p)
        intermediates_s = model.module.encoders(component_s)
        combination = model.module.decoder(torch.concatenate((intermediates_p, intermediates_s), axis=1))
        loss_cross_recon = loss_fn["recon_criterion"](combination, target)
        loss_clf = loss_fn["clf_criterion"](output_c, label)

        loss = loss_recon + 0.75*loss_trip_p + 0.75*loss_trip_s + loss_cross_recon + loss_clf
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        train_loss += loss.item() * inputs.size(0)
        _, prediction = torch.max(output_c.data, -1)
        train_correct += (prediction == label).sum().item()

    return train_loss, train_correct

def train_epoch_ae(model, device, dataloader, loss_re, loss_cl, optimiser):
    train_loss = 0.0
    train_correct = 0
    model.train()

    for i, (inputs, label) in enumerate(tqdm(dataloader)):
        inputs, label = inputs.to(device), label.to(device)
        label = label.to(torch.int64)
        optimiser.zero_grad()

        reconstructed, mu, logvar, classification = model(inputs)
        loss_r = loss_re(reconstructed, inputs)
        loss_c = loss_cl(classification, label)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = loss_r + KLD + loss_c
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        train_loss += total_loss.item() * inputs.size(0)
        _, prediction = torch.max(classification.data, -1)
        train_correct += (prediction == label).sum().item()

    return train_loss, train_correct

