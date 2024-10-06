import torch
from tqdm import tqdm
from Utility.SamplePairing import SamplePairing, CrossSamplePairing
import numpy as np

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

def train_epoch_dem(model, device, dataloader, loss_fn, optimiser):
    train_loss = 0.0
    train_correct = 0
    model.train()

    for i, (inputs, label, id) in enumerate(tqdm(dataloader)):
        inputs, label, id = inputs.to(device), label.to(device), id.to(device)
        label = label.to(torch.int64)

        optimiser.zero_grad()
        output_d, output_p, output_s, output_c = model(inputs)
        loss_recon = loss_fn["recon_criterion"](output_d, inputs)
        achor_p, positive_p, negative_p = SamplePairing(label, id, output_p)
        loss_trip_p = loss_fn["trip_criterion"](achor_p, positive_p, negative_p)
        achor_s, positive_s, negative_s = SamplePairing(id, label, output_s)
        loss_trip_s = loss_fn["trip_criterion"](achor_s, positive_s, negative_s)
        target, component_p, component_s = CrossSamplePairing(label, id, inputs)
        intermediates_p = model.module.encoderp(component_p)
        intermediates_s = model.module.encoders(component_s)
        combination = model.module.decoder(torch.concatenate((intermediates_p, intermediates_s), axis=1))
        loss_cross_recon = loss_fn["recon_criterion"](combination, target)
        loss_clf = loss_fn["clf_criterion"](output_c, label)

        loss = loss_recon + 0.5*loss_trip_p + 0.5*loss_trip_s + loss_cross_recon + loss_clf
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        train_loss += loss.item() * inputs.size(0)
        _, prediction = torch.max(output_c.data, -1)
        train_correct += (prediction == label).sum().item()

    return train_loss, train_correct

def train_epoch_vae(model, device, dataloader, loss_re, loss_cl, optimiser):
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

def train_meta_epoch(model, device, dataloader, loss_fn, meta_optimiser, inner_steps=1, inner_lr=0.01):
    train_loss = 0.0
    train_correct = 0
    model.train()

    meta_optimiser.zero_grad()

    for task_id in np.unique(dataloader.dataset.win_lb): #set(dataloader.dataset.win_lb)  # 假设dataloader.dataset.targets包含所有id
        task_loss = 0.0
        task_correct = 0
        task_samples = [(inputs, label) for inputs, label, id in dataloader if id == [task_id]]

        # 内循环：在当前任务上进行多步梯度下降
        for step in range(inner_steps):
            for inputs, label in task_samples:
                inputs, label = inputs.to(device), label.to(device)
                label = label.to(torch.int64)

                outputs = model(inputs)
                loss = loss_fn(outputs, label)
                loss.backward()

                # 使用SGD优化器进行任务特定更新
                for param in model.parameters():
                    param.data -= inner_lr * param.grad
                    param.grad = None

        # 计算内循环结束后的梯度并累加到元梯度
        for inputs, label in task_samples:
            inputs, label = inputs.to(device), label.to(device)
            label = label.to(torch.int64)

            outputs = model(inputs)
            loss = loss_fn(outputs, label)
            task_loss += loss.item() * inputs.size(0)
            loss.backward()

            _, prediction = torch.max(outputs.data, -1)
            task_correct += (prediction == label).sum().item()

        # 累积任务特定的损失和正确数
        train_loss += task_loss
        train_correct += task_correct

    # 外循环：使用累积的梯度更新元参数
    meta_optimiser.step()

    return train_loss, train_correct
