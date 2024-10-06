from argparse import ArgumentParser
from configparser import ConfigParser
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import wandb
import uuid
from Database.SubDataset import SubsetDataset

if __name__ == '__main__':

    # region AP setup
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, help='select model', required=True)
    parser.add_argument('--subject', type=str, help='select subject', required=True)
    parser.add_argument('--prep', type=str, help='select preprocessing', required=True)
    args = parser.parse_args()
    MODEL_NAME = args.model
    DATASET_NAME = args.datasets
    # endregion AP setup

    # region CP setup
    cp_datasets = ConfigParser()
    cp_datasets.read(os.path.dirname(os.path.realpath(__file__)) + '/Configs/Database/' + args.datasets + '.ini')
    ROOT_DIR = cp_datasets.get('dataloader', 'root_dir')
    SAVE_DIR = ROOT_DIR + cp_datasets.get('dataloader', 'save_dir')
    BUF_DIR = ROOT_DIR + cp_datasets.get('dataloader', 'buf_dir')
    dict_str = cp_datasets.get('dataloader', 'db')
    dict_config = dict(line.split(': ') for line in dict_str.split('\n') if line)
    for key in dict_config:
        dict_config[key] = [x for x in dict_config[key].strip(',').split(',')]
    DB = dict_config
    MOVEMENTS = list((cp_datasets.get('dataloader', 'movements').split(',')))
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    cp_prep = ConfigParser()
    cp_prep.read(os.path.dirname(os.path.realpath(__file__)) + '/Configs/Preprocessing/' + args.prep + '.ini')
    FEATURE_WIN_SIZE = cp_prep.getint('preprocessing', 'feature_win_size')
    FEATURE_WIN_STRIDE = cp_prep.getint('preprocessing', 'feature_win_stride')
    LOW_CUT = cp_prep.getint('preprocessing', 'low_cut')
    HIGH_CUT = cp_prep.getint('preprocessing', 'high_cut')
    FILTER_ORDER = cp_prep.getint('preprocessing', 'filter_order')
    FEATURE_SET = list((cp_prep.get('preprocessing', 'feature_set').split(',')))

    cp_model = ConfigParser()
    cp_model.read(os.path.dirname(os.path.realpath(__file__))+'/Configs/Model/'+args.model+'.ini')
    DEVICE_IDS = list(map(int, (cp_model.get('training', 'device_ids').split(','))))
    EPOCH_NUM = cp_model.getint('training', 'epoch_num')
    BATCH_SIZE = cp_model.getint('training', 'batch_size')
    TEST_BATCH_SIZE = cp_model.getint('training', 'test_batch_size')
    SLIDING_WIN_SIZE = cp_model.getint('training', 'win_size')
    SLIDING_WIN_STRIDE = cp_model.getint('training', 'win_stride')
    LR = cp_model.getfloat('training', 'lr')
    LR_STEP_SIZE = cp_model.getint('training', 'lr_step_size')
    LR_GAMMA = cp_model.getfloat('training', 'gamma')
    WEIGHT_DECAY = cp_model.getfloat('training', 'weight_decay')

    INPUT_LENGTH = cp_model.getint('network', 'input_length')
    INPUT_CHANNEL = cp_model.getint('network', 'input_channel')
    HIDDEN_LAYERS = list(map(int, (cp_model.get('network', 'hidden_layers').split(','))))
    KERNEL_SIZE = cp_model.getint('network', 'kernel_size')
    DROP_PROB = cp_model.getfloat('network', 'drop_prob')

    if MODEL_NAME == 'Disentanglement':
        from Models.Disentanglement import DEM as selected_model
        from Training.Training import train_epoch_dem as train
        from Testing.Testing import test_epoch_dem as test
        criterion = {
            "recon_criterion": nn.MSELoss().cuda(),
            "trip_criterion": nn.TripletMarginLoss(margin=0.3).cuda(),
            "clf_criterion": torch.nn.CrossEntropyLoss().cuda(),
        }

    elif MODEL_NAME == 'AttentionTCN':
        from Models.AttentionTCN import AttentionTCN as selected_model
        from Training.Training import train_epoch as train
        from Testing.Testing import test_epoch as test
        criterion = nn.CrossEntropyLoss().cuda()

    elif MODEL_NAME == 'TCN':
        from Models.TCN import TCN as selected_model
        from Training.Training import train_epoch as train
        from Testing.Testing import test_epoch as test
        criterion = nn.CrossEntropyLoss().cuda()

    elif MODEL_NAME == 'MetaTCN':
        from Models.TCN import TCN as selected_model
        from Training.Training import train_meta_epoch as train
        from Testing.Testing import test_epoch as test
        criterion = nn.CrossEntropyLoss().cuda()

    else:
        raise Exception("Unknown Model")
    # endregion

    # region Torch setup
    torch.manual_seed(7)
    torch.set_default_dtype(torch.float64)
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(DEVICE_IDS[0]))
    else:
        device = torch.device('cpu')
    # endregion torch setup

    # region DB validating
    if DATASET_NAME == "MoveR":
        from Database.MoveRDataset import Dataset
        print('VALIDATING DATASET')
        pre_load_dataset = Dataset(root_dir=ROOT_DIR,
                                buf_dir=BUF_DIR,
                                db_subjects=DB,
                                movements=MOVEMENTS,
                                feature_set=FEATURE_SET,
                                feature_win_size=FEATURE_WIN_SIZE,
                                feature_win_stride=FEATURE_WIN_STRIDE,
                                sliding_win_size=SLIDING_WIN_SIZE,
                                sliding_win_stride=SLIDING_WIN_STRIDE,
                                low_cut=LOW_CUT,
                                high_cut=HIGH_CUT,
                                filter_order=FILTER_ORDER)
        del pre_load_dataset
    # endregion

    for key, int_list in DB.items():

        for value in int_list:

            DB_TRAIN = DB.copy()
            DB_TRAIN[key] = [v for v in DB_TRAIN[key] if v != value]
            DB_TEST = {key:[value]}

            # region load the training dataset
            train_dataset = Dataset(root_dir=ROOT_DIR,
                                    buf_dir=BUF_DIR,
                                    db_subjects=DB_TRAIN,
                                    movements=MOVEMENTS,
                                    feature_set=FEATURE_SET,
                                    feature_win_size=FEATURE_WIN_SIZE,
                                    feature_win_stride=FEATURE_WIN_STRIDE,
                                    sliding_win_size=SLIDING_WIN_SIZE,
                                    sliding_win_stride=SLIDING_WIN_STRIDE,
                                    low_cut=LOW_CUT,
                                    high_cut=HIGH_CUT,
                                    filter_order=FILTER_ORDER)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, pin_memory=True, prefetch_factor=2)
            # endregion

            # region load the test dataset
            valid_test_dataset = Dataset(root_dir=ROOT_DIR,
                                   buf_dir=BUF_DIR,
                                   db_subjects=DB_TEST,
                                   movements=MOVEMENTS,
                                   feature_set=FEATURE_SET,
                                   feature_win_size=FEATURE_WIN_SIZE,
                                   feature_win_stride=FEATURE_WIN_STRIDE,
                                   sliding_win_size=SLIDING_WIN_SIZE,
                                   sliding_win_stride=SLIDING_WIN_STRIDE,
                                   low_cut=LOW_CUT,
                                   high_cut=HIGH_CUT,
                                   filter_order=FILTER_ORDER)
            valid_dataset = SubsetDataset(valid_test_dataset, [0, 1, 2, 3, 4, 5],[0, 1/10])
            valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                       batch_size=TEST_BATCH_SIZE,
                                                       shuffle=True,
                                                       drop_last=True,
                                                       num_workers=4,
                                                       pin_memory=True,
                                                       prefetch_factor=2)
            test_dataset = SubsetDataset(valid_test_dataset, [0, 1, 2, 3, 4, 5],[1/10, 1])
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                      batch_size=TEST_BATCH_SIZE,
                                                      shuffle=False,
                                                      drop_last=False,
                                                      num_workers=4,
                                                      pin_memory=True,
                                                      prefetch_factor=2)
            # endregion

            # region Model loading
            model = selected_model(input_length=SLIDING_WIN_SIZE, input_channels=INPUT_CHANNEL, kernel_size=KERNEL_SIZE, hidden_layers=HIDDEN_LAYERS, dropout=DROP_PROB)
            model = model.to(device)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model, device_ids=DEVICE_IDS)
            # endregion

            # region optimiser
            optimiser = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = lr_scheduler.StepLR(optimiser, LR_STEP_SIZE, LR_GAMMA)
            # endregion

            # region wandb init
            RUNNAME = str(uuid.uuid4())
            CONFIG = {**cp_datasets.__dict__['_sections'].copy(), **cp_prep.__dict__['_sections'].copy(), **cp_model.__dict__['_sections'].copy()}
            CONFIG.update({'runsinfo' : {'test_subject': key+'-'+str(value),
                                        'model_name': MODEL_NAME,
                                        }})
            wandb.init(project="MoveR", name=RUNNAME, config=CONFIG)
            # endregion

            # region Training & Validation
            best_record = {'train_loss': 10, 'train_acc': 0, 'valid_loss': 10, 'valid_acc': 0}
            for epoch in range(EPOCH_NUM):
                train_loss, train_acc = train(model, device, train_loader, criterion, optimiser)
                train_loss = train_loss / (len(train_loader) * BATCH_SIZE)
                train_acc = train_acc / (len(train_loader) * BATCH_SIZE)
                print("Epoch:{}/{} AVG Training Loss:{:.3f} Acc {:.2f} ".format(epoch + 1, EPOCH_NUM, train_loss, train_acc))
                wandb.log({"train_acc": train_acc, "train_loss": train_loss})

                valid_loss, valid_acc = test(model, device, valid_loader, criterion)
                valid_loss = valid_loss / (len(valid_loader) * TEST_BATCH_SIZE)
                valid_acc = valid_acc / (len(valid_loader) * TEST_BATCH_SIZE)
                print("Epoch:{}/{} AVG Validation Loss:{:.3f} Acc {:.2f} ".format(epoch + 1, EPOCH_NUM, valid_loss, valid_acc))
                wandb.log({"valid_acc": valid_acc, "valid_loss": valid_loss})

                if valid_acc > best_record['valid_acc'] and valid_loss <= best_record['valid_loss']:
                    best_record['train_acc'] = train_acc
                    best_record['train_loss'] = train_loss
                    best_record['valid_acc'] = valid_acc
                    best_record['valid_loss'] = valid_loss
                    torch.save(model.state_dict(), SAVE_DIR + RUNNAME + f'_pretraining.pth')
                    print('- pretrained model saved')
                scheduler.step()
            print("Best Train Loss: {:.4f} Train Acc: {:.3f} Valid Loss: {:.4f} Valid Acc: {:.3f}".format(
                best_record['train_loss'], best_record['train_acc'], best_record['valid_loss'], best_record['valid_acc']))
            #endregion

            # region Model reloading
            model = selected_model(input_length=SLIDING_WIN_SIZE, input_channels=INPUT_CHANNEL, kernel_size=KERNEL_SIZE, hidden_layers=HIDDEN_LAYERS, dropout=DROP_PROB)
            model = model.to(device)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model, device_ids=DEVICE_IDS)

            if os.path.exists(SAVE_DIR + RUNNAME + f'_pretraining.pth'):
                model.load_state_dict(torch.load(SAVE_DIR + RUNNAME + f'_pretraining.pth', map_location=device))
                print(f'pretraining model loaded')
            else:
                raise OSError('There is no pretraining model exisiting at' + SAVE_DIR + ' with run name ' + RUNNAME)
            # endregion

            # region Test
            test_loss, test_acc = test(model, device, test_loader, criterion)
            test_loss = test_loss / (len(test_loader) * TEST_BATCH_SIZE)
            test_acc = test_acc / (len(test_loader) * TEST_BATCH_SIZE)
            print("Test Loss: {:.4f} Test Acc: {:.3f}".format(test_loss, test_acc))
            wandb.log({"test_acc": test_acc, "test_loss": test_loss})
            # endregion

            # region wandb ends
            wandb.finish()
            # endregion


