from torch.utils.data import Dataset, DataLoader
import os, pickle
from scipy.signal import resample
from Preprocessing.Filtering import EMGFilter
from Preprocessing.Features import *
from Preprocessing.Augmentation import Densifing
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

class Dataset(Dataset):
    def __init__(self, root_dir, buf_dir, db_subjects, movements, feature_set, feature_win_size, feature_win_stride, sliding_win_size, sliding_win_stride, low_cut, high_cut, filter_order):

        if not os.path.exists(buf_dir):
            os.mkdir(buf_dir)
        self.features = []
        self.labels = []
        self.ids = []
        self.sliding_win_size = sliding_win_size
        self.sliding_win_stride = sliding_win_stride

        db_list = list(db_subjects.keys())
        sub_ra = 0

        for db_name in db_list:
            print(f'Database {db_name}')
            db_dir = root_dir + db_name + '/'
            db_buf_dir = buf_dir + db_name + '/'
            if not os.path.exists(db_buf_dir):
                os.mkdir(db_buf_dir)

            sublist = os.listdir(db_dir)
            sublist = [f for f in sublist if not f.startswith('.') and f in db_subjects[db_name]]
            sublist.sort(key=lambda x: int(x))
            if sublist != db_subjects[db_name]:
                raise TypeError('Some subjects do not match with the database.')

            db_features = []
            db_labels = []
            db_ids = []

            for sub in sublist:
                print(f'-subject {sub}')
                sub_dir = db_dir + '{:s}/'.format(sub)
                sub_buf_dir = db_buf_dir + sub + '/'
                if not os.path.exists(sub_buf_dir):
                    os.mkdir(sub_buf_dir)
                buf_file_name = "{:s}_{:s}_{:d}_{:d}_{:d}_{:d}_{:d}.pickle".format(str(movements), str(feature_set), feature_win_size, feature_win_stride, low_cut, high_cut, filter_order)
                buf_file_dir = sub_buf_dir + buf_file_name

                if os.path.exists(buf_file_dir):
                    with open(buf_file_dir, 'rb') as handle:
                        sub_features = pickle.load(handle)
                        sub_labels = pickle.load(handle)
                        sub_ids = pickle.load(handle)
                        print('--feature loaded')

                else:
                    blocklist = os.listdir(sub_dir)
                    blocklist = [f for f in blocklist if not f.startswith('.') and ('test' in f or 'calibration' in f)]

                    sub_emg = []
                    sub_lb = []

                    for blk in blocklist:
                        block_dir = sub_dir + blk + '/'
                        if db_name == 'Database_VFN':
                            if blk == 'calibration':
                                block_emg = np.load(block_dir + 'data.npy')[:, :, -4000:-2000]
                            else:
                                continue
                        elif db_name == 'Database_SCE' or db_name == 'Database_LP':
                            # block_emg = np.load(block_dir + 'data.npy')[:, np.r_[0:10, 11:16], -2000:]   #[0,8,1,9,2,3,11,4,12,5,13,6,14,7,15]
                            block_emg = np.load(block_dir + 'data.npy')[:, [0,8,1,9,2,3,11,4,12,5,13,6,14,7,15], -2000:]
                        else:
                            block_emg = np.load(block_dir + 'data.npy')[:, :, -2000:]
                        block_lb = np.load(block_dir + 'label.npy')

                        if 6 in block_lb:
                            block_lb[block_lb == 6] = 0

                        sub_emg.append(block_emg.transpose([1, 0, 2]).reshape(block_emg.shape[1], -1))
                        sub_lb.append(np.repeat(block_lb, 2000))

                    sub_emg = np.hstack(sub_emg.copy())
                    sub_lb = np.hstack(sub_lb.copy())
                    original_channels = np.linspace(0, 1, num=sub_emg.shape[0])
                    target_channels = np.linspace(0, 1, num=21)
                    interpolated_data = np.array([np.interp(target_channels, original_channels, sub_emg[:, t]) for t in range(sub_emg.shape[1])])
                    sub_emg = interpolated_data.T
                    sub_emg = EMGFilter(sub_emg, low_cut, high_cut, filter_order)
                    print('--data selected')

                    # from Utility.ChannelPlot import singleplot
                    # from matplotlib import pyplot as plt
                    # singleplot(sub_emg,sub_lb,2000,str(sub))
                    # plt.show()

                    sub_emg = resample(sub_emg, sub_emg.shape[1]//2, axis=1)
                    sub_lb = sub_lb[::2]
                    upperbound = np.quantile(sub_emg, 0.999995, axis=-1)
                    upperbound_expanded = np.expand_dims(upperbound, axis=-1)
                    upperbound_expanded = np.repeat(upperbound_expanded, sub_emg.shape[-1], axis=-1)
                    sub_emg = np.where(sub_emg > upperbound_expanded, upperbound_expanded, sub_emg)
                    lowerbound = np.quantile(sub_emg, 0.000005)
                    lowerbound_expanded = np.expand_dims(lowerbound, axis=-1)
                    lowerbound_expanded = np.repeat(lowerbound_expanded, sub_emg.shape[-1], axis=-1)
                    sub_emg = np.where(sub_emg < lowerbound_expanded, lowerbound_expanded, sub_emg)

                    feature = []
                    label = []
                    for f in feature_set:
                        fts = []
                        lbs = []
                        for i in np.unique(sub_lb):
                            emg = np.squeeze(sub_emg[:, np.where(sub_lb == i)].copy())
                            ft = globals()[f](emg, feature_win_size, feature_win_stride)
                            lb = np.array([i] * ft.shape[-1])
                            fts.append(ft.copy())
                            lbs.append(lb.copy())
                        feature.append(Normalise(np.concatenate(fts, axis=-1)))
                        label.append(np.concatenate(lbs, axis=-1))
                    sub_features = np.concatenate(feature, axis=0).copy()
                    if all(np.array_equal(label[0], arr) for arr in label):
                        sub_labels = label[0]
                    else:
                        raise ValueError('label length does not match')
                    sub_ids = np.array([sub_ra] * sub_features.shape[-1])
                    del feature, label
                    print('--feature generated')

                    with open(buf_file_dir, 'wb') as handle:
                        pickle.dump(sub_features, handle)
                        pickle.dump(sub_labels, handle)
                        pickle.dump(sub_ids, handle)
                        print('--data saved')

                sub_ra += 1

                db_features.append(sub_features)
                db_labels.append(sub_labels)
                db_ids.append(sub_ids)

            if len(db_ids) > 0:
                db_features = np.concatenate(db_features, axis=-1)
                db_labels = np.concatenate(db_labels, axis=-1)
                db_ids = np.concatenate(db_ids, axis=-1)
            else:
                db_features = np.array([])
                db_labels = np.array([])
                db_ids = np.array([])
            # if db_features.shape[0] < 21 * len(feature_set):
            #     padding_channels = 21 * len(feature_set) - db_features.shape[0]
            #     db_features = np.pad(db_features, ((0, padding_channels), (0, 0)), mode='constant', constant_values=0)
            # else:
            #     db_features = db_features
            print('Data Concatenate')

            self.features.append(db_features)
            self.labels.append(db_labels)
            self.ids.append(db_ids)

        self.features = [f for f in self.features if f.size > 0]
        self.ids = [i for i in self.ids if i.size > 0]
        self.labels = [l for l in self.labels if l.size > 0]
        self.features = np.concatenate(self.features, axis=-1)
        self.ids = np.concatenate(self.ids, axis=-1)
        self.labels = np.concatenate(self.labels, axis=-1)
        print('Data Completed')

        if self.sliding_win_size == self.sliding_win_stride == 1:
            self.win_idx = np.column_stack((np.arange(self.features.shape[-1]-1), np.arange(1, self.features.shape[-1])))#[[i, i + 1] for i in range(self.features.shape[-1]-1)]
            self.win_lb = self.labels
            self.win_id = self.ids
        else:
            self.win_idx = np.column_stack((np.arange((self.features.shape[-1]-self.sliding_win_size)//self.sliding_win_stride+1)*self.sliding_win_stride,
                                            (np.arange((self.features.shape[-1]-self.sliding_win_size)//self.sliding_win_stride+1)*self.sliding_win_stride+self.sliding_win_size)))
            self.win_lb = [np.unique(self.labels[i[0]:i[1]]) for i in self.win_idx]
            self.win_id = [np.unique(self.ids[i[0]:i[1]]) for i in self.win_idx]
            valid_idx = [i for i, item in enumerate(self.win_idx) if len(self.win_id[i]) == 1 and len(self.win_lb[i]) == 1]
            self.win_idx = self.win_idx[valid_idx]
            self.win_lb = [self.win_lb[i] for i in valid_idx]
            self.win_id = [self.win_id[i] for i in valid_idx]
            print('Sliding window generated')

    def __len__(self):
        return len(self.win_idx)

    def __getitem__(self, index):
        return self.features[..., self.win_idx[index][0]:self.win_idx[index][1]], np.squeeze(self.win_lb[index]), np.squeeze(self.win_id[index])

if __name__ == '__main__':
    train_dataset = Dataset(root_dir='/home/eric/Database/',
                            buf_dir='/home/eric/Database/.saved_buf_mover/',
                            db_subjects={'Database_VFN': ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'],
                                         'Database_SCE': ['18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45'],
                                         'Database_BLDN': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17'], #
                                         'Database_LP': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']},
                            movements=['rest','power','lateral','tripod','pointer','open'],
                            feature_set=['WL', 'LV', 'SSC', 'SKW', 'MNF', 'PKF'],
                            feature_win_size=300,
                            feature_win_stride=10,
                            sliding_win_size=512,
                            sliding_win_stride=10,
                            low_cut=10,
                            high_cut=500,
                            filter_order=4)

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)

    x_col = []
    y_col = []
    for inputs, labels, lbs in train_loader:
        x_col.append(inputs.detach_().numpy())
        y_col.append(labels.detach_().numpy())
        print(inputs.shape)
        print(labels.shape)
        print(labels)

    x_col = np.concatenate(x_col)
    y_col = np.concatenate(y_col)
    print(x_col.shape)
    print(y_col.shape)
