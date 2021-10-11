import h5py
import numpy as np
from torch.utils.data import Dataset, SubsetRandomSampler


# Define the dataset class (more or less a wrapper)
class RadioML18Dataset(Dataset):
    def __init__(self, dataset_path: str, min_snr: int = -20, mod_classes: [int] = range(0, 24),
                 filter_testset: bool = False):
        super(RadioML18Dataset, self).__init__()
        h5_file = h5py.File(dataset_path, 'r')
        self.data = h5_file['X']
        self.mod = np.argmax(h5_file['Y'], axis=1)  # comes in one-hot encoding
        self.snr = h5_file['Z'][:, 0]
        self.len = self.data.shape[0]

        self.mod_classes = self.all_mod_classes()

        self.snr_classes = np.arange(-20., 32., 2)  # -20dB to 30dB

        # do not touch this seed to ensure the prescribed train/test split!
        np.random.seed(2018)
        train_indices = []
        test_indices = []
        for mod in range(0, 24):  # all modulations (0 to 23)
            for snr_idx in range(0, 26):  # all SNRs (0 to 25 = -20dB to +30dB)
                # 'X' holds frames strictly ordered by modulation and SNR
                start_idx = 26 * 4096 * mod + 4096 * snr_idx

                # Indices of the samples of the frame with index <start_idx>
                indices_subclass = list(range(start_idx, start_idx + 4096))

                # 90%/10% training/test split, applied evenly for each mod-SNR pair
                split = int(np.ceil(0.1 * 4096))

                # Shuffle the indexes
                # FIXME The shuffle is only applied to MOD/SNR Frames. In my opinion the shuffle should be global.
                np.random.shuffle(indices_subclass)

                # Add train indexes to train array (last 90% of samples)
                train_indices_subclass = indices_subclass[split:]

                # Add test indexes to test array (first 10% of samples)
                test_indices_subclass = indices_subclass[:split]

                # Important: you could train on a subset of the data, e.g. based on the SNR
                # here we use all available training samples
                if mod in mod_classes:
                    min_snr_idx = int(0.5 * min_snr + 10)
                    if snr_idx >= min_snr_idx:
                        train_indices.extend(train_indices_subclass)
                        if filter_testset:
                            test_indices.extend(test_indices_subclass)

                if not filter_testset:
                    # The test is applied to all SNR and Mod Classes
                    test_indices.extend(test_indices_subclass)

        # TODO What's this (SubsetRandomSampler) ?
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.test_sampler = SubsetRandomSampler(test_indices)

    def __getitem__(self, idx):
        # transpose frame into Pytorch channels-first format (NCL = -1,2,1024)
        return self.data[idx].transpose(), self.mod[idx], self.snr[idx]

    def __len__(self):
        return self.len

    @staticmethod
    def all_mod_classes() -> [str]:
        return ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK',
                '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
                'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
