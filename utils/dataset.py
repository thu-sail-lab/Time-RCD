import torch
import torch.utils.data
import numpy as np
epsilon = 1e-8

class TimeRCDDataset(torch.utils.data.Dataset):

    def __init__(self, data, window_size, stride=1, normalize=False, pad_to_multiple=True):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        # Ensure numpy array and a consistent 2D shape (N, C)
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.original_length = data.shape[0]
        self.pad_to_multiple = pad_to_multiple

        # Normalize data if other than UCR
        self.data = self._normalize_data(data) if normalize else data
        # self.data = data
        # self.univariate = self.data.shape[0] == 1

        # Handle padding if requested
        if self.pad_to_multiple:
            self.data, self.padding_mask = self._pad_data_to_multiple()
        else:
            self.padding_mask = np.ones(self.data.shape[0], dtype=bool)  # All data is real

    def _normalize_data(self, data, epsilon=1e-8):
        """ Normalize data using mean and standard deviation. """
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std = np.where(std == 0, epsilon, std)
        return ((data - mean) / std)

    def _pad_data_to_multiple(self):
        """ Pad data to make its length a multiple of window_size and return padding mask. """
        data_length = self.data.shape[0]
        remainder = data_length % self.window_size

        if remainder == 0:
            # No padding needed - all data is real
            padding_mask = np.ones(data_length, dtype=bool)
            return self.data, padding_mask

        # Calculate padding needed
        padding_length = self.window_size - remainder
        print(f"Padding AnomalyClipDataset: original length {data_length}, window_size {self.window_size}, adding {padding_length} samples")

        # Pad by repeating the last row, keeping 2D shape (1, C)
        last_row = self.data[-1:, :]
        padding_data = np.repeat(last_row, padding_length, axis=0)
        padded_data = np.vstack([self.data, padding_data])

        # Create padding mask: True for real data, False for padded data
        padding_mask = np.ones(data_length + padding_length, dtype=bool)
        padding_mask[data_length:] = False  # Mark padded samples as False

        return padded_data, padding_mask

    def __getitem__(self, index):
        start = index * self.stride
        end = start + self.window_size

        if end > self.data.shape[0]:
            raise IndexError("Index out of bounds for the dataset.")

        # Always return (window_size, num_features)
        sample = torch.tensor(self.data[start:end, :], dtype=torch.float32)
        mask = torch.tensor(self.padding_mask[start:end], dtype=torch.bool)

        # if self.univariate:
        #     sample = sample.unsqueeze(-1)  # Add channel dimension for univariate data

        return sample, mask

    def __len__(self):
        return max(0, (self.data.shape[0] - self.window_size) // self.stride + 1)


class ReconstructDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size, stride=1, normalize=True):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.data = self._normalize_data(data) if normalize else data
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.univariate = data.shape[1] == 1
        self.sample_num = max(0, (self.data.shape[0] - window_size) // stride + 1)
        self.samples, self.targets = self._generate_samples()

    def _normalize_data(self, data, epsilon=1e-8):
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std = np.where(std == 0, epsilon, std)  # Avoid division by zero
        return (data - mean) / std

    def _generate_samples(self):
        data = torch.tensor(self.data, dtype=torch.float32)

        if self.univariate:
            data = data.squeeze()
            X = torch.stack([data[i * self.stride : i * self.stride + self.window_size] for i in range(self.sample_num)])
            X = X.unsqueeze(-1)
        else:
            X = torch.stack([data[i * self.stride : i * self.stride + self.window_size, :] for i in range(self.sample_num)])

        return X, X

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        return self.samples[index], self.targets[index]

class ForecastDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size, pred_len, stride=1, normalize=True):
        super().__init__()
        self.window_size = window_size
        self.pred_len = pred_len
        self.stride = stride
        self.data = self._normalize_data(data) if normalize else data

        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.sample_num = max((self.data.shape[0] - window_size - pred_len) // stride + 1, 0)

        # Generate samples efficiently
        self.samples, self.targets = self._generate_samples()

    def _normalize_data(self, data, epsilon=1e-8):
        """ Normalize data using mean and standard deviation. """
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std = np.where(std == 0, epsilon, std)  # Avoid division by zero
        return (data - mean) / std

    def _generate_samples(self):
        """ Generate windowed samples efficiently using vectorized slicing. """
        data = torch.tensor(self.data, dtype=torch.float32)

        indices = np.arange(0, self.sample_num * self.stride, self.stride)

        X = torch.stack([data[i : i + self.window_size] for i in indices])
        Y = torch.stack([data[i + self.window_size : i + self.window_size + self.pred_len] for i in indices])

        return X, Y  # Inputs & targets

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        return self.samples[index], self.targets[index]

# class ForecastDataset(torch.utils.data.Dataset):
#     def __init__(self, data, window_size, pred_len, normalize=True):
#         super().__init__()
#         self.normalize = normalize

#         if self.normalize:
#             data_mean = np.mean(data, axis=0)
#             data_std = np.std(data, axis=0)
#             data_std = np.where(data_std == 0, epsilon, data_std)
#             self.data = (data - data_mean) / data_std
#         else:
#             self.data = data

#         self.window_size = window_size
        
#         if data.shape[1] == 1:
#             data = data.squeeze()
#             self.len, = data.shape
#             self.sample_num = max(self.len - self.window_size - pred_len + 1, 0)
#             X = torch.zeros((self.sample_num, self.window_size))
#             Y = torch.zeros((self.sample_num, pred_len))
            
#             for i in range(self.sample_num):
#                 X[i, :] = torch.from_numpy(data[i : i + self.window_size])
#                 Y[i, :] = torch.from_numpy(np.array(
#                     data[i + self.window_size: i + self.window_size + pred_len]
#                 ))
            
#             self.samples, self.targets = torch.unsqueeze(X, -1), torch.unsqueeze(Y, -1)
    
#         else:
#             self.len = self.data.shape[0]
#             self.sample_num = max(self.len - self.window_size - pred_len + 1, 0)

#             X = torch.zeros((self.sample_num, self.window_size, self.data.shape[1]))
#             Y = torch.zeros((self.sample_num, pred_len, self.data.shape[1]))

#             for i in range(self.sample_num):
#                 X[i, :] = torch.from_numpy(data[i : i + self.window_size, :])
#                 Y[i, :] = torch.from_numpy(data[i + self.window_size: i + self.window_size + pred_len, :])
            
#             self.samples, self.targets = X, Y

#     def __len__(self):
#         return self.sample_num

#     def __getitem__(self, index):
#         return self.samples[index, :, :], self.targets[index, :, :]

class TSDataset(torch.utils.data.Dataset):

    def __init__(self, X, y=None, mean=None, std=None):
        super(TSDataset, self).__init__()
        self.X = X
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.X[idx, :]

        if self.mean is not None and self.std is not None:
            sample = (sample - self.mean) / self.std
            # assert_almost_equal (0, sample.mean(), decimal=1)

        return torch.from_numpy(sample), idx


class ReconstructDataset_Moment(torch.utils.data.Dataset):
    def __init__(self, data, window_size, stride=1, normalize=True):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.data = self._normalize_data(data) if normalize else data

        self.univariate = self.data.shape[1] == 1
        self.sample_num = max((self.data.shape[0] - window_size) // stride + 1, 0)

        self.samples = self._generate_samples()
        self.input_mask = np.ones(self.window_size, dtype=np.float32)  # Fixed input mask

    def _normalize_data(self, data, epsilon=1e-8):
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std = np.where(std == 0, epsilon, std)  # Avoid division by zero
        return (data - mean) / std

    def _generate_samples(self):
        data = torch.tensor(self.data, dtype=torch.float32)
        indices = np.arange(0, self.sample_num * self.stride, self.stride)

        if self.univariate:
            X = torch.stack([data[i : i + self.window_size] for i in indices])
        else:
            X = torch.stack([data[i : i + self.window_size, :] for i in indices])

        return X

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        return self.samples[index], self.input_mask

class TACLipDataset(torch.utils.data.Dataset):
    def __init__(self, data, win_size, step=1, flag="test"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.test = data
        print("Before normalization", self.test[:20])
        self.test = self._normalize_data(self.test)
        print("After normalization", self.test[:20])
        self.test_labels = np.zeros(self.test.shape[0])
        
    def _normalize_data(self, data, epsilon=1e-8):
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std = np.where(std == 0, epsilon, std)  # Avoid division by zero
        return (data - mean) / std        

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
