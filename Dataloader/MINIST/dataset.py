from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import random, time
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2, os, copy

class NMINIST(Dataset):
    def __init__(self, txtPath, classPath, num_events=20000, median_length=100000,
                 frame=6, resize_width=224, resize_height=224, representation=None,
                 augmentation=False, pad_frame_255=False):
        self.txtPath = txtPath
        self.files = []
        self.labels = []
        self.length = self._readTXT(self.txtPath)
        self.augmentation = augmentation
        self.width, self.height = resize_width, resize_height
        self.representation = representation
        self.frame = frame
        self.num_events = num_events
        self.median_length = median_length
        self.pad_frame_255 = pad_frame_255
        tf = open(classPath, "r")
        self.classnames_dict = json.load(tf)  # class name idx start from 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        :param idx:
        :return: events_image 3,T,H,W
                 image 3,H,W
                 label_idx 0 to cls 1
        """
        event_stream_path, image_path = self.files[idx].split('\t')

        # label
        label_str = event_stream_path.split('/')[-2]
        label_idx = int(label_str)

        events = self.load_ATIS_bin(event_stream_path)
        events_stream = np.array([events['x'], events['y'], events['t'], events['p']]).transpose()  # nx4,(x,y,t,p)
        image = plt.imread(image_path[:-1])  # RGB
        image = self.scale_image(image)
        image = image.transpose(2, 0, 1)  # H,W,3 -> 3,H,W

        if self.representation == 'frame':
            N, _ = events_stream.shape
            time_window = int(N / self.frame)
            all_frame = []
            for i in range(self.frame):
                events_tmp = events_stream[i * time_window: (i + 1) * time_window, :]
                events_image = self.generate_event_image(events_tmp, (self.height, self.width))
                all_frame.append(events_image)
            events_image = np.array(all_frame)
            events_data = events_image.transpose(3, 0, 1, 2)  # T,H,W,3 -> 3,T,H,W

        elif self.representation == 'gray_scale' or self.representation == 'rgb':
            real_n, _  = events_stream.shape
            real_num_frame = int(real_n / self.num_events)
            events_stream, pad_flag = self.pad_event_stream(events_stream, median_length=self.median_length)
            N, _ = events_stream.shape
            # print(N)
            num_frame = int(N / self.num_events)
            # print(num_frame)
            all_frame = []
            for i in range(num_frame):
                if pad_flag and i > real_num_frame and self.pad_frame_255:
                    all_frame.append(255*np.ones((self.height, self.width, 3), dtype=np.float64))
                else:
                    events_tmp = events_stream[i * self.num_events: (i + 1) * self.num_events, :]
                    events_image = self.generate_gray_scale_event_image(events_tmp, (self.height, self.width),
                                                                        self.representation)
                    all_frame.append(events_image)
            all_frame = np.array(all_frame)
            events_data = all_frame.transpose(3, 0, 1, 2)  # T,H,W,3 -> 3,T,H,W

        elif self.representation == 'mlp_learned':
            events_data,_ = self.pad_event_stream(events_stream)
            # print(events_data)
        real_num_frame = 5
        return events_data, image, label_idx, real_num_frame

    def load_ATIS_bin(self, file_name):
        '''
        :param file_name: path of the aedat v3 file
        :type file_name: str
        :return: a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
        :rtype: Dict
        This function is written by referring to https://github.com/jackd/events-tfds .
        Each ATIS binary example is a separate binary file consisting of a list of events. Each event occupies 40 bits as described below:
        bit 39 - 32: Xaddress (in pixels)
        bit 31 - 24: Yaddress (in pixels)
        bit 23: Polarity (0 for OFF, 1 for ON)
        bit 22 - 0: Timestamp (in microseconds)
        '''
        with open(file_name, 'rb') as bin_f:
            # `& 128` 是取一个8位二进制数的最高位
            # `& 127` 是取其除了最高位，也就是剩下的7位
            raw_data = np.uint32(np.fromfile(bin_f, dtype=np.uint8))
            x = raw_data[0::5]
            y = raw_data[1::5]
            rd_2__5 = raw_data[2::5]
            p = (rd_2__5 & 128) >> 7
            t = ((rd_2__5 & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
        return {'t': t, 'x': x, 'y': y, 'p': p}

    def _readTXT(self, txtPath):
        with open(txtPath, 'r') as f:
            for line in f.readlines():
                self.files.append(line)
        random.shuffle(self.files)
        return len(self.files)

    def pad_event_stream(self, event_stream, median_length = 104815):
        """
        pad event stream along n dim with 0
        so that event streams in one batch have the same dimension
        """
        # max_length = 428595
        pad_flag = False
        (N, _) = event_stream.shape
        if N < median_length:
            n = median_length - N
            pad = np.ones((n, 4))
            event_stream = np.concatenate((event_stream, pad), axis=0)
            pad_flag = True
        else:
            event_stream = event_stream[:median_length, :]
        return event_stream, pad_flag

    def generate_gray_scale_event_image(self, events, shape, representation):
        """
        Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}.
        x and y correspond to image coordinates u and v.
        """
        H, W = shape
        x, y, t, p = events.T
        x = x.astype(np.int32)
        y = y.astype(np.int32)

        w_event = x.max() + 1
        h_event = y.max() + 1
        img_pos = np.zeros((h_event * w_event,), dtype="float32")
        img_neg = np.zeros((h_event * w_event,), dtype="float32")
        np.add.at(img_pos, x[p == 1] + w_event * y[p == 1], 1)
        np.add.at(img_neg, x[p == 0] + w_event * y[p == 0], 1)
        if representation == 'rgb':
            gray_scale = 1 - (img_pos.reshape((h_event, w_event, 1))* [0, 255, 255] + img_neg.reshape((h_event, w_event, 1)) * [255,255,0]) / 255
        elif representation == 'gray_scale':
            gray_scale = 1 - (img_pos.reshape((h_event, w_event, 1)) + img_neg.reshape((h_event, w_event, 1))) * [127,127,127] / 255
        gray_scale = np.clip(gray_scale, 0, 255)

        # scale
        scale = H * 1.0 / h_event
        scale2 = W * 1.0 / w_event
        gray_scale = cv2.resize(gray_scale, dsize=None, fx=scale2, fy=scale)
        return gray_scale

    def scale_image(self, img):
        """
        0.For binary image, transform it into gray image by letting img=R=G=B
        # 1.Pad the image lower than H,W,3 with 255
        2.Resize the padded image to H,W,3
        """
        # for binary image
        if img.ndim == 2:
            img = np.array([img, img, img]).transpose(1, 2, 0)  # H,W,3

        # h, w, _ = img.shape
        # a = self.height - h
        # b = self.width - w
        #
        # if a > 0:
        #     img = np.pad(img, ((0, a), (0, 0), (0, 0)), "constant", constant_values=255)
        # if b > 0:
        #     img = np.pad(img, ((0, 0), (0, b), (0, 0)), "constant", constant_values=255)

        h2, w2 = img.shape[0:2]
        scale = self.height * 1.0 / h2
        scale2 = self.width * 1.0 / w2
        img = cv2.resize(img, dsize=None, fx=scale2, fy=scale)
        return img

def pad_event(event, max_event_length):
    C, N, H, W = event.shape
    pad_num = max_event_length - N
    if pad_num > 0:
        pad_zeros = np.zeros((C, pad_num, H, W))
        event = np.concatenate((event, pad_zeros), axis=1)

    return event

