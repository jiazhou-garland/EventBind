from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch, random, time, cv2, os, copy, json
import numpy as np
import matplotlib.pyplot as plt

class Nimagenet1k(Dataset):

    def __init__(self, txtPath, classPath, num_events=20000,
                 median_length=100000, frame=6, resize_width=224,
                 resize_height=224, representation=None,
                 augmentation=False, pad_frame_255=False,
                 EventCLIP=False):

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
        foldername_idx_tf = open(classPath, "r")
        self.classnames_dict = json.load(foldername_idx_tf)  # class name idx start from 0
        self.EventCLIP = EventCLIP
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        :param idx:
        :return: events_image 3,T,H,W
                 image 3,H,W
                 label_idx 0 to cls 1
        """
        # print(self.classnames_dict)
        event_stream_path, image_path = self.files[idx].split('\t')

        # label
        label_str = event_stream_path.split('/')[-2]
        label_idx = int(self.classnames_dict[label_str])
        # event
        events_stream = self.load_event(event_stream_path)
        # print(image_path)
        image = plt.imread(image_path[:-1])  # RGB
        image = self.scale_image(image) / 255.0
        image = image.transpose(2, 0, 1)  # H,W,3 -> 3,H,W

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
                if not self.EventCLIP:
                    events_image = self.generate_event_image_EventBind(events_tmp, (self.height, self.width),
                                                                       self.representation)
                else:
                    events_image = self.generate_color_event_image_EventCLIP(events_tmp, (self.height, self.width),
                                                                             self.representation)
                all_frame.append(events_image)
        if self.augmentation and random.random() > 0.5:
                # print("flip along x")
                all_frame = [cv2.flip(all_frame[i], 1) for i in range(len(all_frame))]
                image = cv2.flip(image, 1)
        all_frame = np.array(all_frame)
        events_data = all_frame.transpose(3, 0, 1, 2) # T,H,W,3 -> 3,T,H,W
        real_num_frame = 5

        b_im, _, _ = image.shape
        if b_im == 4:
            image = image[:3,:,:]

        b_ev, _, _,_ = events_data.shape
        if b_ev == 4:
            events_data = events_data[:3,:,:]

        return events_data, image, label_idx, real_num_frame

    def _readTXT(self, txtPath):
        with open(txtPath, 'r') as f:
            for line in f.readlines():
                self.files.append(line)
        random.shuffle(self.files)
        return len(self.files)

    def load_event(self, event_path):
        # Returns time-shifted numpy array event from event_path
        event = np.load(event_path)
        event = event['event_data']
        event = np.vstack([event['x'], event['y'], event['t'], event['p'].astype(np.uint8)]).T
        event = event.astype(np.float64)

        # Account for int-type timestamp
        event[:, 2] /= 1000000

        return event

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

    def generate_event_image_EventBind(self, events, shape, representation):
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

    def generate_color_event_image_EventCLIP(self, events, shape, representation, thresh=10.):
        """EventCLIP image2event process"""
        # count the number of positive and negative events per pixel
        H, W = shape
        x, y, t, p = events.T
        w_event = int(x.max() + 1)
        h_event = int(y.max() + 1)
        if representation == 'rgb':
            red = np.array([255, 0, 0], dtype=np.uint8)
            blue = np.array([0, 0, 255], dtype=np.uint8)
        else:
            red = np.array([127, 127, 127], dtype=np.uint8)
            blue = np.array([127, 127, 127], dtype=np.uint8)
        pos_x, pos_y = x[p > 0].astype(np.int64), y[p > 0].astype(np.int64)
        pos_count = np.bincount(pos_x + pos_y * w_event, minlength=h_event * w_event).reshape(h_event, w_event)
        neg_x, neg_y = x[p < 0].astype(np.int64), y[p < 0].astype(np.int64)
        neg_count = np.bincount(neg_x + neg_y * w_event, minlength=h_event * w_event).reshape(h_event, w_event)

        hist = np.stack([pos_count, neg_count], axis=-1)  # [H, W, 2]

        # remove hotpixels, i.e. pixels with event num > thresh * std + mean
        mean = hist[hist > 0].mean()
        std = hist[hist > 0].std()
        hist[hist > thresh * std + mean] = 0

        # normalize
        hist = hist.astype(np.float64) / hist.max()  # [H, W, 2]

        # colorize
        cmap = np.stack([red, blue], axis=0).astype(np.float64)  # [2, 3]
        img = hist @ cmap  # [H, W, 3]

        # alpha-masking with pure white background
        weights = np.clip(hist.sum(-1, keepdims=True), a_min=0, a_max=1)
        background = np.ones_like(img) * 255.
        img = img * weights + background * (1. - weights)

        # Change the image size to (H, W) by replacing the smaller part with 0 pixels
        img_resized = 255 * np.ones((H, W, 3), dtype=np.float32)

        current_height, current_width, _ = img.shape
        crop_top = np.random.randint(0, max(current_height - H + 1, 1))
        crop_left = np.random.randint(0, max(current_width - W + 1, 1))

        img_resized[:min(current_height, H), :min(current_width, W), :] = img[crop_top:crop_top + H, crop_left:crop_left + W, :]

        return img_resized / 255.0

    def scale_image(self, img):
        """
        0.For binary image, transform it into gray image by letting img=R=G=B
        # 1.Pad the image lower than H,W,3 with 255
        2.Resize the padded image to H,W,3

        """
        # for binary image
        if img.ndim == 2:
            img = np.array([img, img, img]).transpose(1, 2, 0)  # H,W,3
        # for image with 4 channels ??? RGBD
        if img.ndim == 4:
            img = img[:,:,:3]

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
