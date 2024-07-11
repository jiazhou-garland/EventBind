import os, random,json
from tqdm import tqdm

def write_file(mode, events, images):
    """
    save the list into a __.txt file
    """
    #events = sorted(events)
    #images = sorted(images)
    with open('./{}.txt'.format(mode), 'w') as f:
        for i in range(len(events)):
            f.write('{}\t{}\n'.format(events[i], images[i]))
        f.close()

def train(event_path_dataset):
    """
    divide the train ana validation data,

    val_ratio = 0.8 划分训练集与测试集比例8：2
    fold = 0 控制x折交叉验证的训练集与验证集划分，0表示每隔5张的第五张为验证集
    return: __.txt
    """
    print('开始划分训练集与验证集')
    events, images = [], []

    for cat_id, cat in enumerate(tqdm(os.listdir(event_path_dataset))):
        file_samples = os.listdir(os.path.join(event_path_dataset, cat))
        for fs in file_samples:
            event_str = str(os.path.join(event_path_dataset, cat, fs))
            num = fs.split('.')[0]
            image_name = 'mnist_train_' + num.lstrip('0') + '.png'
            image_str = event_str.replace('N-MNIST','MINIST').replace(fs, image_name)
            events.append(event_str)
            images.append(image_str)

    return events, images

def val(event_path_dataset):
    """
    divide the train ana validation data,

    val_ratio = 0.8 划分训练集与测试集比例8：2
    fold = 0 控制x折交叉验证的训练集与验证集划分，0表示每隔5张的第五张为验证集
    return: __.txt
    """
    print('开始划分训练集与验证集')
    events, images = [], []

    for cat_id, cat in enumerate(tqdm(os.listdir(event_path_dataset))):
        file_samples = os.listdir(os.path.join(event_path_dataset, cat))
        for fs in file_samples:
            event_str = str(os.path.join(event_path_dataset, cat, fs))
            num = fs.split('.')[0]
            image_name = 'mnist_train_' + num.lstrip('0') + '.png'
            image_str = event_str.replace('N-MNIST','MINIST').replace(fs, image_name)
            events.append(event_str)
            images.append(image_str)

    return events, images

def train_divide_shot(event_path_dataset, shot, val_interval=5):
    """
    divide the train ana validation data,

    val_ratio = 0.8 划分训练集与测试集比例8：2
    fold = 0 控制x折交叉验证的训练集与验证集划分，0表示每隔5张的第五张为验证集
    return: __.txt
    """
    print('开始划分训练集与验证集')
    train_e, train_im = [], []

    for cat_id, cat in enumerate(tqdm(os.listdir(event_path_dataset))):
        file_samples = os.listdir(os.path.join(event_path_dataset, cat))
        events_i, images_i = [], []
        for fs in file_samples:
            event_str = str(os.path.join(event_path_dataset, cat, fs))
            num = fs.split('.')[0]
            image_name = 'mnist_train_' + num.lstrip('0') + '.png'
            image_str = event_str.replace('N-MNIST','MINIST').replace(fs, image_name)
            events_i.append(event_str)
            images_i.append(image_str)
        i = 0
        while i != shot:
            random_num = random.randint(0, len(events_i)-1)
            if (random_num+1)!= 0:
                train_e.append(events_i[random_num])
                train_im.append(images_i[random_num])
                i += 1

    print("训练集与验证集划分结束")
    return train_e, train_im



if __name__ == "__main__":
    # Source data folder
    event_train_path_dataset = 'Path-to/N-MNIST/Train/'
    event_val_path_dataset = 'Path-to/N-MNIST/Test/'
    image_train_path_dataset = 'Path-to/MINIST/train'
    image_val_path_dataset = 'Path-to/MINIST/test/'
    events_train, images_train = train(event_train_path_dataset)
    events_val, images_val = val(event_val_path_dataset)
    write_file('N_MINIST_Train', events_train, images_train)
    write_file('N_MINIST_Val', events_val, images_val)

    # shot_list = [1,2,5,10,20,100]
    # for i in range(len(shot_list)):
    #     shot = shot_list[i]
    #     print(f'shot num f{shot}')
    #     train_e, train_im = train_divide_shot(event_train_path_dataset, shot, val_interval=5)
    #     write_file('MNIST_train_'+str(shot)+"_shot", train_e, train_im)





