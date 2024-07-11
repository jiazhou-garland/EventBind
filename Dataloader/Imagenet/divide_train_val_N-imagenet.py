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

def train(event_path_dataset, mini=False):
    """
    divide the train ana validation data
    val_ratio = 0.8 划分训练集与测试集比例8：2
    fold = 0 控制x折交叉验证的训练集与验证集划分，0表示每隔5张的第五张为验证集
    return: __.txt
    """
    print('开始划分训练集')
    events, images = [], []

    for cat_id, cat in enumerate(tqdm(os.listdir(event_path_dataset))):
        file_samples = os.listdir(os.path.join(event_path_dataset, cat))
        for fs in file_samples:
            if not mini:
                event_samples = os.listdir(os.path.join(event_path_dataset, cat, fs))
                for event in event_samples:
                    event_str = str(os.path.join(event_path_dataset,cat,fs,event))
                    tmp = event_str.split('/')[6] +'/'+ event_str.split('/')[7]
                    image_str = event_str.replace(tmp,'Imagenet/train').replace('npz','JPEG')
                    images.append(image_str)
                    events.append(event_str)
            else:
                event_str = os.path.join(event_path_dataset, cat, fs)
                tmp = event_str.split('/')[6] + '/' + event_str.split('/')[7]
                image_str = event_str.replace(tmp, 'Imagenet/train').replace('npz', 'JPEG')
                images.append(image_str)
                events.append(event_str)

    print(len(events))
    return events, images

def val(event_path_dataset, Imagenet_dict_path):
    """
    divide the train ana validation data,

    val_ratio = 0.8 划分训练集与测试集比例8：2
    fold = 0 控制x折交叉验证的训练集与验证集划分，0表示每隔5张的第五张为验证集
    return: __.txt
    """
    print('开始划分验证集')
    events, images = [], []
    with open(Imagenet_dict_path, 'r') as f:
        Imagenet_dict = json.load(f)
    # print(len(Imagenet_dict.keys()))
    class_dic = {}
    folder_dic = {}
    name_list = []
    for cat_id, cat in enumerate(tqdm(os.listdir(event_path_dataset))):
        cat_name = Imagenet_dict[cat]
        class_dic[cat_name] = cat_id
        folder_dic[cat] = cat_id
        name_list.append(cat_name)
        file_samples = os.listdir(os.path.join(event_path_dataset, cat))
        for fs in file_samples:
            event_str = str(os.path.join(event_path_dataset,cat,fs))
            image_str = event_str.replace('N-imagenet-val','Imagenet/val').replace('npz','JPEG')
            events.append(event_str)
            images.append(image_str)
    print(cat_id)
    print(len(class_dic.keys()))
    print(len(name_list))
    print(len(events))
    return events, images, class_dic, folder_dic, name_list

def train_val_divide_shot(event_path_dataset, shot):
    """
    divide the train ana validation data,

    val_ratio = 0.8 划分训练集与测试集比例8：2
    fold = 0 控制x折交叉验证的训练集与验证集划分，0表示每隔5张的第五张为验证集
    return: __.txt
    """
    print('开始划分few-shot训练集')
    events, images = [], []
    train_e, train_im = [], []

    for cat_id, cat in enumerate(tqdm(os.listdir(event_path_dataset))):
        file_samples = os.listdir(os.path.join(event_path_dataset, cat))
        for fs in file_samples:
            event_samples = os.listdir(os.path.join(event_path_dataset,cat,fs))
            for event in event_samples:
                event_str = str(os.path.join(event_path_dataset,cat,fs,event))
                image_str = str(os.path.join(event_path_dataset,fs,event))
                image_str = image_str.replace('N-imagenet-train','Imagenet-1k-train').replace('npz','JPEG')
                events.append(event_str)
                images.append(image_str)
            i = 0
            while i != shot:
                random_num = random.randint(0, len(events)-1)
                train_e.append(events[random_num])
                train_im.append(images[random_num])
                i += 1

    return train_e, train_im



if __name__ == "__main__":
    # Source data folder
    event_train_path_dataset = 'Path-to/N-Imagenet/'
    event_val_path_dataset = 'Path-to/N-imagenet-val/'
    image_train_dataset = 'Path-to/Imagenet/train'
    image_val_dataset = 'Path-to/Imagenet/val'


    Imagenet_dict_path = 'Path-to/Imagenet-1k_classnames.json'
    events_train, images_train = train(event_train_path_dataset, mini=False)
    write_file('N_imagenet_Train', events_train, images_train)
    events_val, images_val, class_dic, folder_dic, name_list = val(event_val_path_dataset, Imagenet_dict_path)
    write_file('N_imagenet_Val', events_val, images_val)


    # shot_list = [1,2,5,10,20]
    # for i in range(len(shot_list)):
    #    shot = shot_list[i]
    #    print(f'shot num {shot}')
    #    train_e, train_im = train_val_divide_shot(event_train_path_dataset, shot)
    #    write_file('NImagenet_train_'+str(shot)+"_shot", train_e, train_im)
    #
    # tf = open("./NImagenet_classnames_idx.json", "w")
    # json.dump(class_dic, tf)
    # tf.close()
    #
    # tf = open("./NImagenet_foldername_idx.json", "w")
    # json.dump(folder_dic, tf)
    # tf.close()
    #
    # with open('./NImagenet_classnames_list.json', 'w') as f:
    #     json.dump(name_list, f)




