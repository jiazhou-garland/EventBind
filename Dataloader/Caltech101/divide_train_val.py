import os, json,random
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

def train_val_divide(event_path_dataset, val_interval=5):
    """
    divide the train ana validation data,

    val_ratio = 0.8 划分训练集与测试集比例8：2
    fold = 0 控制x折交叉验证的训练集与验证集划分，0表示每隔5张的第五张为验证集
    return: __.txt
    """
    print('开始划分训练集与验证集')
    events, images = [], []
    train_e, val_e = [], []
    train_im, val_im = [], []
    class_dic = {}

    for cat_id, cat in enumerate(tqdm(os.listdir(event_path_dataset))):
        file_samples = os.listdir(os.path.join(event_path_dataset, cat))
        class_dic[cat] = cat_id
        for fs in file_samples:
            event_str = str(os.path.join(event_path_dataset, cat, fs))
            image_str = event_str.replace('N_Caltech101', 'Caltech-101').replace('.bin','.jpg')
            events.append(event_str)
            images.append(image_str)

    for i in range(len(events)):
        if ((i+1) % val_interval == 0): #每隔val_interval个划分至val集
            val_e.append(events[i])
            val_im.append(images[i])
        else:
            train_e.append(events[i])
            train_im.append(images[i])
    print("训练集与验证集划分结束")
    return train_e, val_e, train_im, val_im, class_dic

def train_val_divide_shot(event_path_dataset, shot, val_interval=5):
    """
    divide the train ana validation data,

    val_ratio = 0.8 划分训练集与测试集比例8：2
    fold = 0 控制x折交叉验证的训练集与验证集划分，0表示每隔5张的第五张为验证集
    return: __.txt
    """
    print('开始划分训练集与验证集')
    events, images = [], []
    train_e, val_e = [], []
    train_im, val_im = [], []
    class_dic = {}

    for cat_id, cat in enumerate(tqdm(os.listdir(event_path_dataset))):
        file_samples = os.listdir(os.path.join(event_path_dataset, cat))
        class_dic[cat] = cat_id
        events_i, images_i = [], []
        for fs in file_samples:
            event_str = str(os.path.join(event_path_dataset, cat, fs))
            image_str = event_str.replace('N_Caltech101', 'Caltech-101').replace('.bin','.jpg')
            events.append(event_str)
            images.append(image_str)
            events_i.append(event_str)
            images_i.append(image_str)
        i = 0
        while i != shot:
            random_num = random.randint(0, len(events_i)-1)
            if (random_num+1) % val_interval != 0:
                train_e.append(events_i[random_num])
                train_im.append(images_i[random_num])
                i += 1

    for i in range(len(events)):
        if ((i+1) % val_interval == 0): #每隔val_interval个划分至val集
            val_e.append(events[i])
            val_im.append(images[i])

    print("训练集与验证集划分结束")
    return train_e, val_e, train_im, val_im, class_dic


if __name__ == "__main__":
    # Source data folder
    event_path_dataset = 'Path-to/N_Caltech101'
    image_path_dataset = 'Path-to/Caltech-101'

    train_e, val_e, train_im, val_im, class_dic = train_val_divide(event_path_dataset, val_interval=6)
    write_file('Caltech101_train', train_e, train_im)
    write_file('Caltech101_val', val_e, val_im)
    tf = open("./Caltech101_classnames.json", "w")
    json.dump(class_dic, tf)
    tf.close()

    # shot_list = [1,2,5,10,20]
    # for i in range(len(shot_list)):
    #     shot = shot_list[i]
    #     print(f'shot num {shot}')
    #     train_e, val_e, train_im, val_im, class_dic = train_val_divide_shot(event_path_dataset, shot, val_interval=5)
    #     write_file('Caltech101_train_'+str(shot)+"_shot", train_e, train_im)





