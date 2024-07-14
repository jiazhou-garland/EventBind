import torch
from torch.utils.data import DataLoader
import os,gc,time,json,wandb
from os.path import join, abspath, dirname
import numpy as np
from model.EventCLIP import load_clip_to_cpu, EventCLIP
from model.LossFunction import symmetric_cross_entropy_loss
from model.utils.utils import read_yaml,seed_torch
import torch.nn as nn
import pandas as pd
from Dataloader.MINIST.dataset import NMINIST

def train_one_epoch(model, cfg, scaler, optimizer, scheduler, dataloader, epoch):
    epoch_start = time.time()
    length = len(dataloader)
    running_loss, dataset_size, loss, epoch_loss = 0.0, 0.0, 0.0, 0.0
    batch_size = cfg['Dataset']['Train']['Batch_size']
    for step, (events, images, labels, real_num_frame) in enumerate(dataloader):
        batch_start = time.time()
        model = model.train().float()
        if cfg['MODEL']['BACKBONE']['PRE_ENCODING'] == "fp16":
            events = events.half()
            images = images.half()
        with torch.cuda.amp.autocast(enabled=True):
            image_features, event_features, text_features_e, text_features_im \
                    = model(events, images, labels, real_num_frame)
            train_loss, mse_loss, loss_tim_im, logit_scale = 0.0, 0.0, 0.0, 100.0
            mse_loss = torch.tensor(mse_loss).to(image_features.device)
            loss_tim_im = torch.tensor(loss_tim_im).to(image_features.device)
            if cfg['LossFunction']['use_te_e']:
                logits_te_e = logit_scale * event_features @ text_features_e.t()
                loss_te_e = symmetric_cross_entropy_loss(logits_te_e)
                train_loss = train_loss + loss_te_e
            if cfg['LossFunction']['use_te_tim']:
                logits_te_tim = logit_scale * text_features_im @ text_features_e.t()
                loss_te_tim = symmetric_cross_entropy_loss(logits_te_tim)
                train_loss = train_loss + loss_te_tim
            if cfg['LossFunction']['use_im_ev_hi']:
                logit_im_ev_hi = logit_scale * event_features @ image_features.t()
                loss_im_ev_hi = symmetric_cross_entropy_loss(logit_im_ev_hi)
                train_loss = train_loss + loss_im_ev_hi

            if cfg['MODEL']['TextEncoder']['init_ctx'] and cfg['MODEL']['TextEncoder']['leranable_ctx']:
                mse_loss = torch.nn.MSELoss()(text_features_e, text_features_im)
                train_loss = train_loss + 0.1 * mse_loss

            loss_list = torch.stack([train_loss, loss_tim_im, loss_te_e, loss_te_tim, loss_im_ev_hi, mse_loss], dim=0) \
                        / cfg['Trainer']['accumulation_steps']

        scaler.scale(loss_list[0]).backward()
        if (step + 1) % cfg['Trainer']['accumulation_steps'] == 0:
            scaler.step(optimizer)
            scaler.update()
            # zero the parameter gradients
            optimizer.zero_grad()

        running_loss += (loss_list.cpu().detach().numpy() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size
        batch_end = time.time()

        if (step) % cfg['Trainer']['print_freq'] == 0:
            if cfg['MODEL']['TextEncoder']['init_ctx'] and cfg['MODEL']['TextEncoder']['leranable_ctx']:
                print(
                    f'[{step + 1} / {length} | epoch: {epoch}] epoch_total_loss: {epoch_loss[0]:.7f} | '
                    f'epoch_tim_im_loss: {epoch_loss[1]:.7f} | '
                    f'epoch_te_ev_loss: {epoch_loss[2]:.7f} | '
                    f'epoch_te_tim_loss: {epoch_loss[3]:.7f} | '
                    f'epoch_im_ev_hi_loss: {epoch_loss[4]:.7f} | '
                    f'mse_loss: {epoch_loss[5]:.7f} | '
                    f'lr: {optimizer.param_groups[0]["lr"]:.7f} | '
                    f'batch_time: {(batch_end - batch_start):.3f} | '
                )
            else:
                print(
                    f'[{step + 1} / {length} | epoch: {epoch}] epoch_total_loss: {epoch_loss[0]:.7f} | '
                    f'epoch_tim_im_loss: {epoch_loss[1]:.7f} | '
                    f'epoch_te_ev_loss: {epoch_loss[2]:.7f} | '
                    f'epoch_te_tim_loss: {epoch_loss[3]:.7f} | '
                    f'epoch_im_ev_hi_loss: {epoch_loss[4]:.7f} | '
                    f'lr: {optimizer.param_groups[0]["lr"]:.7f} | '
                    f'batch_time: {(batch_end - batch_start):.3f} | '
                )

    scheduler.step()
    epoch_time = time.time() - epoch_start
    print(f"EPOCH {epoch} training takes {epoch_time}s.")
    # torch.cuda.empty_cache()
    gc.collect()
    return epoch_loss

def evaluate_one_epoch(model, cfg, dataloader, classnames_num, logit_scale):
    classnames_idxs = np.arange(0, classnames_num)
    epoch_start = time.time()
    total, hit1_im, hit5_im, hit1_ev, hit5_ev = 0, 0, 0, 0, 0
    model = model.eval().float()
    all_logits_tim_im = []
    all_logits_te_e = []
    all_image_features = []
    all_event_features = []
    all_label = []
#------------------------------------------------------ev&im -> text----------------------------------------------
    for events, images, labels, real_num_frame in dataloader:
        if cfg['MODEL']['BACKBONE']['PRE_ENCODING'] == "fp16":
            events = events.float()
            images = images.float()
        with torch.no_grad():
            image_features, event_features, text_features_e, text_features_im \
                = model(events, images, classnames_idxs, real_num_frame)
            logits_te_e = 100 * event_features @ text_features_e[:classnames_num,:].t()
            logits_tim_im = 100 * image_features @ text_features_im[:classnames_num,:].t()
            scores_im = logits_tim_im.softmax(dim=-1) # b,n
            scores_ev = logits_te_e.softmax(dim=-1) # b,n
            all_logits_tim_im.append(logits_tim_im)
            all_logits_te_e.append(logits_te_e)
            all_image_features.append(image_features) #N,dim
            all_event_features.append(event_features) #N,dim

        B, _ = scores_im.size()
        for i in range(B):
            total += 1
            scores_im_i = scores_im[i]
            scores_ev_i = scores_ev[i]
            label_i = labels[i]
            all_label.append(label_i.cpu().detach().numpy())

            if scores_im_i.topk(1)[1].cpu().detach().numpy()[0] == label_i.cpu().detach().numpy():
                hit1_im += 1
            if int(label_i.cpu().detach().numpy()) in set(scores_im_i.topk(5)[1].cpu().detach().numpy()):
                hit5_im += 1

            if scores_ev_i.topk(1)[1].cpu().detach().numpy()[0] == label_i.cpu().detach().numpy():
                hit1_ev += 1
            if int(label_i.cpu().detach().numpy()) in set(scores_ev_i.topk(5)[1].cpu().detach().numpy()):
                hit5_ev += 1

            acc1_im = hit1_im / total * 100.
            acc5_im = hit5_im / total * 100.

            acc1_ev = hit1_ev / total * 100.
            acc5_ev = hit5_ev / total * 100.

            if total % cfg['Trainer']['print_freq'] == 0:
                print(f'[Evaluation] num_samples: {total}  '
                      f'cumulative_acc1_im: {acc1_im:.2f}%  '
                      f'cumulative_acc5_im: {acc5_im:.2f}%  '
                      f'cumulative_acc1_ev: {acc1_ev:.2f}%  '
                      f'cumulative_acc5_ev: {acc5_ev:.2f}%  ')


    print(f'Accuracy on validation set: im_ev_top1={acc1_im:.2f}%, im_ev_top5={acc5_im:.2f}%,'
          f' ev_top1={acc1_ev:.2f}%, ev_top5={acc5_ev:.2f}%')

    all_image_features = torch.cat(all_image_features, dim=0) # Nxdim
    all_event_features = torch.cat(all_event_features, dim=0) # Nxdim
    logits_im_ev = logit_scale * all_image_features @ all_event_features.t() # NxN
    scores_im_ev = logits_im_ev.softmax(dim=-1) # NxN

    all_logits_tim_im = torch.cat(all_logits_tim_im, dim=0) # Nxn
    all_logits_te_e = torch.cat(all_logits_te_e, dim=0) # Nxn
    scores_tim_im = all_logits_tim_im.t().softmax(dim=-1)  # nxN
    scores_te_e = all_logits_te_e.t().softmax(dim=-1)  # nxN
    N,n = scores_tim_im.t().size()

    all_label = np.array(all_label)

    num_retrival_1_im, num_retrival_5_im,  num_retrival_10_im = 0, 0, 0
    num_retrival_1_e, num_retrival_5_e,  num_retrival_10_e = 0, 0, 0
    num_retrival_1_im_e, num_retrival_5_im_e,  num_retrival_10_im_e = 0, 0, 0

    acc_retrival_1_e, acc_retrival_5_e, acc_retrival_10_e = 0,0,0
    acc_retrival_1_im, acc_retrival_5_im, acc_retrival_10_im = 0,0,0
    acc_retrival_1_im_e, acc_retrival_5_im_e, acc_retrival_10_im_e = 0,0,0

    total_re = 0

#-------------------------------------------------------text -> im&ev----------------------------------------------
    for i in range(n):
        topk_5_e = set()
        topk_10_e = set()
        topk_5_im = set()
        topk_10_im = set()
        total_re += 1
        label_i = classnames_idxs[i]
        score_tim_im_i = scores_tim_im[i]
        scores_te_e_i = scores_te_e[i]

        for num in set(score_tim_im_i.topk(5)[1].cpu().detach().numpy()):
            topk_5_im.add(all_label[num])
        for num in set(score_tim_im_i.topk(10)[1].cpu().detach().numpy()):
            topk_10_im.add(all_label[num])

        if all_label[score_tim_im_i.topk(1)[1].cpu().detach().numpy()[0]] == label_i:
            num_retrival_1_im += 1
        if label_i in topk_5_im:
            num_retrival_5_im += 1
        if label_i in topk_10_im:
            num_retrival_10_im += 1

        retrival_1_im = num_retrival_1_im / total_re * 100.
        retrival_5_im = num_retrival_5_im / total_re * 100.
        retrival_10_im = num_retrival_10_im / total_re * 100.

        if total_re % cfg['Trainer']['print_freq'] == 0:
            print(f'[Evaluation] num_samples: {total_re}  '
                  f'acc_retrival_1_im: {retrival_1_im:.2f}%  '
                  f'acc_retrival_5_im: {retrival_5_im:.2f}%  '
                  f'acc_retrival_10_im: {retrival_10_im:.2f}%  ')

        for num in set(scores_te_e_i.topk(5)[1].cpu().detach().numpy()):
            topk_5_e.add(all_label[num])
        for num in set(scores_te_e_i.topk(10)[1].cpu().detach().numpy()):
            topk_10_e.add(all_label[num])

        if all_label[scores_te_e_i.topk(1)[1].cpu().detach().numpy()[0]] == label_i:
            num_retrival_1_e += 1
        if label_i in topk_5_e:
            num_retrival_5_e += 1
        if label_i in topk_10_e:
            num_retrival_10_e += 1

        retrival_1_e = num_retrival_1_e / total_re * 100.
        retrival_5_e = num_retrival_5_e / total_re * 100.
        retrival_10_e = num_retrival_10_e / total_re * 100.

        if total_re % cfg['Trainer']['print_freq'] == 0:
            print(f'[Evaluation] num_samples: {total_re}  '
                  f'acc_retrival_1_e: {retrival_1_e:.2f}%  '
                  f'acc_retrival_5_e: {retrival_5_e:.2f}%  '
                  f'acc_retrival_10_e: {retrival_10_e:.2f}%  ')
    if total_re != 0:
        acc_retrival_1_e = num_retrival_1_e / total_re * 100.
        acc_retrival_5_e = num_retrival_5_e / total_re * 100.
        acc_retrival_10_e = num_retrival_10_e / total_re * 100.

        acc_retrival_1_im = num_retrival_1_im / total_re * 100.
        acc_retrival_5_im = num_retrival_5_im / total_re * 100.
        acc_retrival_10_im = num_retrival_10_im / total_re * 100.

# -----------------------------------------------------------im->ev------------------------------------------------
    total_re = 0
    for i in range(N):
        scores_im_ev_i = scores_im_ev[i]
        topk_5_im_e = set()
        topk_10_im_e = set()
        total_re += 1
        label_i = all_label[i]

        for num in set(scores_im_ev_i.topk(5)[1].cpu().detach().numpy()):
            topk_5_im_e.add(all_label[num])
        for num in set(scores_im_ev_i.topk(10)[1].cpu().detach().numpy()):
            topk_10_im_e.add(all_label[num])

        if all_label[scores_im_ev_i.topk(1)[1].cpu().detach().numpy()[0]] == label_i:
            num_retrival_1_im_e += 1
        if label_i in topk_5_im_e:
            num_retrival_5_im_e += 1
        if label_i in topk_10_im_e:
            num_retrival_10_im_e += 1

        acc_retrival_1_im_e = num_retrival_1_im_e / total_re * 100.
        acc_retrival_5_im_e = num_retrival_5_im_e / total_re * 100.
        acc_retrival_10_im_e = num_retrival_10_im_e / total_re * 100.

        if total_re % cfg['Trainer']['print_freq'] == 0:
            print(f'[Evaluation] num_samples: {total_re}  '
                  f'acc_retrival_1_im_e: {acc_retrival_1_im_e:.2f}%  '
                  f'acc_retrival_5_im_e: {acc_retrival_5_im_e:.2f}%  '
                  f'acc_retrival_10_im_e: {acc_retrival_10_im_e:.2f}%  ')
    if total_re != 0:
        acc_retrival_1_im_e = num_retrival_1_im_e / total_re * 100.
        acc_retrival_5_im_e = num_retrival_5_im_e / total_re * 100.
        acc_retrival_10_im_e = num_retrival_10_im_e / total_re * 100.
#
#     epoch_time = time.time() - epoch_start
#     torch.cuda.empty_cache()
#     gc.collect()
    return acc1_im, acc5_im, acc1_ev, acc5_ev, \
        acc_retrival_1_e, acc_retrival_5_e, acc_retrival_10_e,\
        acc_retrival_1_im, acc_retrival_5_im, acc_retrival_10_im,\
        acc_retrival_1_im_e, acc_retrival_5_im_e, acc_retrival_10_im_e

if __name__ == '__main__':
    # ---------------------------------------------------init----------------------------------------------------------------
    cfg = read_yaml('./Configs/N-MINIST.yaml')
    THIS_DIR = abspath(dirname(__file__))
    RESULT_DIR = join(THIS_DIR, "Result")
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    EXP_DIR = join(RESULT_DIR, f"{cfg['Wandb']['exp_group_name']}-" + str(cfg['Wandb']['exp_num']))
    if not os.path.exists(EXP_DIR):
        os.makedirs(EXP_DIR)

    run = wandb.init(project=cfg['Wandb']['project'],
                     config=cfg,
                     entity='garland-chou',
                     name=str(cfg['Wandb']['exp_num']),
                     group=cfg['Wandb']['exp_group_name'],
                     )
    seed_torch(cfg['Trainer']['seed'])
    #
    tf = open(cfg['Dataset']['Classnames'], "r")
    classnames_dict = json.load(tf)  # class name idx start from 0
    classnames_list = [i for i in classnames_dict.keys()]
    classnames_num = len(classnames_list)
    # print(classnames_num)

    # -----------------------------------------------dataset-----------------------------------------------------------------
    train_dataset = NMINIST(cfg['Dataset']['Train']['Path'], cfg['Dataset']['Classnames'],
                                pad_frame_255=cfg['Dataset']['pad_frame_255'],
                                num_events=cfg['Dataset']['num_events'],
                                median_length=cfg['Dataset']['median_length'],
                                resize_width=cfg['Dataset']['resize_width'],
                                resize_height=cfg['Dataset']['resize_height'],
                                representation=cfg['Dataset']['Representation'],
                                augmentation=cfg['Dataset']['Train']['Augmentation'])
    train_loader = DataLoader(train_dataset, batch_size=cfg['Dataset']['Train']['Batch_size'],
                              shuffle=True, drop_last=True, num_workers=4,
                              prefetch_factor=2, pin_memory=True,)

    val_dataset = NMINIST(cfg['Dataset']['Val']['Path'], cfg['Dataset']['Classnames'],
                              pad_frame_255=cfg['Dataset']['pad_frame_255'],
                              num_events=cfg['Dataset']['num_events'],
                              median_length=cfg['Dataset']['median_length'],
                              resize_width=cfg['Dataset']['resize_width'],
                              resize_height=cfg['Dataset']['resize_height'],
                              representation=cfg['Dataset']['Representation'],
                              augmentation=cfg['Dataset']['Val']['Augmentation'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['Dataset']['Val']['Batch_size'],
                            shuffle=False, drop_last=False, num_workers=4,
                            prefetch_factor=2, pin_memory=True)

    # -------------------------------------------------model-----------------------------------------------------------------
    gpus = cfg['Trainer']['GPU_ids']
    device = torch.device("cuda:{}".format(gpus[0]) if torch.cuda.is_available() else "cpu")

    print(f"Loading CLIP (backbone: {cfg['MODEL']['BACKBONE']['Name']})")
    clip_model_im = load_clip_to_cpu(cfg)
    for name, param in clip_model_im.named_parameters():
        param.requires_grad = False
    clip_model_ev = load_clip_to_cpu(cfg)
    for name, param in clip_model_ev.named_parameters():
        if cfg['MODEL']['EventEncoder']['train_clip_backbone']:
            param.requires_grad = True
        else:
            param.requires_grad = False

    print('----------------------------------------------------')
    print('Trainable Parameters')
    EventCLIP = EventCLIP(cfg, clip_model_im, clip_model_ev).to(device)
    for name, param in EventCLIP.named_parameters():
        if "prompt_learner.clip_model" in name:
            param.requires_grad = False
        if param.requires_grad == True:
            print('Trainable layers includes: ' + name)

    # Calculate total parameter size in megabytes
    total_param_size_bytes = sum(param.numel() * param.element_size() for param in EventCLIP.parameters())
    total_param_size_mb = total_param_size_bytes / (1024 * 1024)
    print(f"Total Model Parameter Size: {total_param_size_mb:.5f} MB")
            
    EventCLIP = nn.DataParallel(EventCLIP.to(device), device_ids=gpus, output_device=gpus[0])
    if cfg['MODEL']['Load_Path'] != 'None':
        EventCLIP.load_state_dict(torch.load(cfg['MODEL']['Load_Path']))
    # ----------------------------------------------optimizer&lr-------------------------------------------------------------
    optimizer = torch.optim.AdamW(EventCLIP.parameters(), lr=float(cfg['Trainer']['lr']),
                                  weight_decay=float(cfg['Trainer']['weight_decay']))
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['Trainer']['epoch'],eta_min=float(cfg['Trainer']['min_lr']))
    loss_scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=True)

    # ----------------------------------------------train&val----------------------------------------------------------------
    num_epochs = cfg['Trainer']['epoch']
    best_acc1_im, best_acc5_im, best_acc1_ev, best_acc5_ev = -np.inf, -np.inf, -np.inf, -np.inf
    logit_scale = clip_model_im.logit_scale.exp()

    for epoch in range(1, num_epochs + 1):
        epoch_loss = train_one_epoch(EventCLIP, cfg, loss_scaler, optimizer, lr_sched, train_loader, epoch)

        if cfg['MODEL']['TextEncoder']['init_ctx'] and cfg['MODEL']['TextEncoder']['leranable_ctx']:
            wandb.log({"train_epoch_total_loss": epoch_loss[0],
                       "train_epoch_tim_im_loss": epoch_loss[1],
                       "train_epoch_te_ev_loss": epoch_loss[2],
                       "train_epoch_te_tim_loss": epoch_loss[3],
                       "train_epoch_im_ev_hi_loss": epoch_loss[4],
                       "train_epoch_mse_loss": epoch_loss[5]})
        else:
            wandb.log({"train_epoch_total_loss": epoch_loss[0],
                       "train_epoch_tim_im_loss": epoch_loss[1],
                       "train_epoch_te_ev_loss": epoch_loss[2],
                       "train_epoch_te_tim_loss": epoch_loss[3],
                       "train_epoch_im_ev_hi_loss": epoch_loss[4]})

        acc1_im, acc5_im, acc1_ev, acc5_ev, \
        acc_retrival_1_e, acc_retrival_5_e, acc_retrival_10_e, \
        acc_retrival_1_im, acc_retrival_5_im, acc_retrival_10_im, \
        acc_retrival_1_im_e, acc_retrival_5_im_e, acc_retrival_10_im_e \
            = evaluate_one_epoch(EventCLIP, cfg, val_loader, classnames_num, logit_scale)

        # Log the metrics
        wandb.log({"acc1_im": acc1_im, "acc5_im": acc5_im,
                   "acc1_ev": acc1_ev, "acc5_ev": acc5_ev,
                   'acc_retrival_1_e': acc_retrival_1_e,
                   'acc_retrival_5_e': acc_retrival_5_e,
                   'acc_retrival_10_e': acc_retrival_10_e,
                   'acc_retrival_1_im': acc_retrival_1_im,
                   'acc_retrival_5_im': acc_retrival_5_im,
                   'acc_retrival_10_im': acc_retrival_10_im,
                   'acc_retrival_1_im_e': acc_retrival_1_im_e,
                   'acc_retrival_5_im_e': acc_retrival_5_im_e,
                   'acc_retrival_10_im_e': acc_retrival_10_im_e,
                   "LR": lr_sched.get_last_lr()[0]})

        #save model based on event
        if acc1_ev >= best_acc1_ev:
            print(f"acc Improved ({best_acc1_ev:0.4f}% ---> {acc1_ev:0.4f}%), ({best_acc5_ev:0.4f}% ---> {acc5_ev:0.4f})%")
            best_acc1_ev, best_acc5_ev = acc1_ev, acc5_ev

            PATH = join(EXP_DIR, f"best_im_ev_epoch"+str(acc1_ev)+".bin")
            torch.save(EventCLIP.state_dict(), PATH)
            print(f"Model Saved at {PATH}")
