import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

import os
import json
import logging
import argparse
import torch
from model.model import *
from model.loss import *
from model.metric import *
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data._utils.collate import default_collate
from data_loader.dataset import *
from trainer.lstm_trainer import LSTMTrainer
from utils.data_augmentation import Compose, RandomRotationFlip, RandomCrop, CenterCrop
from os.path import join
import shutil
from torch.utils.data import Subset
from functools import partial

logging.basicConfig(level=logging.INFO, format='')
def pad_seq(list_tensor, pad_val=0.0):
    """
    list[Ti, ...]  →  (N, T_max, ...)  +  mask(N, T_max)
    뒤쪽 차원(...) 수가 0(1‑D), 3(4‑D) 모두 OK
    """
    T_max = max(t.shape[0] for t in list_tensor)
    trailing_shape = list_tensor[0].shape[1:]      # 가변 길이 제외 나머지 차원
    N = len(list_tensor)

    out_shape = (N, T_max, *trailing_shape)        # e.g. (N, T, C, H, W) 또는 (N, T)
    out  = list_tensor[0].new_full(out_shape, pad_val)
    mask = torch.zeros(N, T_max, dtype=torch.bool)

    for n, t in enumerate(list_tensor):
        L = t.shape[0]
        out[n, :L]  = t
        mask[n, :L] = 1

    return out, mask


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(v, device) for v in x)
    return x

def collate_keep_sequence(batch, device):
    # batch: [N] where each item is a sequence [L] of dicts
    # 절대 default_collate 호출하지 말 것!
    return _to_device(batch, device)

def stack_to_cuda(batch, device="cuda:0"):
    first = batch[0]

    # Case ①: SequenceUpsampledFramesDataset → list-of-dict
    if isinstance(first, list) and isinstance(first[0], dict):
        L     = len(first)        # 시퀀스 길이
        keys  = first[0].keys()
        N     = len(batch)

        collated = {}
        for k in keys:
            seq_items = [sample[l][k] for sample in batch for l in range(L)]

            if k in ("frames", "stamps"):  # 가변 길이
                stacked, mask = pad_seq(seq_items)
                collated[k]        = stacked.view(N, L, *stacked.shape[1:]).to(device, non_blocking=True)
                collated[f"{k}_mask"] = mask.view(N, L, -1).to(device, non_blocking=True)
            else:
                stacked = torch.stack(seq_items)
                collated[k] = stacked.view(N, L, *stacked.shape[1:]).to(device, non_blocking=True)

        return collated

    # Case ②: 단일 dict
    if isinstance(first, dict):
        return {k: torch.stack([b[k] for b in batch]).to(device, non_blocking=True)
                for k in first.keys()}

    # fallback
    return default_collate(batch)



def concatenate_subfolders(base_folder, dataset_type, event_folder, depth_folder, frame_folder, sequence_length, transform=None,
                           proba_pause_when_running=0.0, proba_pause_when_paused=0.0, step_size=1, clip_distance=100.0,
                           normalize=True, scale_factor=1.0, inverse=False):
    """
    Create an instance of ConcatDataset by aggregating all the datasets in a given folder
    """

    #subfolders = os.listdir(base_folder)
    subfolders = [d for d in os.listdir(base_folder) if not d.startswith('.')]

    print('Found {} samples in {}'.format(len(subfolders), base_folder))

    train_datasets = []
    # breakpoint()
    for dataset_name in subfolders:
        train_datasets.append(eval(dataset_type)(base_folder=join(base_folder, dataset_name),
                                                 event_folder=event_folder,
                                                 depth_folder=depth_folder,
                                                 frame_folder=frame_folder,
                                                 sequence_length=sequence_length,
                                                 transform=transform,
                                                 proba_pause_when_running=proba_pause_when_running,
                                                 proba_pause_when_paused=proba_pause_when_paused,
                                                 step_size=step_size,
                                                 clip_distance=clip_distance,
                                                 normalize=normalize,
                                                 scale_factor=scale_factor,
                                                 inverse = inverse))
        
    
    concat_dataset = ConcatDataset(train_datasets)
    # breakpoint()



    return concat_dataset



def main(config, resume, initial_checkpoint=None):
    train_logger = None

    L = config['trainer']['sequence_length']
    assert(L > 0)

    dataset_type, base_folder, event_folder, depth_folder, frame_folder = {}, {}, {}, {}, {}
    proba_pause_when_running, proba_pause_when_paused = {}, {}
    step_size = {}
    clip_distance = {}
    scale_factor = {}

    # this will raise an exception is the env variable is not set
    preprocessed_datasets_folder = os.environ['PREPROCESSED_DATASETS_FOLDER']

    for split in ['train', 'validation']:
        dataset_type[split] = config['data_loader'][split]['type']
        base_folder[split] = join(preprocessed_datasets_folder, config['data_loader'][split]['base_folder'])
        event_folder[split] = config['data_loader'][split]['event_folder']
        depth_folder[split] = config['data_loader'][split]['depth_folder']
        frame_folder[split] = config['data_loader'][split]['frame_folder']
        proba_pause_when_running[split] = config['data_loader'][split]['proba_pause_when_running']
        proba_pause_when_paused[split] = config['data_loader'][split]['proba_pause_when_paused']
        scale_factor[split] = config['data_loader'][split]['scale_factor']
        # breakpoint()

        try:
            step_size[split] = config['data_loader'][split]['step_size']
        except KeyError:
            step_size[split] = 1

        try:
            clip_distance[split] = config['data_loader'][split]['clip_distance']
        except KeyError:
            clip_distance[split] = 100.0

    normalize = config['data_loader'].get('normalize', True)
    use_psf = config['use_psf']

    try:
        inverse = config['data_loader']['inverse']
    except KeyError:
        inverse = False

    train_dataset = concatenate_subfolders(base_folder['train'],
                                           dataset_type['train'],
                                           event_folder['train'],
                                           depth_folder['train'],
                                           frame_folder['train'],
                                           sequence_length=L,
                                        #    transform=Compose([RandomRotationFlip(0.0, 0.5, 0.0),
                                        #                       RandomCrop(112)]),
                                           proba_pause_when_running=proba_pause_when_running['train'],
                                           proba_pause_when_paused=proba_pause_when_paused['train'],
                                           step_size=step_size['train'],
                                           clip_distance=clip_distance['train'],
                                           normalize=normalize,
                                           scale_factor=scale_factor['train'],
                                           inverse = inverse)
    

    ###### DEBUG : REDUCE DATASET ########
    # total = len(train_dataset)          # 예: 3 964
    # k = 10                  # 50 %
    # torch.manual_seed(0)               # 재현용(옵션)
    # indices = torch.randperm(total)[:k] # 섞어서 앞 k개 선택
    # train_dataset = Subset(train_dataset, indices)

    validation_dataset = concatenate_subfolders(base_folder['validation'],
                                                dataset_type['validation'],
                                                event_folder['validation'],
                                                depth_folder['validation'],
                                                frame_folder['validation'],
                                                sequence_length=L,
                                                transform=CenterCrop(112),
                                                proba_pause_when_running=proba_pause_when_running['validation'],
                                                proba_pause_when_paused=proba_pause_when_paused['validation'],
                                                step_size=step_size['validation'],
                                                clip_distance=clip_distance['validation'],
                                                normalize=normalize,
                                                scale_factor=scale_factor['validation'],
                                                inverse = inverse)
    
    # validation_dataset = train_dataset

    # Set up data loaders
    kwargs = {'num_workers': config['data_loader']['num_workers'],
              'pin_memory': config['data_loader']['pin_memory']} if config['cuda'] else {}
    
    if use_psf:
        device = torch.device(f"cuda:{config['gpu']}")
        collate = partial(collate_keep_sequence, device=device)

        data_loader = DataLoader(train_dataset,
                                batch_size=config['data_loader']['batch_size'],
                                shuffle=config['data_loader']['shuffle'],
                                collate_fn=collate,
                                num_workers=config['data_loader']['num_workers'],
                                pin_memory=config['data_loader']['pin_memory'])

        valid_data_loader = DataLoader(validation_dataset,
                                    batch_size=config['data_loader']['batch_size'],
                                    shuffle=False,
                                    collate_fn=collate,
                                    num_workers=config['data_loader']['num_workers'],
                                    pin_memory=config['data_loader']['pin_memory'])


    model = eval(config['arch'])(config['model'])

    ### TANMAEY ###
    # param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    # print(param_size)
    # for name,p in model.named_parameters():
    #     print(p.dtype)
    #     raise
    


    if initial_checkpoint is not None:
        print('Loading initial model weights from: {}'.format(initial_checkpoint))
        checkpoint = torch.load(initial_checkpoint)
        model.load_state_dict(checkpoint['state_dict'])


    model.summary()

    loss = eval(config['loss']['type'])
    loss_params = config['loss']['config'] if 'config' in config['loss'] else None
    print ("Using %s with config %s" % (config['loss']['type'], config['loss']['config']))
    metrics = [eval(metric) for metric in config['metrics']]

    trainer = LSTMTrainer(model, loss, loss_params, metrics,
                          resume=resume,
                          config=config,
                          data_loader=data_loader,
                          valid_data_loader=valid_data_loader,
                          train_logger=train_logger)

    trainer.train()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(
        description='Learning DVS Image Reconstruction')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-i', '--initial_checkpoint', default=None, type=str,
                        help='path to the checkpoint with which to initialize the model weights (default: None)')

    args = parser.parse_args()

    config = None
    if args.resume is not None:
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        if args.initial_checkpoint is not None:
            logger.warning(
                'Warning: --initial_checkpoint overriden by --resume')
        config = torch.load(args.resume)['config']
    if args.config is not None:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        # breakpoint()
        # if args.resume is None:
        #     assert not os.path.exists(path), "Path {} already exists!".format(path)
        if os.path.exists(path):
            print(f"[INFO] Removing existing experiment directory: {path}")
            shutil.rmtree(path) 


    assert config is not None

    main(config, args.resume, args.initial_checkpoint)
