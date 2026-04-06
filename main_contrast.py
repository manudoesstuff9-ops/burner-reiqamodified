"""
DDP training for Contrastive Learning
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.utils.data.distributed
import torch.multiprocessing as mp

from options.train_options import TrainOptions
from learning.contrast_trainer import ContrastTrainer
from networks.build_backbone import build_model
from datasets.util import build_contrast_loader
from memory.build_memory import build_mem

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import moco.optimizer


def main():
    args = TrainOptions().parse()

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if ngpus_per_node <= 1:
        # Single GPU mode or CPU mode
        main_worker(0, 1, args)
        return
    elif args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        raise NotImplementedError('Set multiprocessing_distributed=True for multi-GPU')


def main_worker(gpu, ngpus_per_node, args):

    trainer = ContrastTrainer(args)
    
    # Only init DDP if multi-GPU
    if ngpus_per_node > 1:
        args.distributed = True
        args.multiprocessing_distributed = True
        trainer.init_ddp_environment(gpu, ngpus_per_node)
    else:
        # Single GPU mode - set all rank/local_rank variables to 0
        args.distributed = False
        args.multiprocessing_distributed = False
        args.rank = 0
        args.local_rank = 0
        args.node_rank = 0
        args.ngpus_per_node = 1
        args.local_center = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            args.gpu = 0
            print("Use GPU: 0 for training (single GPU mode)")
        else:
            args.gpu = -1  # CPU mode
            print("CUDA not available - using CPU for training")

    model, model_ema = build_model(args)

    train_dataset, train_loader, train_sampler = \
        build_contrast_loader(args, ngpus_per_node)

    contrast = build_mem(args, len(train_dataset))
    if torch.cuda.is_available():
        contrast.cuda()

    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "AdamW" : 
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)  
    elif args.optimizer == "LARS" : 
        optimizer = moco.optimizer.LARS(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
    else :
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Skip DDP wrapping for single GPU
    if ngpus_per_node == 1:
        if torch.cuda.is_available():
            model.cuda()
            model_ema.cuda()
    else:
        model, model_ema, optimizer = trainer.wrap_up(model, model_ema, optimizer)

    trainer.broadcast_memory(contrast)

    start_epoch = trainer.resume_model(model, model_ema, contrast, optimizer)

    trainer.init_tensorboard_logger()

    for epoch in range(start_epoch, args.epochs + 1):
        if hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)
        trainer.adjust_learning_rate(optimizer, epoch)

        outs = trainer.train(epoch, train_loader, model, model_ema,
                             contrast, criterion, optimizer)

        trainer.logging(epoch, outs, optimizer.param_groups[0]['lr'])

        trainer.save(model, model_ema, contrast, optimizer, epoch)


if __name__ == '__main__':
    main()
