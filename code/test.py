# main.py

import os

import pytorch_lightning as pl
import pytorch_lightning.callbacks as cbs
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import config
import dataloaders
from seq2seq import Seq2Seq

if __name__ == '__main__':
    
    args = config.parse_args()
    pl.seed_everything(args.manual_seed)


    dataloader = getattr(dataloaders, args.dataset)(**args.dataset_options)
    dir = "/research/hal-gaudisac/Deep_learning/homework-2-sachit3022/code/logs/Seq2Seq/"
    
    checkpoint = f"{dir}version_146/checkpoints/Seq2Seq-epoch=034-val_loss=0.161.ckpt"
    model = Seq2Seq.load_from_checkpoint(checkpoint,args=args, dataloader = dataloader)
    
    # add yopur sentences to the list below
    TEST_SENTENCES = [
        'is conditioning pleasure-grounds'
    ]
    for sentence in TEST_SENTENCES:
        translated = model.generate_sequence(sentence)
        print("\nsource:\t\t{} \ntranslated:\t{}".format(sentence, translated))