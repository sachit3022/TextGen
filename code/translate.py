# main.py

import os
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from seq2seq import Seq2Seq
from code.dataloaders.dataloader import PigLatinData

if __name__ == '__main__':
    # parse the arguments
    parser = argparse.ArgumentParser(description="Train Seq2Seq")
    parser.add_argument('--ngpu', default=0, help="number of GPUs for training")
    parser.add_argument('--data_file', choices=["pig_latin_small.txt", "pig_latin_large.txt"], help="dataset to run")
    parser.add_argument('--encoder', choices=["GRUEncoder", "TransformerEncoder"], help="encoder to run")
    parser.add_argument('--decoder', choices=["GRUDecoder", "GRUAttentionDecoder", "TransformerDecoder"], help="decoder to run")
    parser.add_argument('--attention_type', choices=["AdditiveAttention", "ScaledDotAttention"], help="attention type")
    parser.add_argument('--seed', default=0, type=int, help="Numpy random seed")
    parser.add_argument('--load_checkpoint_file', help="checkpoint to load")
    parser.add_argument('--logs_dir', default='results/logs', help="directory to save results")
    parser.add_argument('--data_dir', default='data/', help="directory to save downloaded data")
    parser.add_argument('--project_name', default='Seq2Seq', help="Name of the Project")
    parser.add_argument('--precision', default=32, help="Precision for training. Options are 32 or 16")
    parser.add_argument('--hidden_size', default=32, type=int, help="hidden size")
    parser.add_argument('--num_layers', default=3, type=int, help="number of transformer layers")

    args = parser.parse_args()

    if (args.ngpu == 0):
        args.device = 'cpu'

    pl.seed_everything(args.seed)

    logger = TensorBoardLogger(
        save_dir=args.logs_dir,
        name=args.project_name
    )

    dataloader = PigLatinData(args)
    model = Seq2Seq(args, dataloader)
    model = Seq2Seq.load_from_checkpoint(args.load_checkpoint_file)

    if args.ngpu == 0:
        accelerator = None
        sync_batchnorm = False
    elif args.ngpu > 1:
        accelerator = 'ddp'
        sync_batchnorm = True
    else:
        accelerator = 'dp'
        sync_batchnorm = False

    trainer = pl.Trainer(
        gpus=args.ngpu,
        accelerator=accelerator,
        sync_batchnorm=sync_batchnorm,
        benchmark=True,
        logger=logger,
        precision=args.precision,
        )

    # add yopur sentences to the list below
    TEST_SENTENCES = [
        'the air conditioning is working',
        'the quick brown fox',
        ]
    for sentence in TEST_SENTENCES:
        translated = model.translate_sentence(sentence)
        print("\nsource:\t\t{} \ntranslated:\t{}".format(sentence, translated))

    # visualize_attention_maps
    TEST_WORDS = [
        'street',
    ]
    for word in TEST_WORDS:
        model.translate(word, visualize_attention=True)
