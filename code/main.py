# main.py

import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from seq2seq import Seq2Seq
from dataloader import PigLatinData

if __name__ == '__main__':
    # parse the arguments
    parser = argparse.ArgumentParser(description="Train Seq2Seq")
    parser.add_argument('--ngpu', default=0, help="number of GPUs for training")
    parser.add_argument('--data_file', choices=["data/pig_latin_small.txt", "data/pig_latin_large.txt"], help="dataset to run")
    parser.add_argument('--encoder', choices=["GRUEncoder", "TransformerEncoder"], help="encoder to run")
    parser.add_argument('--decoder', choices=["GRUDecoder", "GRUAttentionDecoder", "TransformerDecoder"], help="decoder to run")
    parser.add_argument('--hidden_size', default=32, type=int, help="hidden size")
    parser.add_argument('--num_layers', default=3, type=int, help="number of transformer layers")
    parser.add_argument('--attention_type', choices=["AdditiveAttention", "ScaledDotAttention"], help="attention type")
    parser.add_argument('--learn_rate', default=5e-4, type=float, help="Learning rate")
    parser.add_argument('--lr_decay', default=0.99, type=float, help="learning rate decay")
    parser.add_argument('--batch_size', default=64, type=int, help="Batch size")
    parser.add_argument('--nepochs', default=100, type=int, help="Number of epochs to train")
    parser.add_argument('--seed', default=0, type=int, help="Numpy random seed")
    parser.add_argument('--save_dir', default='results/checkpoints', help="directory to save results")
    parser.add_argument('--logs_dir', default='results/logs', help="directory to save results")
    parser.add_argument('--data_dir', default='data/', help="directory to save downloaded data")
    parser.add_argument('--project_name', default='Seq2Seq', help="Name of the Project")
    parser.add_argument('--precision', default=32, help="Precision for training. Options are 32 or 16")
    parser.add_argument('--optimizer', default='AdamW', help="Optimization method")

    args = parser.parse_args()

    if (args.ngpu == 0):
        args.device = 'cpu'

    pl.seed_everything(args.seed)

    name = 'results/h{}-bs{}-{}-{}-{}'.format(args.hidden_size, args.batch_size,
                                      args.encoder, args.decoder, args.data_file)
    logger = TensorBoardLogger(
        save_dir=os.path.join(name, args.logs_dir),
        log_graph=True,
        name=args.project_name
    )

    dataloader = PigLatinData(args)
    model = Seq2Seq(args, dataloader)

    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(name, args.save_dir),
        filename=args.project_name + '-{epoch:03d}-{val_loss:.6f}',
        monitor='val_loss',
        save_top_k=5)

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
        callbacks=[checkpoint],
        checkpoint_callback=True,
        logger=logger,
        min_epochs=1,
        max_epochs=args.nepochs,
        precision=args.precision,
        reload_dataloaders_every_epoch=True
        )

    trainer.fit(model)

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
