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
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [cbs.RichProgressBar(),lr_monitor]
    
    print("starting main")
    
    print(args.dataset)
    if args.save_results:
        logger = TensorBoardLogger(
            save_dir=args.logs_dir,
            log_graph=True,
            name=args.project_name
        )
        checkpoint = cbs.ModelCheckpoint(
            dirpath=os.path.join(args.save_dir, args.project_name),
            filename=args.project_name + '-{epoch:03d}-{val_loss:.3f}',
            monitor='val_loss',
            save_top_k=args.checkpoint_max_history,
            save_weights_only=True
            )
        enable_checkpointing = True
        callbacks.append(checkpoint)
    else:
        logger=False
        checkpoint=None
        enable_checkpointing=False
    
    if args.swa:
        callbacks.append(cbs.StochasticWeightAveraging())

    dataloader = getattr(dataloaders, args.dataset)(**args.dataset_options)
    model = Seq2Seq(args, dataloader)

    trainer = pl.Trainer(
        devices='auto',
        accelerator='auto',
        strategy='auto',
        benchmark=True,
        callbacks=callbacks,
        logger=logger,
        min_epochs=1,
        max_epochs=args.nepochs,
        precision=args.precision,
        reload_dataloaders_every_n_epochs=1,
        enable_checkpointing=enable_checkpointing,
        gradient_clip_val=1.0
        )
    
    pl.Trainer()

    trainer.fit(model)
    
    # add yopur sentences to the list below
    TEST_SENTENCES = [
        'the air conditioning is working',
        'the quick brown fox',
    ]
    for sentence in TEST_SENTENCES:
        translated = model.generate_sequence(sentence)
        print("\nsource:\t\t{} \ntranslated:\t{}".format(sentence, translated))

    # visualize_attention_maps
    TEST_WORDS = [
        'street',
    ]
    for word in TEST_WORDS:
        model.translate(word)