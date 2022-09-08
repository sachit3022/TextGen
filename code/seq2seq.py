# seq2seq.py

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import models

class Seq2Seq(pl.LightningModule):
    def __init__(self, args, dataloader):
        super().__init__()
        self.args = args

        if dataloader is not None:
            self.val_dataloader = dataloader.val_dataloader
            self.test_dataloader = dataloader.test_dataloader
            self.train_dataloader = dataloader.train_dataloader

        self.encoder = getattr(models, args.encoder)(
            dataloader.vocab_size, args.hidden_size, args.num_layers, args.attention_type)
        self.decoder = getattr(models, args.decoder)(
            dataloader.vocab_size, args.hidden_size, args.num_layers, args.attention_type)
        self.loss = torch.nn.CrossEntropyLoss()

        self.start_token = dataloader.idx_dict['start_token']
        self.end_token = dataloader.idx_dict['end_token']
        self.char_to_index = dataloader.idx_dict['char_to_index']
        self.index_to_char = dataloader.idx_dict['index_to_char']
        self.max_generated_chars = 20
        self.test_sentence = 'the air conditioning is working'

        self.save_hyperparameters()

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams,
        {
            "hp/metric_lr": self.args.learn_rate,
            "hp/metric_hs": self.args.hidden_size,
        })

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        batch_size = inputs.size(0)
        encoder_annotations, encoder_hidden = self.encoder(inputs)

        # The last hidden state of the encoder becomes the first hidden state of the decoder
        decoder_hidden = encoder_hidden

        decoder_input = torch.ones(batch_size).long().unsqueeze(1) * self.start_token  # BS x 1 --> 16x1  CHECKED

        decoder_inputs = torch.cat([decoder_input, targets[:, 0:-1]], dim=1)  # Gets decoder inputs by shifting the targets to the right
        decoder_outputs, attention_weights = self.decoder(decoder_inputs, encoder_annotations, decoder_hidden)
        decoder_outputs_flatten = decoder_outputs.view(-1, decoder_outputs.size(2))
        targets_flatten = targets.view(-1)

        loss = self.loss(decoder_outputs_flatten, targets_flatten)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        batch_size = inputs.size(0)
        encoder_annotations, encoder_hidden = self.encoder(inputs)

        # The last hidden state of the encoder becomes the first hidden state of the decoder
        decoder_hidden = encoder_hidden

        decoder_input = torch.ones(batch_size).long().unsqueeze(1) * self.start_token  # BS x 1 --> 16x1  CHECKED

        decoder_inputs = torch.cat([decoder_input, targets[:, 0:-1]], dim=1)  # Gets decoder inputs by shifting the targets to the right
        decoder_outputs, attention_weights = self.decoder(decoder_inputs, encoder_annotations, decoder_hidden)
        decoder_outputs_flatten = decoder_outputs.view(-1, decoder_outputs.size(2))
        targets_flatten = targets.view(-1)

        loss = self.loss(decoder_outputs_flatten, targets_flatten)
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_epoch_end(self, outputs):
        gen_string = self.translate_sentence(self.test_sentence)
        self.logger.experiment.add_text(self.test_sentence, gen_string, global_step=self.current_epoch)

    def string_to_index_list(self, s):
        """Converts a sentence into a list of indexes (for each character).
        """
        return [self.char_to_index[char] for char in s] + [self.end_token]  # Adds the end token to each index list

    def translate_sentence(self, sentence):
        return ' '.join([self.translate(word) for word in sentence.split()])

    def translate(self, word, visualize_attention=False):

        gen_string = ''
        indexes = torch.LongTensor(self.string_to_index_list(word)).unsqueeze(0)
        encoder_annotations, encoder_last_hidden = self.encoder(indexes)

        decoder_hidden = encoder_last_hidden
        decoder_input = torch.LongTensor([[self.start_token]]) # For BS = 1
        decoder_inputs = decoder_input

        for i in range(self.max_generated_chars):
            ## slow decoding, recompute everything at each time
            decoder_outputs, attention_weights = self.decoder(decoder_inputs, encoder_annotations, decoder_hidden)

            generated_words = F.softmax(decoder_outputs, dim=2).max(2)[1]
            ni = generated_words.cpu().numpy().reshape(-1)  # LongTensor of size 1
            ni = ni[-1] #latest output token

            decoder_inputs = torch.cat([decoder_input, generated_words], dim=1)
            if ni == self.end_token:
                break
            else:
                gen_string = "".join(
                    [self.index_to_char[int(item)]
                    for item in generated_words.cpu().numpy().reshape(-1)])

        if visualize_attention == True and attention_weights is not None:
            produced_end_token = False
            if isinstance(attention_weights, tuple):
                ## transformer's attention mweights
                attention_weights, self_attention_weights = attention_weights

            all_attention_weights = attention_weights.data.cpu().numpy()
            for i in range(len(all_attention_weights)):
                attention_weights_matrix = all_attention_weights[i].squeeze()
                fig = plt.figure()
                ax = fig.add_subplot(111)
                cax = ax.matshow(attention_weights_matrix, cmap='bone')
                fig.colorbar(cax)

                # Set up axes
                ax.set_yticklabels([''] + list(word) + ['EOS'], rotation=90)
                ax.set_xticklabels([''] + list(gen_string) +
                                (['EOS'] if produced_end_token else []))

                # Show label at every tick
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
                # Add title
                plt.xlabel(
                    'Attention weights to the source sentence in layer {}'.format(i+1))
                plt.tight_layout()
                plt.grid('off')
                self.logger.experiment.add_figure('Fig-' + word + '-' + 'layer {}'.format(i+1), plt.gcf())

        return gen_string

    def configure_optimizers(self):
        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = getattr(torch.optim, self.args.optimizer)(
            filter(lambda p: p.requires_grad, parameters), lr=self.args.learn_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.args.lr_decay)
        return [optimizer], [scheduler]
