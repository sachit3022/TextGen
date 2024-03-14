# seq2seq.py

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import models

class Seq2Seq(pl.LightningModule):
    def __init__(self, args, dataloader):
        super().__init__()
        self.args = args

        if dataloader is not None:
            self.val_dataloader = dataloader.val_dataloader
            self.test_dataloader = dataloader.test_dataloader
            self.train_dataloader = dataloader.train_dataloader

        self.vocab_size = dataloader.vocab_size
        self.hidden_size = args.model_options['hidden_size']
        
        self.embedding = getattr(torch.nn, args.embedding)(self.vocab_size, self.hidden_size)
        self.model = getattr(models, args.model_type)(**args.model_options)
        self.output = torch.nn.Linear(self.hidden_size, self.vocab_size)
        self.loss = getattr(torch.nn, args.loss_type)(**args.loss_options)

        self.data_prop = dataloader.idx_dict
        self.max_generated_chars = dataloader.max_seq_len
        self.test = dataloader.test

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams,
        {
            "hp/metric_lr": self.args.learning_rate,
            "hp/metric_hs": self.hidden_size,
        })

    def training_step(self, batch, _):
        
        self.model.train()
        inputs, targets = batch
        batch_size = inputs.size(0)

        start_token = torch.ones(batch_size, device=self.device).long().unsqueeze(1) * self.data_prop['start_token']  # BS x 1 --> 16x1  CHECKED        
        if hasattr(self.model, 'encoder'): # this is for transformer
            embedded = self.embedding(inputs)
            encoder_annotations = self.model.encoder(embedded)
            decoder_inputs = torch.cat([start_token, targets[:, 0:-1]], dim=1)  # Gets decoder inputs by shifting the targets to the right
            decoder_inputs = self.embedding(decoder_inputs)
            decoder_outputs = self.model.decoder(decoder_inputs, encoder_annotations)
            outputs_flatten = decoder_outputs.view(-1, decoder_outputs.size(2))
        else: # this is for SSM
            all = torch.cat([inputs, start_token, targets[:, :-1]], dim=1)
            annotations = self.embedding(all)
            outputs = self.model(annotations)
            outputs = outputs[:, -targets.size(1):, :]
            outputs_flatten = outputs.reshape(-1, outputs.size(2))
        
        outputs_flatten = self.output(outputs_flatten)
        targets_flatten = targets.view(-1)
        
        loss = self.loss(outputs_flatten, targets_flatten)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, _):
        self.model.eval()
        
        inputs, targets = batch
        batch_size = inputs.size(0)
                
        start_token = torch.ones(batch_size, device=self.device).long().unsqueeze(1) * self.data_prop['start_token']  # BS x 1 --> 16x1  CHECKED        
        if hasattr(self.model, 'encoder'): # this is for transformer
            embedded = self.embedding(inputs)
            encoder_annotations = self.model.encoder(embedded)
            decoder_inputs = torch.cat([start_token, targets[:, 0:-1]], dim=1)  # Gets decoder inputs by shifting the targets to the right
            decoder_inputs = self.embedding(decoder_inputs)
            decoder_outputs = self.model.decoder(decoder_inputs, encoder_annotations)
            outputs_flatten = decoder_outputs.view(-1, decoder_outputs.size(2))
        else: # this is for SSM
            all = torch.cat([inputs, start_token, targets[:, :-1]], dim=1)
            annotations = self.embedding(all)
            outputs = self.model(annotations)
            outputs = outputs[:, -targets.size(1):, :]
            outputs_flatten = outputs.reshape(-1, outputs.size(2))
            
        
        outputs_flatten = self.output(outputs_flatten)
        targets_flatten = targets.view(-1)


        loss = self.loss(outputs_flatten, targets_flatten)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)


        return loss

    def on_validation_epoch_end(self):
        gen = self.generate_sequence(self.test)
        print(self.test,gen)
        if isinstance(self.test, str):
            self.logger.experiment.add_text(self.test, gen, global_step=self.current_epoch)
        else:
            self.logger.experiment.add_image(self.test, gen, global_step=self.current_epoch)

    def string_to_index_list(self, s):
        """Converts a sentence into a list of indexes (for each character).
        """
        return [self.data_prop['char_to_index'][char] for char in s] + [self.data_prop['end_token']]  # Adds the end token to each index list

    def generate_sequence(self, sentence):
        if isinstance(sentence, str):
            output= ' '.join([self.translate(word) for word in sentence.split()])
            
            return output

    def translate(self, word):
        self.model.eval()
        gen_string = ''
        indexes = torch.Tensor(self.string_to_index_list(word)).long().to(self.device).unsqueeze(0)
       
        start_token = torch.Tensor([[self.data_prop['start_token']]]).long().to(self.device) # For BS = 1
        if hasattr(self.model, 'encoder'): # this is for transformer
            embedded = self.embedding(indexes)
            encoder_annotations = self.model.encoder(embedded)
            decoder_inputs = start_token 
            for i in range(self.max_generated_chars):
                ## slow decoding, recompute everything at each time
               
                decoder_emb = self.embedding(decoder_inputs)
                decoder_outputs = self.model.decoder(decoder_emb, encoder_annotations)
                output = self.output(decoder_outputs)
                
                generated_words = F.softmax(output, dim=2).max(2)[1]
                ni = generated_words.cpu().numpy().reshape(-1)  # LongTensor of size 1
                ni = ni[-1] #latest output token

                decoder_inputs = torch.cat([decoder_inputs, generated_words[:,-1:]], dim=1)
                if ni == self.data_prop['end_token']:
                    break
                else:
                    gen_string = "".join(
                        [self.data_prop['index_to_char'][int(item)]
                        for item in generated_words.cpu().numpy().reshape(-1)])
        else: # this is for SSM
            all = torch.cat([indexes, start_token], dim=1)
            decoder_inputs = all
            for i in range(self.max_generated_chars):
                ## slow decoding, recompute everything at each time
                decoder_inputs = self.embedding(decoder_inputs)
                decoder_outputs = self.model(decoder_inputs)
                output = self.output(decoder_outputs)[:, all.size(1)-1:, :]
                # output = self.output(decoder_outputs)

                generated_words = F.softmax(output, dim=2).max(2)[1]
                ni = generated_words.cpu().numpy().reshape(-1)  # LongTensor of size 1
                ni = ni[-1] #latest output token

                decoder_inputs = torch.cat([all, generated_words], dim=1)
                if ni == self.data_prop['end_token']:
                    break
                else:
                    gen_string = ''.join(
                        [self.data_prop['index_to_char'][int(item)]
                        for item in generated_words.cpu().numpy().reshape(-1)])
       
        return gen_string

    def configure_optimizers(self):
        parameters = self.model.parameters()
        out = {}
        
        out['optimizer'] = getattr(torch.optim, self.args.optim_method)(
            filter(lambda p: p.requires_grad, parameters),
            lr=self.args.learning_rate, **self.args.optim_options)
        
        if self.args.scheduler_method is not None:
            out['scheduler'] = getattr(torch.optim.lr_scheduler, self.args.scheduler_method)(
                out['optimizer'], **self.args.scheduler_options
            )
            out['monitor'] = self.args.scheduler_monitor
        
        return out