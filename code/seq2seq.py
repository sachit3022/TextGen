# seq2seq.py

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import models
from bertviz import model_view

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
        
        
        #if not  hasattr(self.model, 'encoder'): # this is for SSM to tie weights ( recommended in the original file but didnot find any improvements.)
        #    self.output.weight =self.embedding.weight 

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

    """
    #logging gradients to observe training.
    def on_after_backward(self):
        global_step = self.global_step
        for name, param in self.model.named_parameters():
            #self.logger.experiment.add_histogram(name, param, global_step)
            if param.requires_grad :
                self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)
    """

    def training_step(self, batch, _):
        
        self.model.train()
        inputs, targets = batch
        batch_size = inputs.size(0)

        start_token = torch.ones(batch_size, device=self.device).long().unsqueeze(1) * self.data_prop['start_token']  # BS x 1 --> 16x1  CHECKED        
        if hasattr(self.model, 'encoder'): # this is for transformer
            embedded = self.embedding(inputs)
            enc_attn,encoder_annotations = self.model.encoder(embedded)
            decoder_inputs = torch.cat([start_token, targets[:, 0:-1]], dim=1)  # Gets decoder inputs by shifting the targets to the right
            decoder_inputs = self.embedding(decoder_inputs)
            dec_attn,decoder_outputs = self.model.decoder(decoder_inputs, encoder_annotations)
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
        
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True,batch_size=targets_flatten.size(0))

        acc = self.accuracy_batch(batch,_)
        self.log('train_acc',  acc, prog_bar=True,  on_step=False, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, _):

        self.model.eval()
        
        inputs, targets = batch
        batch_size = inputs.size(0)
                
        start_token = torch.ones(batch_size, device=self.device).long().unsqueeze(1) * self.data_prop['start_token']  # BS x 1 --> 16x1  CHECKED        
        if hasattr(self.model, 'encoder'): # this is for transformer
            embedded = self.embedding(inputs)
            enc_attn,encoder_annotations = self.model.encoder(embedded)
            decoder_inputs = torch.cat([start_token, targets[:, 0:-1]], dim=1)  # Gets decoder inputs by shifting the targets to the right
            decoder_inputs = self.embedding(decoder_inputs)
            dec_attn,decoder_outputs = self.model.decoder(decoder_inputs, encoder_annotations)
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
        
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True,batch_size=targets_flatten.size(0))
        
        acc = self.accuracy_batch(batch,_)
        self.log('val_acc',  acc, prog_bar=True, on_step=False, on_epoch=True, logger=True)

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

    def accuracy_batch(self,batch,_):
        
        self.model.eval()
        inputs, targets = batch
        max_context_len = targets.size(-1)
        batch_size = inputs.size(0)        
        acc = 0
        start_token = torch.ones(batch_size, device=self.device).long().unsqueeze(1) * self.data_prop['start_token']  # BS x 1 --> 16x1  CHECKED        
        
        if hasattr(self.model, 'encoder'): # this is for transformer
            embedded = self.embedding(inputs)
            enc_attn,encoder_annotations = self.model.encoder(embedded)
            decoder_inputs = start_token
 
                ## slow decoding, recompute everything at each time
            for i in range(max_context_len):
                decoder_emb = self.embedding(decoder_inputs)
                dec_attn, decoder_outputs = self.model.decoder(decoder_emb, encoder_annotations)
                output = self.output(decoder_outputs)
                
                generated_words = output.max(2)[1]
                decoder_inputs = torch.cat([decoder_inputs, generated_words[:,-1:]], dim=1)
                
                end_token_reached = decoder_inputs[:,-1] == self.data_prop['end_token']
                
                if decoder_inputs.size(-1)-1 == targets.size(-1):
                    acc += (decoder_inputs[end_token_reached,1:] == targets[end_token_reached,:]).all(dim=-1).sum().item()
    
                decoder_inputs = decoder_inputs[~ end_token_reached]
                encoder_annotations = encoder_annotations[~ end_token_reached]
                targets = targets[~end_token_reached]
                if not decoder_inputs.size(0):
                    break
            
            """
            #Script to check which words are wrong generally the large wordsa re inccorect compared to the smaller ones. 
            if decoder_inputs.size(0) and (self.current_epoch>60 and self.current_epoch%10==0):
   
                gen_string = "".join(
                        [self.data_prop['index_to_char'][int(item)]
                        for item in decoder_inputs.cpu().numpy().reshape(-1)])
                gen_words = gen_string.split('SOS')[1:]
                

                target_string = "".join(
                        [self.data_prop['index_to_char'][int(item)]
                        for item in targets.cpu().numpy().reshape(-1)])
                target_words = target_string.split('EOS')[:-1]


                for s,t in zip(gen_words,target_words):
                    self.logger.experiment.add_text(t, s, global_step=self.current_epoch)
            """
    

        return acc/batch_size

    def translate(self, word):

        #auto regressive translation.
        self.model.eval()
        gen_string = ''
        indexes = torch.Tensor(self.string_to_index_list(word)).long().to(self.device).unsqueeze(0)
       
        start_token = torch.Tensor([[self.data_prop['start_token']]]).long().to(self.device) # For BS = 1

        if hasattr(self.model, 'encoder'): # this is for transformer
            embedded = self.embedding(indexes)
            enc_attn,encoder_annotations = self.model.encoder(embedded)
            decoder_inputs = start_token 
            for i in range(self.max_generated_chars):
                ## slow decoding, recompute everything at each time
                decoder_emb = self.embedding(decoder_inputs)
                dec_attn,decoder_outputs = self.model.decoder(decoder_emb, encoder_annotations)
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
            
            html_model_view = model_view(
                encoder_attention=enc_attn,
                decoder_attention=dec_attn[0],
                cross_attention=dec_attn[1],
                encoder_tokens= [self.data_prop['index_to_char'][int(item)] for item in indexes[0].tolist()],
                decoder_tokens = [self.data_prop['index_to_char'][int(item)] for item in decoder_inputs[0,:-1].tolist()],
                html_action='return'
            )
            with open(f"{self.logger.log_dir}/{self.current_epoch}_{word}_model_view.html", 'w') as file:
                file.write(html_model_view.data)

        else: # this is for SSM
            all = torch.cat([indexes, start_token], dim=1)
            decoder_inputs = all
            for i in range(self.max_generated_chars):
                ## slow decoding, recompute everything at each time
                decoder_inputs = self.embedding(decoder_inputs)
                decoder_outputs = self.model(decoder_inputs)
                output = self.output(decoder_outputs)[:, all.size(1)-1:, :]

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

