import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

import json
import pandas as pd
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

# WandB – Import the wandb library
# import wandb

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, sp_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.sp_len = sp_len
        self.text = self.data.sp
        self.ctext = self.data.context_preprocess

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt', truncation=True)
        target = self.tokenizer.batch_encode_plus([text], max_length= self.sp_len, pad_to_max_length=True,return_tensors='pt', truncation=True)

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }






def preprocess(context):
  new_sentences = ["".join(x[1]) for x in context]
  new_context = " ".join(new_sentences)
  return new_context

def supporting_facts(f):
  sp = []
  articles = dict()
  for key, val in f["context"]:
    articles[key] = val
  for i in f["supporting_facts"]:
    # print(i)
    # print(articles[i[0]][i[1]])
    if(len(articles[i[0]])>=i[1]+1):
      art = articles[i[0]][i[1]]
      sp.append(articles[i[0]][i[1]])
    # else:
      # print(i)
      # print(articles[i[0]])
      # print(i[1])
  return " ".join(sp)

def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for _,data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
        loss = outputs[0]
        
        if _%10 == 0:
        	print(f'Epoch: {epoch}, Loss:  {loss.item()}')
            # wandb.log({"Training Loss": loss.item()})

        # if _%500==0:
        #     print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # xm.optimizer_step(optimizer)
        # xm.mark_step()

def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=150, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            if _%100==0:
                print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals

def main():
    # WandB – Initialize a new run

    train_df = pd.read_json(r'./data/hotpot_train_v1.1.json')
    test_df = pd.read_json(r'./data/hotpot_dev_distractor_v1.json')

    source_len=4000
    sp_len=500

    train_df["context_preprocess"] = train_df["context"].apply(preprocess)
    test_df["context_preprocess"] = test_df["context"].apply(preprocess)
    train_df["sp"] = train_df.apply(supporting_facts, axis=1)
    test_df["sp"] = test_df.apply(supporting_facts, axis=1)
    train_df = train_df[["context_preprocess", "sp"]]
    test_df = test_df[["context_preprocess", "sp"]]
    print(train_df["context_preprocess"].map(lambda x: len(x.split(" "))).max())
    print(test_df["context_preprocess"].map(lambda x: len(x.split(" "))).max())

    train_df_small = train_df[:50]
    test_df_small = test_df[:20]

    # wandb.init(project="transformers_tutorials_summarization")

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    # Defining some key variables that will be used later on in the training  
    # config = wandb.config          # Initialize config
    config = dict()
    config["TRAIN_BATCH_SIZE"] = 5    # input batch size for training (default: 64)
    config["VALID_BATCH_SIZE"] = 5    # input batch size for testing (default: 1000)
    config["TRAIN_EPOCHS"] = 2        # number of epochs to train (default: 10)
    config["VAL_EPOCHS"] = 1 
    config["LEARNING_RATE"] = 1e-4    # learning rate (default: 0.01)
    config["SEED"] = 42               # random seed (default: 42)
    config["MAX_LEN"] = 4000
    config["SUMMARY_LEN"] = 250 

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(config["SEED"]) # pytorch random seed
    np.random.seed(config["SEED"]) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    # tokenizer = T5Tokenizer.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    

    # Importing and Pre-Processing the domain data
    # Selecting the needed columns only. 
    # Adding the summarzie text in front of the text. This is to format the dataset similar to how T5 model was trained for summarization task. 
    # df = pd.read_csv('./data/news_summary.csv',encoding='latin-1')
    # df = df[['text','ctext']]
    # train_df_small.sp = 'extract: ' + train_df_small.sp
    # test_df_small.sp = 'extract: ' + test_df_small.sp
    print(train_df_small.head())

    
    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest will be used for validation. 
    train_size = 0.8
    train_dataset=train_df_small.sample(frac=train_size,random_state = config["SEED"])
    val_dataset=train_df_small.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(train_df_small.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("VALID Dataset: {}".format(val_dataset.shape))

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = CustomDataset(train_dataset, tokenizer, config["MAX_LEN"], config["SUMMARY_LEN"])
    val_set = CustomDataset(val_dataset, tokenizer, config["MAX_LEN"], config["SUMMARY_LEN"])

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': config["TRAIN_BATCH_SIZE"],
        'shuffle': True,
        'num_workers': 0
        }

    val_params = {
        'batch_size': config["VALID_BATCH_SIZE"],
        'shuffle': False,
        'num_workers': 0
        }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)


    
    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary. 
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    # model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model = model.to(device)

    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=config["LEARNING_RATE"])

    # Log metrics with wandb
    # wandb.watch(model, log="all")
    # Training loop
    print('Initiating Fine-Tuning for the model on our dataset')

    for epoch in range(config["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, device, training_loader, optimizer)


    # Validation loop and saving the resulting file with predictions and acutals in a dataframe.
    # Saving the dataframe as predictions.csv
    print('Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe')
    for epoch in range(config["VAL_EPOCHS"]):
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
        final_df.to_csv('./models/predictions.csv')
        print('Output Files generated for review')

if __name__ == '__main__':
    main()

