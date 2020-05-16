# Fine-Tuning BERT for Text Summarization 


## Getting Started 

## Installation 
* Clone the repository 
```
git clone https://github.com/Omkar-Ranadive/Fine-Tuning-BERT
```
* Install Pytorch by following the instructions given on their [website.](https://pytorch.org/get-started/locally/)
* Install the remaining dependencies by going into the project directory and running the following command: 
```
pip install -r requirements.txt
```
* Download the pre-processed data from here (Credits to Yang Liu): https://drive.google.com/open?id=1x0d61LP9UAN389YN00z0Pv-7jQgirVg6

## Training the model 
* To download the BERT model, run the following code once 
```
python train.py -mode train -encoder classifier -dropout 0.1 -bert_data_path ../bert_data/cnndm -model_path ../models/bert_classifier -lr 2e-3 -visible_gpus 0  -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 100 -batch_size 1000 -decay_method noam -train_steps 100 -accum_count 2 -log_file ../logs/bert_classifier -use_interval true -warmup_steps 5
``` 

## References
* BertSum Repository: https://github.com/nlpyang/BertSum
* Hugging Face Transformers: https://github.com/huggingface/transformers

