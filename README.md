## Improving Document-level Relation Extraction via Context Guided Mention Integration and Inter-pair Reasoning



The information of our mainly hardware environment is:

1. NVIDIA RTX 3090 24GB
2. CPU  Intel(R) Xeon(R) Silver 4180 CPU @ 1.80GHz*32
3. RAM 32GB  
4. CUDA (tested on 11.3)

The version of python is: Python 3.8.8 and the implementation steps are as follows:

### Step 1.  Install dependency python package  
```
    pip install -r requirements.txt
```
### Step 2.  Prepare the dataset

The [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset can be downloaded following the instructions at [link](https://github.com/thunlp/DocRED/tree/master/data). The CDR and GDA datasets can be obtained following the instructions in [edge-oriented graph](https://github.com/fenchri/edge-oriented-graph). The expected structure of files is:
```
CGM2IR
|-- dataset
|    |-- docred
|    |    |-- train_annotated.json
|    |    |-- train_distant.json
|    |    |-- dev.json
|    |    |-- test.json
|    |    |-- rel_info.json
|    |    |-- label_map.json
|    |-- cdr
|    |    |-- train_filter.data
|    |    |-- dev_filter.data
|    |    |-- test_filter.data
|    |-- gda
|    |    |-- train.data
|    |    |-- dev.data
|    |    |-- test.data
```

### Training and Evaluation
#### DocRED
##### Step 3. Process the DocRED
When finishing the step1 and step2, you can process the data by executing the following command:  
```
sh scripts/process_bert.sh # for BERT
sh scripts/process_roberta.sh # for RoBERTa
```

##### Step 4. Training procedure
When finishing the step1, step2 and step3, you can train our proposed model by excuting the folloing command:  
Train the BERT model on DocRED with the following command:
```
>> sh scripts/run_bert.sh  # for BERT
>> sh scripts/run_roberta.sh  # for RoBERTa
```

#### CDR and GDA
Train CDR and GDA model with the following command:

##### Step 3. Process the Dataset

When finishing the step1 and step2, you can process the data by executing the following command:  
For CDR:
```
>> sh scripts/process_cdr.sh 
```
Or for GDA:
```
>> sh scripts/process_gda.sh 
```

##### Step 4.Training procedure
When finishing the step1, step2 and step3, you can train our proposed model by excuting the folloing command:  
Train the BERT model on CDR or GDA with the following command:
For CDR:
```
>> sh scripts/run_cdr.sh  # for sciBERT
```
Or for GDA:
```
>> sh scripts/run_gda.sh  # for sciBERT
```

The training loss and evaluation results on the dev set are synced to the wandb dashboard.

## Saving and Evaluating Models
You can save the model by setting the `--save_path` argument before training. The model correponds to the best dev results will be saved. After that, You can evaluate the saved model by setting the `--load_path` argument, then the code will skip training and evaluate the saved model on benchmarks. 