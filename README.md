## Improving Document-level Relation Extraction via Context Guided Mention Integration and Inter-pair Reasoning



The information of our mainly hardware environment is:

1.  NVIDIA RTX 3090 24GB
2. CPU  Intel(R) Xeon(R) Silver 4180 CPU @ 1.80GHz*32
3. RAM 32GM  

The version of python is: Python 3.8.8 and the implementation steps are as follows:

### 1.  Install dependency python package  
    `pip install -r requirements.txt`
### 2.  Prepare the dataset

The [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset can be downloaded following the instructions at [link](https://github.com/thunlp/DocRED/tree/master/data). The CDR and GDA datasets can be obtained following the instructions in [edge-oriented graph](https://github.com/fenchri/edge-oriented-graph). The expected structure of files is:
```
CGM2IR
|-- dataset
|    |-- docred
|    |    |-- train_annotated.json
|    |    |-- train_distant.json
|    |    |-- dev.json
|    |    |-- test.json
|    |-- cdr
|    |    |-- train_filter.data
|    |    |-- dev_filter.data
|    |    |-- test_filter.data
|    |-- gda
|    |    |-- train.data
|    |    |-- dev.data
|    |    |-- test.data
|-- meta
|    |-- rel2id.json
```
Note that the instance format in the download DocRED should be:  
```
    Data Format:
    {
    'title',
    'sents':     [
                    [word in sent 0],
                    [word in sent 1]
                ]
    'vertexSet': [
                    [
                        { 'name': mention_name, 
                        'sent_id': mention in which sentence, 
                        'pos': postion of mention in a sentence, 
                        'type': NER_type}
                        {anthor mention}
                    ], 
                    [anthoer entity]
                    ]
    'labels':   [
                    {
                    'h': idx of head entity in vertexSet,
                    't': idx of tail entity in vertexSet,
                    'r': relation,
                    'evidence': evidence sentences' id
                    }
                ]
    }
```
### 3. Process the dataset
When finishing the step1 and step2, you can process the data by executing the following command:  
```
sh process_DocRED.sh
sh process_CDR.sh
sh process_GDA.sh
```

### 4.Training procedure
When finishing the step1, step2 and step3, you can train our proposed model by excuting the folloing command:  
Train the BERT model on DocRED with the following command:
```
>> sh scripts/run_bert.sh  # for BERT
>> sh scripts/run_roberta.sh  # for RoBERTa
```

Train CDA and GDA model with the following command:
```
>> sh scripts/run_cdr.sh  # for CDR
>> sh scripts/run_gda.sh  # for GDA
```
The training loss and evaluation results on the dev set are synced to the wandb dashboard.