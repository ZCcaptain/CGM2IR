*CETA: A Conﬁdence Enhanced Training Approach for Denoising in Distantly Supervised Relation Extraction*



The information of our mainly hardware environment is:

1. Tesla V100 32GB
2. CPU  Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz
3. RAM 256GM  

The version of python is: Python 3.6.12 and the implementation steps are as follows:

1.  Install dependency python package  
    `pip install -r requirements.txt`
2.  Prepare the dataset, the pre-trained language model, and the NERToolKit 
    1. The dataset is the most widely-used NYT10 shown in the paper, This dataset is opensource and public to everyone. Here we donnot provide the download link, because of the double blind principle. We recommend you to download the NYT10 dataset via the project `opennre`, which is an opensource project. The downloaded NYT10 training file, validation file. test file should be named as `nyt10_train.txt`, `nyt10_val.txt`, `nyt10_val.txt` and in the `benchmark` folder. Note that the instance format in the download NYT10 should be:  
       `["text": "", "relation": "", "h": {"id": "m.071cn", "name": "Syracuse", "pos": [96, 104]}, "t": {"id": "m.02_v74", "name": "Onondaga Lake", "pos": [75, 88]}},...,]`

    2. The version of pre-trained language model adopted by this paper is the `bert-base-uncased`, which is opensource and public to everyone. Here we donnot provide the download link, because of the double blind principle.
    3. Our relation extraction model needs to use the entity type to construct the special token for encoding the sentence in the entity-aware way, which is presented in our paper. However, the raw NYT10 dataset downloaded from the project `opennre` does not contain the entity type in each instance. The NYT10 dataset is constructed by the opensource NER toolkit StandfordNER. Thus, we use the same StanfordNER toolkit to annotate the entity type for each instance. Note that the StandfordNER Toolkit is implemented by the JAVA language. You should set up the environment which required by StandFordNER. Besides, the resource package `english.all.3class.distsim.crf.ser.gz` and `stanford-ner.jar` adopted by the opensource StandFordNER should be download.
    4. Then you can set up the config of the relevant path in the file `whole_config.py`.

3.  Process the dataset
    1. When finishing the step1 and step2, you can process the data by executing the following command:  
       `sh process_data.sh`

4.  Training procedure
    1. When finishing the step1, step2 and step3, you can train our proposed model by excuting the folloing command:  
       `sh train_ceta.sh`
5.  Testing procedure
    1. When finishing the step1, step2, step3 and step4, you can test our proposed model by excuting the function `test_ceta()`  defined in the file `train_ceta.py`