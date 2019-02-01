# Complex Imperative Program Induction From Terminal Rewards (CIPITR)
This repository contains the implementation of the program induction model proposed in the TACL paper **Complex Program Induction for Querying Knowledge Bases in the
Absence of Gold Programs** and links to download associated datasets. 

Currently this code only handles program induction where the input variables to the program are `gold` i.e. for example if KBQA requires entity, relation type linking on the query before program induction, this code sends the oracle entity, relation, type linker's output to CIPITR.

# Datasets
Datasets on Complex Question answering on Knowledge Bases, used for evaluating CIPITR
1. Complex Sequential Question Answering (https://amritasaha1812.github.io/CSQA/) Dataset
2. WebQuestionsSP (https://www.microsoft.com/en-us/download/details.aspx?id=52763) Dataset

# Experiments on CQA
* **Step 1:** Inside the CSQA_TACL_FINAL folder, create a folder named *data*

* **Step 2:** To Download Preprocessed Dataset:
  * For experiments on full CQA download 
    1. *preprocessed_data_full.zip* (https://drive.google.com/file/d/1jkMk2ReeGd6x5wzU8SOHlAwsPkzIBLf2/view?usp=sharing)
  * For experiments on subset of CQA having 10k QA pairs per type download 
    1. *preprocessed_data_alltypes_10k.zip* (https://drive.google.com/file/d/1BHkGU_9fHXC0fTTrvxsQrA2TiDYBVmCt/view?usp=sharing)
    2. *preprocessed_data_alltypes_10k_noisy.zip* (https://drive.google.com/file/d/1q4qomyYrLNG_2JOUxBMBVsToDl0JsRXI/view?usp=sharing)
  * For experiments on subset of CQA having 100 QA pairs per type download 
    1. *preprocessed_data_alltypes_100.zip* (https://drive.google.com/file/d/1JnzvSL7QKVNORdOYBE0qhRq7shvg7NsE/view?usp=sharing)
    2. *preprocessed_data_alltypes_100_noisy.zip* (https://drive.google.com/file/d/10HqWeSQEeicRHsh2lEjkib5Cw75FJ_vH/view?usp=sharing)
* **Step 3:** Put the preprocessed data inside the *data* folder 

* **Step 4:** To Download Preprocessed Knowledge Base:
For experiments on full CQA or subset of CQA having 10K questions per type, download full preprocessed *wikidata.zip* (https://drive.google.com/file/d/1_whud6L-VmYsFMDSSjw6pW7oPzojPp01/view?usp=sharing)
For experiments on subset of CQA having 100 questions per type, download the preprocessed version of the corresponding subset of wikidata, i.e. *wikidata_100.zip* (https://drive.google.com/file/d/1yInB34aS7GyUuSd7F8nz7ATSrMr3LyQv/view?usp=sharing)

* **Step 5:** For running any of the tensorflow scripts, go inside CSQA_TACL_FINAL/code/NPI/tensorflow and install the dependencies by running `$pip install -r requirements.txt`
* **Step 6:** Similarly, for running any of the pytorch scripts,  go inside CSQA_TACL_FINAL/code/NPI/pytorch and install the dependencies by running `$pip install -r requirements.txt`

<!--## Experiments on the gold entity, relation, type linking data-->
* **Step 7:** For running the experiments on CQA (or any subset of CQA) with the gold entity, relation, type linking, we recommend using the tensorflow version. 

* **Step 8:** To do so go inside *CSQA_TACL_FINAL/code/NPI/tensorflow/gold_WikiData_CSQA* folder

* **Step 9:** Each of the experiments are configured with a parameter file (in the parameter folder).  There are seven question types (simple, logical, verify, quanti, quanti_count, comp, comp_count) and each of the variants can be run on either the smaller subset of the dataset ( i.e. CQA subset with 100 QA pairs per question type) or the full dataset. For e.g. for running on the *simple* question type on CQA-100 subset, use the parameter file parameters_simple_small.json and to run on full CQA dataset, use the parameter file parameters_simple_big.json (*small* is for 100 QA pair subset of CQA and *big* is for full CQA)

* **Step 10:** Create a folder *model*. 

* **Step 11:** To run training on any of the question categories (simple/logical/verify/quanti/comp/quanti_count/comp_count) run `python train.py <parameter_file> <time-stamp>` (time-stamp is the ID of the current experiment run). Example script to run is in run.sh. This script will start the training as well as dump the trained model in the model and also run validation. 
* **Step 12:** To load the trained model and run test, run `python load.py <parameter_file> <time-stamp>` (use the same ID as used during training)

* **Step 13:** To download pre-trained models and log files:
	* Download model.zip from https://drive.google.com/file/d/1rRZcDhWaRnv3BtZuwC0rFl8yrgElgM47/view?usp=sharing and extract
	* model/gold_WikiData_CSQA contains model.zip and out.zip
	* model.zip contains all the models trained on the 7 query types
	* out.zip contains all the log files of the training on the 7 query types

* For *e.g.* to train and test the tensorflow code on *simple* question type on 100-QA pair subset of CQA:
	* `cd CSQA_TACL_FINAL/code/NPI/tensorflow/gold_WikiData_CSQA`
	* `python train.py  parameters/parameters_simple_small.json small_Jan_7` *#this will create a folder model/simple_small_Jan_7 to dump the trained model*
	* `python load.py parameters/parameters_simple_small.json small_Jan_7` *#this will run the trained model on the test data, as mentioned in the parameter file*




<!-- 
## Experiments on the noisy entity, relation, type linking data
* **Step 14:** For running the experiments on CQA (or any subset of CQA) with the noisy entity, relation, type linking, we recommend using the pytorch code.

* **Step 15:** To do so go inside *CSQA_TACL_FINAL/code/NPI/pytorch/noisy_WikiData_CSQA* folder

* **Step 16:** Each of the experiments are configured with a parameter file (in the parameter folder). There are three question types (simple, logical, quanti_count) in the parameter folder and each of the variants can be run on either the smaller subset of the dataset (i.e. CQA with 100 QA pairs per question type) or on the bigger subset (CQA with 10K QA pairs per question type). For e.g. for running on the *simple* question type on CQA-100 subset, use the parameter file parameters_simple_small.jso and to run on the CQA-10K dataset use the parameter file parameters_simple_big.json (*small* is for 100 QA pair subset of CQA and *big* is for 10K QA pair subset of CQA)

* **Step 17:** Create a folder *model*

* **Step 18:** To run the SRP model on any of the question categories (simple/logical/quanti_count) run `python train_SRP.py <parameter_file> <time_stamp>` (time-stamp is the ID of the current experiment run). Example script to run is in *run_SRP.sh*. This script will start the training as well as dump the trained model in the model and also run validation. 

* **Step 19:** To run the SSRP model on any of the question categories (simple/logical/quanti_count) run `python train_SSRP.py <parameter_file> <time_stamp>` (time-stamp is the ID of the current experiment run). Example script to run is in *run_SSRP.sh*. This script will start the training as well as dump the trained model in the model and also run validation. 

* **Step 20:** To load the trained model and run test, run `python load.py <parameter_file> <time-stamp>` (use the same ID as used during training)

For *e.g.* to train and test the tensorflow code on *simple* question type on 100-QA pair subset of CQA:
1. `cd CSQA_TACL_FINAL/code/NPI/pytorch/noisy_WikiData_CSQA`
2. `python train_SRP.py  parameters/parameters_simple_small.json SRP_Jan_7` *#this will create a folder model/simple_SRP_Jan_7 to dump the trained model*
3. `python load.py parameters/parameters_simple_big.json SRP_Jan_7` *#this will run the trained model on the big test data, as mentioned in the parameter file*
4. `python train_SSRP.py parameters/parameters_simple_small.json SSRP_Jan_7` *#this will create a folder model/simple_SSRP_Jan_7 to dump the trained model*
5. `python load.py parameters/parameters_simple_big.json SSRP_Jan_7` *#this will run the trained model on the big test data, as mentioned in the parameter file*
-->   

# Experiments on WebQuestionsSP
* **Step 1:** For experiments on the WebQuestionsSP dataset, download the preprocessed version of the dataset and the corresponding subset of freebase,  i.e. *freebase_webqsp.zip* (https://drive.google.com/file/d/1CuV4QJxknTqDmAaLwBfO1kyNW7IXTd1Q/view?usp=sharing)

* **Step 2:** For running any of the tensorflow scripts, go inside *CSQA_TACL_FINAL/code/NPI/tensorflow* and install the dependencies by running `$pip install -r requirements.txt`

* **Step 3:** Similarly, for running any of the pytorch scripts,  go inside *CSQA_TACL_FINAL/code/NPI/pytorch* and install the dependencies by running `$pip install -r requirements.txt`

* **Step 4:** Go inside *code/NPI/pytorch/gold_FB_webQuestionsSP* folder. 

* **Step 5:** Each of the experiments are configured with a parameter file (in the parameters folder). The experiments on the gold entity, relation, type (ERT) linking data have parameters inside the *parameters/gold* folder and the experiments on the noisy ERT linking data have parameters inside the *parameters/noisy* folder. There are five categories of questions, *1infc* and *1inf* (i.e. questions with inference chain length 1, with and without additional non-temporal constraint), *2infc* and *2inf* (i.e. questions with inference chain length 2, with and without additional non-temporal constraint), *c_date* (i.e. questions with temporal constraints, having inference chain of any length). Accordingly parameter files in the *gold* folder are named parameters_\<category\>.json. For e.g. to run an experiment on questions with length-1 inference chain and no constraint with gold ERT linker data, use the parameter file *parameters/gold/parameters_1inf.json*

* **Step 6:** Create a folder *model*

* **Step 7:** To run training on any of the question categories (1inf/1infc/2inf/2infc/c_date) run `python train.py <parameters_file> <time-stamp>` (time-stamp is the ID of the current experiment run). Example script to run is in run.sh. This script will start the training as well as dump the trained model in the model and also run validation. 

* **Step 8:** To load the trained model and run test, run `python load.py <parameter_file> <time-stamp>` (use the same ID as used during training)

* **Step 9:** To download pre-trained models and log files:
	* Download model.zip from https://drive.google.com/file/d/1rRZcDhWaRnv3BtZuwC0rFl8yrgElgM47/view?usp=sharing and extract
	* model/gold_FB_webQuestionsSP contains model.zip and out.zip
	* model.zip contains all the models trained on the 4 query types
	* out.zip contains all the log files of the training on the 4 query types

* For *e.g.* to train and test the pytorch code on *1inf* question type on WebQuestionsSP:
	* `cd CSQA_TACL_FINAL/code/NPI/pytorch/gold_FB_webQuestionsSP`
	* `python train.py parameters/gold/parameters_1inf.json Jan_7` *#this will create a folder model/1inf_Jan_7 to dump the trained model*
	* `python load.py parameters/gold/parameters_1inf.json Jan_7` *#this will run the trained model on the test data with gold ERT linking, as mentioned in the parameter file* 
	* `python load.py parameters/noisy/parameters_1inf.json Jan_7` *#this will run the trained model on the test data with noisy ERT linking, as mentioned in the parameter file* 	
# RL Environment for CQA Dataset
We have also provided a simple RL environment for doing **Question answering over the CQA dataset using Wikidata Knowledge base**.
* **Step 1:** The RL environment is located at **NPI/RL_ENVIRONMENT_CSQA/code/** directory.
* **Step 2:** To incorporate the environment one has to simply import the **environment.py** file.
* **Step 3:** To instantiate an environment you will need to input a parameter file. Sample parameter files are located in the **parameters** folder. 
* **Step 4** A detailed and sufficient instruction on using and instantiating an environment object is provided in the **sample_env_usage.ipynb** notebook.
 
