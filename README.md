# Silent Breaks Prediction

In this repository, we present Pauzee, a silent pause prediction system.
The goal of this project is to predict when it sounds better for a reader to take breaks
in speech while reading a text.

The code contained in this github directory is associated with the work described in the paper 
"_Pauzee : Prédiction des pauses dans la lecture d’un texte_" accepted at TALN 2023.

## Setting
First, clone the github repository.  
After, you have to run a Docker container and start an interactive bash session.
Think to modify the Makefile, especially to modify the variables:
DOCKER_IMAGE_NAME & DOCKER_CONTAINER_NAME (line 2 and 3).
You can also specify which GPU(s) you want to use (line 9).
Then, run the following commands:

    `$ make build`
    `$ make run-bash`

Warning: the build will download the model used (2Go). 
Delete the last three lines of the dockerfile if you do not want to download it.

## How to try it?
This code can be tested by passing in input the sentence you wish to test. 
To do so, in your docker, just run the script cli.py in test mode with the model (_models/version_17-02-2023--pauzee_) after "--model" 
and the text you want to test after "--prompt". 
Here an example of the commandline :

   `$ poetry run python break_detection/code/cli.py test --model models/version_17-02-2023--pauzee --test --prompt "ceci est un test pour illustrer le systeme Il montre quelles sont les pauses predites"`


## How to test the code presented in the paper?

### Corpus
We mainly use the following two corpora:
* [SynpaFlex](http://synpaflex.irisa.fr/) - not available online.
* [French Oral Narrative Corpus](https://www.ortolang.fr/market/corpora/cefc-orfeo?path=%2Foral%2Ffrenchoralnarrative) :
  this corpora can be download [here](https://www.ortolang.fr/market/corpora/cefc-orfeo?path=%2Foral) (2,52 GB). Just click on "_frenchoralnarrative_" and then on "_Download_". Unzip it.
  To be compatible with our code you have to keep only files with the .orfeo extension located here in the download repository:
  `cefc-orfeo/11/oral/frenchoralnarrative/`
  For the following, this dataset will be considered as stored at this location.: `break_detection/data/oralNarrative/`

### Code
We are adapting code of Unbabel presented at [SEPP 2021](https://sites.google.com/view/sentence-segmentation) to the task of breath prediction.
Here the link for the [initial code](https://github.com/Unbabel/caption/tree/shared-task) & the [linked paper](https://repositorio.iscte-iul.pt/bitstream/10071/23672/1/conferenceobject_82723.pdf).

The code shared here is a modified version of Unbabel, it allows to predict pauses instead of punctuation.
This code is composed of 3 bricks: 1) preprocessing of the data to put them in Unbabel format and to infer the breaks they contain, 2) the pause prediction and, finally, 3) evaluation of this system.  

#### 1) Preprocessing of the dataset

To work properly the automatic prediction code needs a dataset provided in a particular format. 
This dataset must contain one word per line. Each line is composed of three columns separated by a comma: 
the first one concerns the word, 
the second one contains a value (0 or 1) indicating the absence or presence of a pause after this word and, 
to finish, the last one indicates the length of the pause. 
0 for no break, 1 for a short break, 2 for a medium break and 3 for a long break.

| word    | isBreak | catBreak |
|---------|---------|----------|
| ça      | 0       | 0        |
| n'      | 0       | 0        |
| existe  | 0       | 0        |
| pas     | 0       | 0        |
| ici     | 1       | 2        |
| les     | 0       | 0        |
| métiers | 0       | 0        |
| faciles | 1       | 1        |

The purpose of the "formatCorpus.py" script is to take data in oral Narrative (*.orfeo) and synpaflex format and convert them to the format needed to run our system.
  
    # Oral Narrative : 
    poetry run python break_detection/code/formatDataset/formatCorpus.py --input break_detection/data/oralNarrative/ --output break_detection/data/oralNarrative_annotated/ --data_type orfeo
    # SynPaFlex : 
    poetry run python break_detection/code/formatDataset/formatCorpus.py --input break_detection/data/synpaflex/ --output break_detection/data/synpaflex_annotated/ --data_type synpaflex

This script takes as argument the directory containing the files to be modified (--input), 
the directory in which we want to store the output of the script (--output) 
and the type of file passed in option (--data_type) - oralNarrative has the type "orfeo"

In the output directory 3 directories will be built : 
a train, a test and a dev (which will be used for learning and prediction).

#### 2) Prediction of the pauses
In order to make this system works, you will have to use a trained model to predict breaks.

The **model** used is automatically downloaded when you build the container .
You can also find it ([here (model_pauzee.tar.gz)](https://github.com/deezer/pauzee_taln23/releases/download/v1.0.0/model_pauzee.tar.gz)) 
or re-trained it following the procedure explained below.

Then, you can use this model to **predict break in a text** with this script:

    poetry run python break_detection/code/cli.py test --model models/version_17-02-2023--pauzee/ --test --dataset break_detection/data/oralNarrative_annotated/ --prediction_dir break_detection/results/orfeo-synpaflex_pauzee/

This script allows to generate breaks in a given dataset.  
It takes as input several parameters:
* --model : the model used to make the prediction of breaks
* --test: which allows it to know the task it should perform (predicting breaks)
* --dataset : the path to the directory which contains the files (.csv) to be analysed. 
The files in this folder must contain one word per line. 
It need to be in unbabel format with 3 columns separated with a comma (e.g. "word,0,0"). 
The first line is ignored.
* --prediction_dir : the path to the directory in which we want to write the results

The results will be written in this directory given as argument.
Each output file will be composed of 3 columns. 
The first column corresponds to the word, 
the second to the pauses (0 = no pause and 1 = presence of pauses) and 
the last one to the lengths of these pauses (0 = no pause, 1, short pause, 2 = medium pause and 3 = long pause)  
_Note that these two columns are not correlated: the first one displays the results of a binary classification and 
the second one the results of a multiclass classification_.


**If you want to train your own model, you can use this command line :**

    $ poetry run python break_detection/code/cli.py train -f break_detection/code/configs/optuna-large.yaml  --dataset break_detection/data/oralNarrative_annotated/

For the training of the model, the script 'break_detection/code/cli.py' is used in train mode.
It needs two arguments:
* --config or -f : the configuration of the for optuna-large
* --dataset : the dataset used for fine-tuningor optuna-large

After running, this script writes the model in the 'models' directory (it will be named like that: _version_date--hour_). 
If you want to save it outside the docker, just copy it to the _break_detection_ repository.


#### 3) Evaluation

The evaluation is after done with this script :

    `poetry run python break_detection/code/evaluation/evaluation.py --gold_path break_detection/data/oralNarrative_annotated/test/ --predicted_path "break_detection/results/orfeo-synpaflex_pauzee/" --subtask break`

To work properly, this file must take into account the directory containing the gold file(s), 
those containing the pauses predicted by pauzee and what we want to evaluate 
(the pause prediction "break" or the pause length prediction "break_size"). 
_Be cautious : to reproduce the evaluation done with Unbabel, it is the "punct2break" subtask 
that must be used to consider the predicted punctuation as a break_.

## Cite

```
@inproceedings{epure2020muzeeglot,
  title={Pauzee: Prédiction des pauses dans la lecture d'un texte},
  author={Baranes, Marion and Hayek, Karl and Hennequin, Romain and Epure, Elena V},
  booktitle={},
  pages={},
  year={2023},
  organization={ATALA}
```