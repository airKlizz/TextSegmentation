**To do:**
- [ ] Try others sentences embedding models or transformers models
- [ ] Create CNN dataset and compore results

# Text segmentation task on Wikinews data

Wikinews passages dataset and code to train a segmenter model to find passages segmentation from continuous text.

| Table of contents |
| ----------------- |
| Part 1. |

## Dataset

The dataset in composed of 18997 Wikinews articles segmented in passages according to the author of the news. All the data is in the ``data.jsonl`` file. The ``wikinews/`` folder contains scripts to convert the ``.jsonl`` file into train and test files.

### Reproduce the dataset

``data.jsonl`` file is composed by an article per line. The article is saved in json format:

```
{"title": "Title of the article", "date": "Weekday, Month Day, Year", "passages": ["Passage 1.", "Passage 2.", ...]}
```

You can create your own ``data.jsonl`` file by running the following command:

```
cd wikinews/
python create_data.py --num 40000 \
                      --output "data.jsonl"
```
*Remark: ``--num`` is the number of wikinews articles to use. The final number of data is less than this number because certain articles are depreciated*

### Create train and test files

To make the training easier, I recommend to transform the ``data.jsonl`` file into ``train.txt`` and ``test.data.txt`` and ``test.gold.txt`` by running the following command:

```
cd wikinews/
python create_train_test_data.py --input "data.jsonl" \
                                 --train_output "train.txt" \
                                 --test_data_output "test.data.txt" \
                                 --test_gold_output "test.gold.txt"
```

*Remark: 80/20 training/test split*

These files contain one sentence per line with the corresponding label (1 if the sentence is the beginning of a passage, 0 otherwise). The passages are segmented into sentences using `` sent_tokenize`` from the `` nltk`` library. Articles are separated by ``\n\n``. Sentences are separated by ``\n``. Elements of a sentence are separated by ``\t``.

```
Article 1
Sentence  1 Text of the sentence 1. label
Sentence  2 Text of the sentence 2. label
...

Article 2
Sentence  1 Text of the sentence 1. label
Sentence  2 Text of the sentence 2. label
...

...
```

``train.txt`` contains all elements of sentences.

``test.data.txt`` does not contain the label element of sentences.

``test.gold.txt`` does not contain the text element of sentences.

## Task

Given a continuous text composed a sentences S1, S2, S3, ..., the segmenter model has to find which sentences are the beginning of a passage. And so give in output a set of passages P1, P2, P3, ... composed by the sentences in the same order. 

The objective is that passages contain one information. In the best case passages are self-contained passages and do not require an external context to be understood. 

## My Model

The model is composed of a [pre-trained sentence encoder from tf hub](https://tfhub.dev/google/universal-sentence-encoder-large/5) follows by a recurrent layer on each sentence and then a classification layer. 

<img src="model/model.png" alt="Architecture of the model" width="800"/>

## Results

The result and the model weight are obtained after a training with parameters :

  - ``learning_rate = 0.001``
  - ``batch_size = 12``
  - ``epochs = 8``
  - ``max_sentences = 64`` (Number max of sentences per article)

|  | Precision | Recall | Fscore |
| --- | ----------- | --- | ----------- |
| My Model | 0.761 | 0.757 | 0.758 | 

Saved weights of the model available [here]().


## Getting started

### Installation

#### Create a virtual environnement and activate it:

```
python3 -m venv textsegmentation_env
source textsegmentation_env/bin/activate
```


#### Install all dependencies:

```
pip install -r requirements.txt
```


#### Download data:

You can download the ``data.jsonl`` file [here](https://drive.google.com/open?id=1E3mfjgL3Z-r8hNGXMrclsLTlBBEyYpFy). Otherwise you can recreate the ``data.jsonl`` file (See [above](#reproduce-the-dataset)). Then move the file to ``wikinews/`` and run ``create_train_test_data.py`` (See [above](#create-train-and-test-files)).

### Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()

```
python train.py --learning_rate 0.001 \
                --max_sentences 64 \
                --epochs 8 \
                --train_path "wikinews/train.txt"
```

To see full usage of ``train.py``, run ``python train.py --help``.

### Evaluation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()

### Use pre-trained models
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()

## License
[MIT](https://choosealicense.com/licenses/mit/)
