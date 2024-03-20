# AI Detection Project

finetunes a pretrained model to predict if a text is AI generated or not. 
two datasets are available in the code: CHEAT (academic abstract) and Tweepfake (Tweets)


# Project File Structure
```commandline
.
├── classifiers
│ ├── AbstractClassifier.py
│ ├── LLMClassifier.py
│ └── TweetClassifier.py
├── data
│ ├── cheat
│ └── tweepfake
├── main.py
├── runners.py
├── README.md
├── requirements.txt
├── run.sub
├── sic_cluster
│ ├── conda_run.sh
│ ├── environment.yml
│ ├── README.md
│ ├── run.sh
│ ├── setup.sh
│ └── setup.sub
└── utils
    ├── parse_trees.py
    └── utils.py
```


# Running Project
to run the baseline (off the shelf), then train the model:

```commandline
python main.py --baseline True
python main.py --epoch 3 --per_device_train_batch_size 8 --learning_rate 5e-5
```

see `sic_cluster/run.sh` file for more examples - or to modify jobs submitted to cluster

to run on the CS department clusters:
```
condor_submit sic_cluster/setup.sub     # set up environment and install requirements
condor_submit run.sub                   # run job
```

outputs are located in `sic_cluster/logs/`
