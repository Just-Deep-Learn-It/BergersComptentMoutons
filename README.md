# CSRNet

#script part for cropping is in Generating_ground_truth_density.ipynb
#still didn't crop the test set ( script to be modified)
# Still didn't set aside the 10% validation set
#Will try to do the training tonight
P.S: Au bout de plusieurs tutos opencv, oui j'ai dû basculer en anglais ;)

## Data
The original ShanghaiTech datasets consists of large `.ppm` images of scenes with bounding box coordinates for the traffic signs. We use here a post-processed variant where signs have already been cropped out from their corresponding images and resized to 32 x 32. 

The ShanghaiTech dataset can be found [here](https://github.com/desenzhou/ShanghaiTechDataset).

## Project structure

The project is structured as following:

```bash
.
├── loaders
|   └── dataset selector
|   └── shanghaitech_loader.py # loading and pre-processing shanghaitech data
├── models
|   └── architecture selector
|   └── csrnet.py # CSRNet
├── toolbox
|   └── optimizer and losses selectors
|   └── logger.py  # keeping track of most results during training and storage to static .html file
|   └── metrics.py # code snippets for computing scores and main values to track
|   └── plotter.py # snippets for plotting and saving plots to disk
|   └── utils.py   # various utility functions
├── commander.py # main file from the project serving for calling all necessary functions for training and testing
├── args.py # parsing all command line arguments for experiments
├── trainer.py # pipelines for training, validation and testing
```

## Launching
Experiments can be launched by calling `commander.py` and a set of input arguments to customize the experiments. You can find the list of available arguments in `args.py` and some default values. Note that not all parameters are mandatory for launching and most of them will be assigned their default value if the user does not modify them.

Here is a typical launch command and some comments:

- `python commander.py --name csrnet0 --root-dir ~/BergersComptentMoutons/ShanghaiTech/part_A --batch-size 128 --optimizer sgd --scheduler ReduceLROnPlateau --lr 1e-6 --lr-decay 0.5 --step 15 --epochs 50 --workers 4 --crop-size 32 --criterion crossentropy --tensorboard`
  + this experiment is on the _gtsrb_ dataset which can be found in `--root-dir/gtsrb` trained over _LeNet5_. It optimizes with _sgd_ with initial learning rate (`--lr`) of `1e-3` which is decayed by half whenever the `--scheduler` _ReduceLRonPlateau_ does not see an improvement in the validation accuracy for more than `--step` epochs. Input images are of size 32  (`--crop-size`). In addition it saves intermediate results to `--tensorboard`.
  + when debugging you can add the `--short-run` argument which performs training and validation epochs of 10 mini-batches. This allows testing your entire pipeline before launching an experiment
  + if you want to resume a previously paused experiment you can use the `--resume` flag which can continue the training from _best_, _latest_ or a specifically designated epoch.
  + if you want to use your model only for evaluation on the test set, add the `--test` flag.
  + when using _resnet_ you should upsample the input images to ensure compatibility. To this end set `--crop-size 64`
 
## Output
For each experiment a folder with the same name is created in the folder `root-dir/gtsrb/runs`
 This folder contains the following items:

```bash
.
├── checkpoints (\*.pth.tar) # models and logs are saved every epoch in .tar files. Non-modulo 5 epochs are then deleted.
├── best model (\*.pth.tar) # the currently best model for the experiment is saved separately
├── config.json  # experiment hyperparameters
├── logger.json  # scores and metrics from all training epochs (loss, learning rate, accuracy,etc.)
├── report.html  # html page displaying most useful metrics for the experiment (loss, learning rate, accuracy, confusion matrix, qualitative examples)
    └── pics # plots displayed in the html report
├── res  # predictions for each sample from the validation set for every epoch
├── tensorboard  # experiment values saved in tensorboard format
 ```
