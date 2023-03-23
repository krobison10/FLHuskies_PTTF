# FLHuskies_PTTF

Repository to share scripts and data for the Pushback to the Future competition.

Used by the team FLHuskies out of the University of Washington Tacoma.

## Members

Faculty:
  - Dr. Martine De Cock
  - Dr. Anderson Nascimento


PhD Students:
  - Steven Golob
  - Sikha Pentyala


Undergraduates:
  - Kyler Robison
  - Yudong Lin
  - David Huynh
  - Jeff Maloney
  - Daniil Filienko
  - Anthony Nguyen
  - Trevor Tomlin

## Data Directory
This repository does not contain the data as it is too large.

Scripts operate under the assumption that there is a directory named "_data" in the 
root of the repository.

It has an underscore so that it stays out of the way up at the very top.

Furthermore, they assume that the directory has a structure as follows:

```
_data
├── <airport>
│   ├── features
│   │   ├── <airport>_config.csv
│   │   ├── <airport>_etd.csv
│   │   ├── <airport>_first_position.csv
│   │   ├── <airport>_lamp.csv
│   │   ├── <airport>_mfs.csv
│   │   ├── <airport>_runways.csv
│   │   ├── <airport>_standtimes.csv
│   │   ├── <airport>_tbfm.csv
│   │   └── <airport>_tfm.csv
│   └── train_labels_<airport>.csv
├── submission_format.csv
├── ...
```

Currently, the scripts in this repository only work with decompressed tables. In time, they will be modified
to work with compressed tables if a command line argument is supplied.

If it is desired to work with compressed tables to save storage space, each airport directory in the 
data folder should appear as follows:

```
├── <airport>
│   ├── <airport>_config.csv.bz2
│   ├── <airport>_etd.csv.bz2
│   ├── <airport>_first_position.csv.bz2
│   ├── <airport>_lamp.csv.bz2
│   ├── <airport>_mfs.csv.bz2
│   ├── <airport>_runways.csv.bz2
│   ├── <airport>_standtimes.csv.bz2
│   └── <airport>_tbfm.csv.bz2
└── train_labels_<airport>.csv.bz2
```

Scripts that can read and use these compressed tables should be supplied a single command line argument "compressed".


## CSV Files

All raw .csv files in the entire project are excluded by .gitignore, except for compressed (.csv.bz2) files.
