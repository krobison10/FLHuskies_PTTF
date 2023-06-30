# FLHuskies_PTTF

Repository to share scripts and data for:

[Pushback to the Future: Predict Pushback Time at US Airports]: https://www.drivendata.org/competitions/group/competition-nasa-airport-pushback/

Used by team FLHuskies from the University of Washington Tacoma.

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
├── private
│   ├── <airport>
│   │   ├── <airport>_AAL_mfs.csv  # contains rows for AAL airline, all columns
│   │   ├── <airport>_AAL_standtimes.csv  # contains rows for ALL, all columns
│   │   ├── ...
│   │   ├── <airport>_UPS_mfs.csv  # contains rows for UPS airline, all columns
│   │   └── <airport>_UPS_standtimes.csv
│   ├── ...
├── public
│   ├── <airport>
│   │   ├── <airport>_config.csv
│   │   ├── <airport>_etd.csv
│   │   ├── <airport>_first_position.csv
│   │   ├── <airport>_lamp.csv
│   │   ├── <airport>_mfs.csv
│   │   ├── <airport>_runways.csv
│   │   ├── <airport>_standtimes.csv
│   │   ├── <airport>_tbfm.csv
│   │   └── <airport>_tfm.csv
│   ├── ...
│   ├── train_labels_open
│   │   ├── train_labels_<airport>.csv
│   │   └── ...
│   ├── train_labels_prescreened
│   │   ├── prescreened_train_labels_<airport>.csv
│   │   └── ...
│   ├── train_labels_phase2
│   │   ├── phase2_train_labels_<airport>.csv
│   │   └── ...
└── submission_format.csv
```

If it is desired to work with compressed tables to save storage space, the directory should appear as follows:

```
_data
├── private
│   ├── <airport>
│   │   ├── <airport>_AAL_mfs.csv.bz2  # contains rows for AAL airline, all columns
│   │   ├── <airport>_AAL_standtimes.csv.bz2  # contains rows for ALL, all columns
│   │   ├── ...
│   │   ├── <airport>_UPS_mfs.csv.bz2  # contains rows for UPS airline, all columns
│   │   └── <airport>_UPS_standtimes.csv.bz2
│   ├── ...
├── public
│   ├── <airport>
│   │   ├── <airport>_config.csv.bz2
│   │   ├── <airport>_etd.csv.bz2
│   │   ├── <airport>_first_position.csv.bz2
│   │   ├── <airport>_lamp.csv.bz2
│   │   ├── <airport>_mfs.csv.bz2
│   │   ├── <airport>_runways.csv.bz2
│   │   ├── <airport>_standtimes.csv.bz2
│   │   ├── <airport>_tbfm.csv.bz2
│   │   └── <airport>_tfm.csv.bz2
│   ├── ...
│   ├── train_labels_open
│   │   ├── train_labels_<airport>.csv.bz2
│   │   └── ...
│   ├── train_labels_prescreened
│   │   ├── prescreened_train_labels_<airport>.csv.bz2
│   │   └── ...
│   ├── train_labels_phase2
│   │   ├── phase2_train_labels_<airport>.csv.bz2
│   │   └── ...
└── submission_format.csv.bz2
```

Scripts that can read and use these compressed tables should be supplied a single command line argument "compressed".
Some scripts are built to automatically use the compressed files if no uncompressed versions are found.


## CSV Files

All raw .csv files in the entire project are excluded by .gitignore, except for compressed (.csv.bz2) files.
