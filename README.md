# Fake Review Detection Project

## Set up environment

1. Create virtual environment: 
```bash
virtualenv -venv
```

2. Activate virtual environment:
```bash
./venv/Scripts/activate
```

3. While virtual environment is activated, install required modules
```bash
pip install -r requirements.txt
```

## Load Dataset
1. Download [YelpCSV folder](https://drive.google.com/drive/folders/1MexwmQTjEbz7UTQ2E_fPxLntE2ugeXtV?usp=share_link), [dataset_v1.csv](https://drive.google.com/file/d/1JzGEI0XdXk1l61vo3giNWUZykJzfRJKw/view?usp=share_link), and [review_feature_table.csv](https://drive.google.com/file/d/1QalYUu1pbkHglrF96GQw_0kL3thj8s17/view?usp=share_link) into DATA subdirectory.
*Note: Program can be executed without downloading dataset_v1.csv and review_feature_table.csv beforehand. This will force program to re-compute all features and conduct undersampling, which will add to overall execution time.
```
.
├── DATA
     └── YelpCSV
            └── metadata.csv
            └── productIdMapping.csv
            └── reviewContent.csv
            └── reviewGraph.csv
            └── userIdMapping.csv
     └── dataset_v1.csv
     └── review_feature_table.csv

```

## Execute program
1. Make sure your virtual environment is activated AND your current working directory is the general repo directory (ex. path/to/dir/fake_review_detection) and run command:
```bash
python ./CODE/main.py
```

## After Executing
All necessary metrics and visualizations will populate in EVALUATIONS subdirectory.