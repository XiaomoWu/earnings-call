# Download
The Python script is now on [Github](https://github.com/XiaomoWu/earnings-call/blob/master/C-model-xiao.py) 

I updated the source data so you may need to re-download the following files.

- [`f_sue_keydevid_car_finratio_vol_transcriptid_sim_inflow_revision_sentiment_text_norm.feather`](https://1drv.ms/u/s!AqUu9ylMgqcDgP-6DwvI_gk8qK-Y928?e=LEhkek)  (~2.6 GB) It's too large to be uploaded to Github so you have to download from Onedrive

- [`split_dates.csv`](https://github.com/XiaomoWu/earnings-call/blob/master/data/split_dates.csv)  (~9 k) It's on Github please feel free to download with `curl`

# Create directory
Say you put the `.py` file under the directory `./`, please create the following two folders:
- `./data`. Put all data that you downloaded to this dir, including:
    - `f_sue_keydevid_car_finratio_vol_transcriptid_sim_inflow_revision_sentiment_text_norm.feather`
    - `split_dates.csv`
    - `preembeddings_all_sbert_roberta_nlistsb_encoded.pt`
- `./data/checkpoint`. It's the path for saved model

