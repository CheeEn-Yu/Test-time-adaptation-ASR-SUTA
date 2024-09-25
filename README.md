# Exploration on TTA for Speech/Audio to natual language

## Environment
The installation is as follow command:
```
conda create -n tta python=3.10
sudo apt update && sudo apt install ffmpeg
pip install -U openai-whisper
pip install -r requirements.txt
```


## Run the Experiment

change the config.yaml first (There are more detail in config.yaml comment)

```
python whisper_main.py exp_name="<EXP_NAME>"
```
After running the code, the result will be saved under ex_data/<EXP_NAME>.  
In ex_data/<EXP_NAME>, file structure is as follow
```
ex_data/<EXP_NAME>-
    --|- fig/
      |- result.txt
      |- log.txt  
```


main.py is for facebook/s2t-small-librispeech-asr

## Code simple outline
The code structure is as follow:
```

--|- corpus/
  |- whisper/
  |- data.py
  |- suta.py
  |- whisper_main.py
```
In suta.py, the main implementation of SUTA is implemented in **TTADecode** class  

**transcriptionProcessor** class parse the WER from the result.txt
