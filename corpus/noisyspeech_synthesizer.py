import glob
import numpy as np
import soundfile as sf
import os
import argparse
import configparser as CP
import random
import numpy as np
from tqdm import tqdm
import shutil
from audiolib import audioread, audiowrite, snr_mixer


def main(cfg):
    snr_lower = float(cfg["snr_lower"])
    snr_upper = float(cfg["snr_upper"])
    total_snrlevels = int(cfg["total_snrlevels"])
    
    clean_dir = os.path.join(os.path.dirname(__file__), 'clean_train')
    if cfg["speech_dir"] != 'None':
        clean_dir = cfg["speech_dir"]
    if not os.path.exists(clean_dir):
        assert False, ("Clean speech data is required")

    noise_dir = os.path.join(os.path.dirname(__file__), 'noise_train')
    if cfg["noise_dir"] != 'None':
        noise_dir = cfg["noise_dir"]
    if not os.path.exists(noise_dir):
        assert False, ("Noise data is required")
        
    fs = float(cfg["sampling_rate"])
    audioformat = cfg["audioformat"]
    noise_audioformat = cfg["noise_audioformat"]
    total_hours = float(cfg["total_hours"])
    audio_length = float(cfg["audio_length"])
    silence_length = float(cfg["silence_length"])
        
    total_secs = total_hours*60*60
    total_samples = int(total_secs * fs)
    audio_length = int(audio_length * fs)
    SNR = np.linspace(snr_lower, snr_upper, total_snrlevels)
    print(f"SNR : {SNR}")
    cleanfilenames = glob.glob(os.path.join(clean_dir, f"**/{audioformat}"), recursive=True)
    if cfg["noise_types_excluded"]=='None':
        noisefilenames = glob.glob(os.path.join(noise_dir, f"**/{noise_audioformat}"), recursive=True)
    else:
        filestoexclude = cfg["noise_types_excluded"].split(',')
        noisefilenames = glob.glob(os.path.join(noise_dir, f"**/{noise_audioformat}"), recursive=True)
        for i in range(len(filestoexclude)):
            noisefilenames = [fn for fn in noisefilenames if not os.path.basename(fn).startswith(filestoexclude[i])]
    if not cfg["random_noise"]:
        print(f"noisefilenames: {noisefilenames}")
        for noisefile in noisefilenames:
            noisyspeech_dir = os.path.join(os.path.dirname(__file__), "../data/" f"libri_test_noise_{SNR[0]}", noisefile.split("/")[-1].split(".")[0])
            for i in range(np.size(SNR)):
                for cleanfile in cleanfilenames:
                    clean, fs = audioread(cleanfile)
                    noise, fs = audioread(noisefile)

                    noiseconcat = noise
                    while len(noiseconcat) <= len(clean):
                        noiseconcat = np.append(noiseconcat, noise)
                    noise = noiseconcat
                    if len(noise)>=len(clean):
                        noise = noise[0:len(clean)]

                    clean_snr, noise_snr, noisy_snr = snr_mixer(clean, noise, SNR[i])
                    noisyfilename = f"{cleanfile.split('/')[-1]}_{noisefile.split('/')[-1].split('.')[0]}_SNR_{str(SNR[i])}.wav"
                    noisypath = os.path.join(noisyspeech_dir, noisyfilename)
                    audiowrite(noisy_snr, fs, noisypath, norm=False)
    else:
        noisefilenames.append("Gaussian")
        print(f"noisefilenames: {noisefilenames}")
        
        noisyspeech_dir = os.path.join(os.path.dirname(__file__), "../data/" f"libri_test_noise_{SNR[0]}")
        # Adjust SNR for each kind of noise
        # For human noise background, Whisper tend to higher WER
        SNR_dict = {
            'res/ShuttingDoor_6.wav': 0,
            'Gaussian': 0,
            'res/AirConditioner_6.wav': 10,
            'res/Typing_2.wav': 10,
            'res/CopyMachine_2.wav': 10,
            'res/AirportAnnouncements_2.wav': 10,
            'res/VacuumCleaner_1.wav': 20,
            'res/Babble_4.wav': 20,
            'res/Neighbor_6.wav': 20,
            'res/Munching_3.wav': 20,
        }
        for cleanfile in tqdm(cleanfilenames):
            noisefile = random.choice(noisefilenames)
            clean, fs = audioread(cleanfile)
            if noisefile == "Gaussian":
                noise = np.random.randn(*clean.shape)
            else:
                noise, fs = audioread(noisefile)

                noiseconcat = noise
                while len(noiseconcat) <= len(clean):
                    noiseconcat = np.append(noiseconcat, noise)
                noise = noiseconcat
                if len(noise)>=len(clean):
                    noise = noise[0:len(clean)]
            

            clean_snr, noise_snr, noisy_snr = snr_mixer(clean, noise, SNR_dict[noisefile])
            noisyfilename = f"{cleanfile.split('/')[-1]}_{noisefile.split('/')[-1].split('.')[0]}_SNR_{SNR_dict[noisefile]}.wav"
            noisypath = os.path.join(noisyspeech_dir, noisyfilename)
            audiowrite(noisy_snr, fs, noisypath, norm=False)

        # Copy label.txt into data_dir for self-define noise corpus
        shutil.copy(os.path.join(os.path.dirname(__file__), "../res/" "label.txt"), noisyspeech_dir)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./noisyspeech_synthesizer.cfg", help="Read noisyspeech_synthesizer.cfg for all the details")
    parser.add_argument("--cfg_str", type=str, default="noisy_speech")
    args = parser.parse_args()

    cfgpath = os.path.join(os.path.dirname(__file__), args.cfg)
    assert os.path.exists(cfgpath), f"No configuration file as [{cfgpath}]"
    cfg = CP.ConfigParser()
    cfg._interpolation = CP.ExtendedInterpolation()
    cfg.read(cfgpath)

    main(cfg._sections[args.cfg_str])