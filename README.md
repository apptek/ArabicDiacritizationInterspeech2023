# Arabic Text Diacritization

This repository contains the code, configuration setups and model architectures to train, diacritize and evaluate 
Arabic Text Diacritization. The primary purpose is to facilitate the reproduction of our experiments.


data/work/apptek_dubbing/diacritizer/workflow/dataset/CreateHDFDatasetJob.u5ezVjvvD3F6/output/segments_list_u5ezVjvvD3F6 --random_mask 1.0

USAGE INSTRUCTIONS
------------------
Check the individual files for usage instructions.
## Files

### [diacritization](/diacritization)

- constants
  - ARABIC_LETTERS_LIST.pickle - Contains list of Arbaic letters
  - CLASSES_LIST.pickle - Contains list of all possible classes
  - CLASSES_NAMES_LIST.pickle - Contains list of all possible classe names
  - DIACRITICS_LIST.pickle - Contains list of all diacritics
- create_hdf_dataset.py - Create hdf files to be used in Returnn
- diacritizer.py - Run RETURNN diacritization
- dictionary.py - Includes the defined Arabic dictionary 
- diacritization_stat.py - Calculates DER and WER using the gold data and the predicted output originated from 
[Tashkeela Repo](https://github.com/AliOsm/arabic-text-diacritization/blob/master/helpers/diacritization_stat.py) system

### [configs](/configs)
- It includes config files used in [RETURNN](https://github.com/rwth-i6/returnn) for our experiments.

## Steps

To create hdf files for your training with 50% masking_factor, run:

    python3 diacritization/create_hdf_dataset.py {training_text} {hdf_source_letter_dataset} {hdf_source_diacritic_dataset} {hdf_target_dataset} --masking_factor 0.5 

To train Returnn models given a config file, invoke the following command:

    python3 returnn/rnn.py configs/config.py

To run diacrizer in decoding without any hints and with left-to-right autoregressive search, do the following:

    python3 diacritization/remove_diacritics.py -in {input_text} (Mandatory step for baseline, optional here)
    python3 diacritization/diacritizer.py configs/config_2SDiac.py --load-epoch {checkpoint_to_use} --from-file {input_txt} --to-file {outout_txt} --device cpu --twoSDiac --random_mask 1.0 --search

To evaluate the output,

    python3 diacritization/diacritization_stat.py {reference} {hypothesis} --confusion {confusion_file} --write-values-to {value_folder}


#### Note: All codes in this repository tested on [Ubuntu 18.04](http://releases.ubuntu.com/18.04)


## License
The project is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).


PUBLICATION
------------

The Arabic text diacrization model is described in:

"[Take the Hint: Improving Arabic Diacritization with Partially-Diacritized Text]()", 
Parnia Bahar, Mattia Di Gangi, Nick Rossenbach, Mohammad Zeineldeen, Interspeech 2023.

## Citation

```
@inproceedings{pbahar-et-al-2023-diacritization,
  author={Bahar, Parnia and Di Gangi, Mattia and 
  Rossenbach, Nick and Zeineldeen, Mohammad},
  title={Take the Hint: Improving Arabic Diacritization with Partially-Diacritized Text},
  booktitle={Annual Conference of the International Speech Communication Association (INTERSPEECH)},
  address={Dublin, Irland},
  month={August},
  year={2023},
}
```
