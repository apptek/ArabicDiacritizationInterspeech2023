# Arabic Text Diacritization

This repository contains the code, configuration setups and model architectures to train, diacritize and evaluate 
Arabic Text Diacritization. The primary purpose is to facilitate the reproduction of our experiments.


USAGE INSTRUCTIONS
------------------

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
- It includes a config file used in [RETURNN](https://github.com/rwth-i6/returnn) for our experiments. See RETURNN dependencies.
- config_2SDiac.py - It needs to be completed with the paths to the files.


## Steps
To process the data, we apply [Moses tokenizer](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl), and normalize Arabic numbers and punctuation to Latin form.

To create hdf files for your training/dev sets with 50% masking_factor, run:

    python3 diacritization/create_hdf_dataset.py {training_text.txt} {source_letter.hdf} {source_diacritic.hdf} {target.hdf} --masking_factor 0.5 

To train RETURNN models given a config file, you have to first specify the paths of the generated hdf files in "train" and "dev" dict. Then, for example invoke the following command:

    python3 returnn/rnn.py configs/config_2SDiac.py

To run diacritizer in decoding without any hints and with left-to-right autoregressive search, do the following. The first command line is optional here.

    python3 diacritization/remove_diacritics.py -in {input_text} 
    export PYTHONPATH=$PYTHONPATH:{path_to_RETURNN_code}
    python3 diacritization/diacritizer.py configs/config_2SDiac.py --load-epoch {checkpoint_to_use} --from-file {input_txt} --to-file {output_txt} --device cpu --twoSDiac --masking_factor 1.0 --search

To evaluate the output,

    python3 diacritization/diacritization_stat.py {reference} {hypothesis} --confusion {confusion_file} --write-values-to {value_folder}


#### Note: All code in this repository was tested on [Ubuntu 18.04](http://releases.ubuntu.com/18.04)


## License
The project is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).


PUBLICATION
------------

The Arabic text diacrization model is described in:

"[Take the Hint: Improving Arabic Diacritization with Partially-Diacritized Text](https://arxiv.org/pdf/2306.03557.pdf)", 
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
