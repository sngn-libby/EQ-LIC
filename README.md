# EQAT
unofficial Pytorch Implementation for "Integer quantized learned image compression", ICIP 2023 (submitted)

## Acknowledgement
The framework is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI), part of the codes benefit from [InvCompress](https://github.com/xyq7/InvCompress). We add our model in compressai.models.actfunctions (not in compressai.models.ours, there are InvCompress models rather). We also modify compressai.zoo, compressai.layers to use the models. 

We add our quantization methods in 'quan' directory, and we add quantization configuration in 'config' directory. We modify examples/train.py and examples/train_qel.py for quantization and using SQE loss.

You can check our proposed quantizer at 'quan/quantizer/lsq_qel.py'.

## Installation
As mentioned in [CompressAI](https://github.com/InterDigitalInc/CompressAI), "A C++17 compiler, a recent version of pip (19.0+), and common python packages are also required (see setup.py for the full list)."
```bash
git clone https://github.com/Geunwoo-Jeon/eqat.git
cd eqat/codes/
conda create -n eqat python=3.7 
conda activate eqat
pip install -U pip && pip install -e .
conda install -c conda-forge tensorboard
```

### Evaluation

Some evaluation dataset can be downloaded from 
[kodak dataset](http://r0k.us/graphics/kodak/), [CLIC](http://challenge.compression.cc/tasks/)

```bash
python -m compressai.utils.eval_model checkpoint $eval_data_dir -a invcompress -exp $exp_name -s $save_dir
```

An example: to evaluate model of quality 1 optimized with mse on kodak dataset. 
```bash
python -m compressai.utils.eval_model checkpoint ../data/kodak -a invcompress -exp exp_01_mse_q1 -s ../results/exp_01
```

If you want to evaluate your trained model on own data, please run update before evaluation. An example:
```bash
python -m compressai.utils.update_model -exp $exp_name -a invcompress
python -m compressai.utils.eval_model checkpoint $eval_data_dir -a invcompress -exp $exp_name -s $save_dir
```

### Train
We use the training dataset processed in the [repo](https://github.com/liujiaheng/CompressionData). We further preprocess with /codes/scripts/flicker_process.py

There are some default setted arguments. Please check them in the train.py file of train_qel.py file.
The default setting is quantizing the model. If you don't want to, add `--lsq` argument.

An example: to train model of LSQ quantized MS-ReLU model of quality 5, with a pretrained full-precision base model.
```bash
cd ~/compressai/codes
python -u examples/train.py -exp MS_LSQ_8bit_q5 -m ms-relu -q 5 --lambda 0.0250 -b ../experiments/MS_q5/checkpoints/checkpoint_best_loss.pth.tar --config configs/ms_8bit_lsq.yaml -d ~/flicker
```

Another example: to train IQ-LIC model (proposed model) of quality 6, with a pretrained full-precision base model.

```bash
cd ~/compressai/codes
python -u examples/train_qel.py -exp MS_proposed_q6 -m ms-relu -q 6 --lambda 0.0483 -b ../experiments/MS_q6/checkpoints/checkpoint_best_loss.pth.tar --config configs/ms_lsqqel.yaml -d ~/flicker 
```
