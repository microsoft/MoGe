# Evaluation

We provide a unified evaluation script that runs baselines on multiple benchmarks. It takes a baseline model and evaluation configurations, evaluates on-the-fly, and reports results instantly in a JSON file.

## Benchmarks

Donwload the processed datasets from [Huggingface Datasets](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation) and put them in the `data/eval` directory, using `huggingface-cli`:

```bash
mkdir -p data/eval
huggingface-cli download Ruicheng/monocular-geometry-evaluation --repo-type dataset --local-dir data/eval --local-dir-use-symlinks False
```

Then unzip the downloaded files:

```bash
cd data/eval  
unzip '*.zip'
# rm *.zip # if you don't keep the zip files
```

## Configuration

See [`configs/eval/all_benchmarks.json`](../configs/eval/all_benchmarks.json) for an example of evaluation configurations on all benchmarks. You can modify this file to evaluate on different benchmarks or different baselines.

## Baseline

Some examples of baselines are provided in [`baselines/`](../baselines/). Pass the path to the baseline model python code to the `--baseline` argument of the evaluation script. 

## Run Evaluation

Run the script [`moge/scripts/eval_baseline.py`](../moge/scripts/eval_baseline.py). 
For example, 

```bash
# Evaluate MoGe on the 10 benchmarks
python moge/scripts/eval_baseline.py --baseline baselines/moge.py --config configs/eval/all_benchmarks.json --output eval_output/moge.json --pretrained Ruicheng/moge-vitl --resolution_level 9

# Evaluate Depth Anything V2 on the 10 benchmarks. (NOTE: affine disparity)
python moge/scripts/eval_baseline.py --baseline baselines/da_v2.py --config configs/eval/all_benchmarks.json --output eval_output/da_v2.json
```

The `--baselies` `--input` `--output` arguments are for the inference script. The rest arguments, e.g. `--pretrained` `--resolution_level`, are custormized for loading the baseline model.

Details of the arguments:

```
Usage: eval_baseline.py [OPTIONS]

  Evaluation script.

Options:
  --baseline PATH  Path to the baseline model python code.
  --config PATH    Path to the evaluation configurations. Defaults to
                   "configs/eval/all_benchmarks.json".
  --output PATH    Path to the output json file.
  --oracle         Use oracle mode for evaluation, i.e., use the GT intrinsics
                   input.
  --dump_pred      Dump predition results.
  --dump_gt        Dump ground truth.
  --help           Show this message and exit.
```



## Wrap a Customized Baseline

Wrap any baseline method with [`moge.test.baseline.MGEBaselineInterface`](../moge/test/baseline.py).
See [`baselines/`](../baselines/) for more examples.

It is a good idea to check the correctness of the baseline implementation by running inference on a small set of images via [`moge/scripts/infer_baselines.py`](../moge/scripts/infer_baselines.py):

```base
python moge/scripts/infer_baselines.py --baseline baselines/moge.py --input example_images/ --output infer_outupt/moge --pretrained Ruicheng/moge-vitl --maps --ply
```


