# Evaluation

We provide a unified evaluation script that runs baselines on multiple benchmarks. It takes a baseline model and evaluation configurations, evaluates on-the-fly, and reports results instantly in a JSON file.

## Benchmarks

Coming soon on Huggingface Datasets.

## Wrap a Baseline

Wrap any baseline method with [`moge.test.baseline.MGEBaselineInterface`](../moge/test/baseline.py).
For example, to evaluate MoGe, you can use [`baselines/moge.py`](../baselines/moge.py):

It is a good idea to check the correctness of the baseline implementation by running inference on a small set of images via [`moge/scripts/infer_baselines.py`](../moge/scripts/infer_baselines.py):

```base
python moge/scripts/infer_baselines.py --baseline baselines/moge.py --input example_images/ --output infer_outupt/moge --pretrained Ruicheng/moge-vitl --maps --ply
```
The `--baselies` `--input` `--output` arguments are for the inference script. The rest arguments are custormized for loading the baseline model.

See [`baselines/`](../baselines/) for more examples.

## Run

Use the script [`moge/scripts/eval_baseline.py`](../moge/scripts/eval_baseline.py). 

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

```bash
python moge/scripts/eval_baseline.py --baseline baselines/moge.py --config configs/eval/all_benchmarks.json --output eval_output/moge.json --pretrained Ruicheng/moge-vitl
```