# README

- Keras version: 2.2.4
- Tensorflow version: `tensorflow-gpu=1.13.1`
- Need nvidia-docker 2 and CUDA 10.0 compatible driver.

To repro bug:

```bash
$ cd <repo_root>

$ cd docker

$ make build

$ make run

# Inside docker
$ cd /keras_debug

$ python repro.py

# To show the slowdown:
$ python repro.py --infer
```
