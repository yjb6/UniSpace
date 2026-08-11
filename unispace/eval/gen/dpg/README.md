# DPG-Bench adapter

This directory retains the DPG-Bench prompts and scorer used by the UniSpace
evaluation adapter. DPG-Bench originates from the official ELLA release; see
the included `LICENSE` and the [ELLA project](https://ella-diffusion.github.io/).

Generate images through `../gen_images_dpg.py`. The scorer requires the official
mPLUG visual-question-answering checkpoint and its ModelScope dependencies.
Configure the checkpoint root explicitly with `CKPT_ROOT` before invoking
`dpg_bench/dist_eval.sh`.

The upstream ELLA model demo, notebook, and decorative assets are not included
because they are unrelated to evaluating UniSpace.
