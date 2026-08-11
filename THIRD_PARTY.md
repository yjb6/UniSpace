# Third-party evaluation components

The repository-level MIT license does not relicense third-party benchmarks,
datasets, models, or services. Obtain external assets from their official
sources and follow their terms.

| Component | How this repository uses it | Upstream terms |
|---|---|---|
| GenEval | Adapted prompt/evaluation utilities under `unispace/eval/gen/geneval/` | MIT notices are retained in the source files. |
| DPG-Bench | Evaluation package under `unispace/eval/gen/dpg/` | See the bundled `unispace/eval/gen/dpg/LICENSE`. |
| VIEScore | Adapted evaluator utilities under `unispace/eval/gen/gedit/viescore/` | MIT; the upstream license is bundled in that directory. |
| GEdit-Bench | Downloaded externally at evaluation time; the dataset is not redistributed here. | The official `stepfun-ai/GEdit-Bench` dataset card declares MIT. |
| OneIG-Bench | `score.sh` invokes a separately obtained official checkout; official prompts, models, and scorer code are not redistributed here. | The official dataset card declares CC BY-NC 4.0. The official GitHub scorer repository did not provide a root license when checked for this preview, so users must review its current terms before use or redistribution. |
| ImgEdit | The repository contains adapted benchmark orchestration and score aggregation; benchmark data is external. | The official `PKU-YuanGroup/ImgEdit` repository did not provide a root license when checked for this preview. Obtain the benchmark separately and review its current data, model, and judge terms before use or redistribution. |

Model dependencies such as Qwen3, SigLIP2, DINOv2, and evaluator checkpoints
are not redistributed by this source tree and remain subject to their own
licenses. API-based judging also remains subject to the selected provider's
terms.
