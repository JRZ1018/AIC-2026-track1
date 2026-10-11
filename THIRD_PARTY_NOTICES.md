# Third-party notices

`src/aictrack1/models/clip_resolution.py` mirrors the visual forward topology
of [OpenAI CLIP](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/model.py),
commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`, distributed under the
[MIT License](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/LICENSE).
The local adapter retains the original modules and pretrained parameter values,
adding only deterministic interpolation of the visual spatial position grid for
the preregistered 336-pixel experiment. The 224 path calls the original forward.
