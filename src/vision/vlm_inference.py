"""
Stage 2: VLM Road Safety Feature Extraction using Qwen2-VL.

Runs on the school GPU cluster (8× RTX A5000, 192 GB VRAM total).
Uses tensor parallelism across GPUs for the 72B model, or data parallelism for 7B.

Output per segment (averaged across up to 3 images):
    pedestrian_infra    float 0-1   sidewalks, crossings, fencing present
    cyclist_infra       float 0-1   bike lanes, cycle paths present
    roadside_activity   float 0-1   markets, vendors, people density
    road_condition      float 0-1   surface quality, markings visible
    signage_quality     float 0-1   speed signs visible and clear
    vru_exposure        float 0-1   pedestrians/cyclists/PTW visible in frame
    visibility_quality  float 0-1   sightlines, bends, obstructions

These feed directly into the final fused Speed Safety Score.
"""
import json
import logging
import re
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

VLM_PROMPT = """You are a road safety expert analyzing a street-level photograph.
Score each of the following safety factors from 0.0 to 1.0 where:
  0.0 = very poor / absent / no concern
  1.0 = excellent / strongly present / serious concern

Respond ONLY with valid JSON — no prose, no markdown.

{
  "pedestrian_infra": <float>,        // sidewalks, crossings, barriers protecting pedestrians
  "cyclist_infra": <float>,           // bike lanes, cycle paths, protective infrastructure
  "roadside_activity": <float>,       // markets, vendors, dense pedestrian activity at roadside
  "road_condition": <float>,          // surface quality, lane markings visibility
  "signage_quality": <float>,         // speed signs clear and legible (1=clearly visible)
  "vru_exposure": <float>,            // pedestrians, cyclists, motorcyclists visible in frame
  "visibility_quality": <float>,      // sightlines clear, no sharp bends or obstructions
  "reasoning": "<one sentence>"
}"""

NULL_FEATURES = {
    "pedestrian_infra": 0.5,
    "cyclist_infra": 0.5,
    "roadside_activity": 0.5,
    "road_condition": 0.5,
    "signage_quality": 0.5,
    "vru_exposure": 0.5,
    "visibility_quality": 0.5,
    "reasoning": "no imagery available",
    "image_count": 0,
}

FEATURE_KEYS = [k for k in NULL_FEATURES if k != "reasoning" and k != "image_count"]


class VLMRoadAnalyzer:
    """
    Wraps Qwen2-VL for road safety feature extraction.

    For 7B: single GPU (fp16, fits in 16GB)
    For 72B: use tensor_parallel_size=8 via vLLM or split across GPUs manually

    Example (cluster):
        analyzer = VLMRoadAnalyzer(model_id="Qwen/Qwen2-VL-72B-Instruct", tensor_parallel=8)
        features = analyzer.analyze_segment(image_paths)
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "auto",
        tensor_parallel: int = 1,
        use_vllm: bool = False,
    ):
        self.model_id = model_id
        self.use_vllm = use_vllm
        self._model = None
        self._processor = None

        if use_vllm:
            self._init_vllm(model_id, tensor_parallel)
        else:
            self._init_hf(model_id, device)

    def _init_hf(self, model_id: str, device: str):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        logger.info(f"Loading {model_id} via HuggingFace transformers …")
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        self._model.eval()
        logger.info("Model loaded.")

    def _init_vllm(self, model_id: str, tensor_parallel: int):
        from vllm import LLM
        logger.info(f"Loading {model_id} via vLLM (tp={tensor_parallel}) …")
        self._vllm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel,
            max_model_len=4096,
            dtype="float16",
            trust_remote_code=True,
        )
        logger.info("vLLM engine ready.")

    def _parse_response(self, text: str) -> dict:
        """Extract JSON from model output; return NULL_FEATURES on failure."""
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                for key in FEATURE_KEYS:
                    if key in parsed:
                        parsed[key] = float(parsed[key])
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        logger.warning(f"VLM parse failed on: {text[:200]!r}")
        return {}

    def _run_hf(self, image_paths: list[Path]) -> list[dict]:
        """Run inference on a list of images, return one dict per image."""
        from PIL import Image

        results = []
        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": VLM_PROMPT},
                        ],
                    }
                ]
                text_input = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self._processor(
                    text=[text_input],
                    images=[image],
                    return_tensors="pt",
                ).to(self._model.device)

                with torch.no_grad():
                    out = self._model.generate(**inputs, max_new_tokens=256, do_sample=False)

                decoded = self._processor.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
                results.append(self._parse_response(decoded))
            except Exception as e:
                logger.warning(f"VLM inference failed for {img_path}: {e}")
                results.append({})

        return results

    def analyze_segment(self, image_paths: list[Path]) -> dict:
        """
        Analyze 1-3 images for a single road segment.
        Returns averaged feature dict, or NULL_FEATURES if no imagery.
        """
        if not image_paths:
            return dict(NULL_FEATURES)

        if self.use_vllm:
            raw_results = self._run_vllm(image_paths)
        else:
            raw_results = self._run_hf(image_paths)

        # Average valid results across images
        valid = [r for r in raw_results if r]
        if not valid:
            return dict(NULL_FEATURES)

        avg = {}
        for key in FEATURE_KEYS:
            vals = [r[key] for r in valid if key in r]
            avg[key] = round(sum(vals) / len(vals), 4) if vals else 0.5

        avg["reasoning"] = valid[0].get("reasoning", "")
        avg["image_count"] = len(valid)
        return avg

    def analyze_batch(
        self,
        segment_image_map: dict[str, list[Path]],
        batch_size: int = 16,
    ) -> dict[str, dict]:
        """
        Analyze a dict of {segment_id: [image_paths]}.
        Returns {segment_id: feature_dict}.
        Processes in batches for memory efficiency.
        """
        results = {}
        items = list(segment_image_map.items())
        total = len(items)

        for i in range(0, total, batch_size):
            batch = items[i : i + batch_size]
            for seg_id, paths in batch:
                results[seg_id] = self.analyze_segment(paths)
            logger.info(f"VLM progress: {min(i + batch_size, total)}/{total}")

        return results

    def _run_vllm(self, image_paths: list[Path]) -> list[dict]:
        """vLLM batch inference path (used for 72B on the cluster)."""
        from vllm import SamplingParams
        from PIL import Image

        images_pil = []
        valid_paths = []
        for p in image_paths:
            try:
                images_pil.append(Image.open(p).convert("RGB"))
                valid_paths.append(p)
            except Exception:
                continue

        if not images_pil:
            return []

        prompts = [
            {
                "prompt": f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{VLM_PROMPT}<|im_end|>\n<|im_start|>assistant\n",
                "multi_modal_data": {"image": img},
            }
            for img in images_pil
        ]

        outputs = self._vllm.generate(
            prompts,
            SamplingParams(max_tokens=256, temperature=0.0),
        )
        return [self._parse_response(o.outputs[0].text) for o in outputs]
