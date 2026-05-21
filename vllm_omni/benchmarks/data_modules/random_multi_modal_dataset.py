import base64
from dataclasses import dataclass
import io
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import soundfile as sf
import torch
from vllm.benchmarks.datasets import RandomMultiModalDataset, SampleRequest, process_image, process_video
import pandas as pd
from vllm.tokenizers import TokenizerLike
logger = logging.getLogger(__name__)


def process_audio(audio: Any) -> Mapping[str, Any]:
    """
    Process a single audio input and return a multimedia content dictionary.

    Supports the following input types:

    1. Dictionary with raw audio bytes: - Expects a dict with a 'bytes' key
       containing raw audio data.

    2. String input: - Treats the string as a URL or local file path.  -
       Prepends "file://" if the string doesn't start with "http://" or
       "file://".  - Returns a dictionary with the audio URL.

    Raises:
        ValueError: If the input is not a supported type.
    """
    if isinstance(audio, dict) and "bytes" in audio:
        audio_bytes = audio["bytes"]
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return {
            "type": "audio_url",
            "audio_url": {"url": f"data:audio/mpeg;base64,{audio_base64}"},
        }
    if isinstance(audio, str):
        audio_url = audio if audio.startswith(("http://", "https://", "file://")) else f"file://{audio}"
        return {"type": "audio_url", "audio_url": {"url": audio_url}}

    raise ValueError(
        f"Invalid audio input {audio}. Must be a string of local path/remote url, "
        f"or a dictionary with raw audio bytes in the form of `{{'bytes': raw_audio_bytes}}`."
    )


# -----------------------------------------------------------------------------
# MultiModalDataset Implementation
# -----------------------------------------------------------------------------
class OmniRandomMultiModalDataset(RandomMultiModalDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_synthetic_audio(
        self,
        duration: int,  # seconds
        num_channels: int,  # 1:Mono，2:Stereo 5:5.1 surround sound
    ) -> dict[str, Any]:
        """Generate synthetic audio with random values.
        Default use 48000Hz.
        """
        sample_rate = 48000
        num_samples = int(sample_rate * duration)
        audio_data = self._rng.uniform(-0.5, 0.5, (num_samples, num_channels))
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_tensor = torch.FloatTensor(audio_data.T)
        audio_np = audio_tensor.numpy()

        buffer = io.BytesIO()

        sf.write(buffer, audio_np.T, sample_rate, format="wav")

        buffer.seek(0)
        audio_bytes = buffer.read()
        buffer.close()
        return {
            "bytes": audio_bytes,
        }

    def generate_mm_item(
        self,
        mm_item_config: tuple[int, int, int],
    ) -> Mapping[str, Any]:
        """
        Create synthetic images and videos and
        apply process_image/process_video respectively.
        This follows the OpenAI API chat completions
        https://github.com/openai/openai-python
        """

        if self.map_config_to_modality(mm_item_config) == "image":
            return process_image(self.generate_synthetic_image(mm_item_config[1], mm_item_config[0]))
        elif self.map_config_to_modality(mm_item_config) == "video":
            return process_video(self.generate_synthetic_video(mm_item_config[1], mm_item_config[0], mm_item_config[2]))
        elif self.map_config_to_modality(mm_item_config) == "audio":
            return process_audio(self.generate_synthetic_audio(mm_item_config[1], mm_item_config[2]))
        else:
            raise ValueError(f"Invalid multimodal item configuration: {mm_item_config}")

    def generate_synthetic_video(self, width: int, height: int, num_frames: int) -> Any:
        """Generate synthetic video with random values."""
        import imageio

        video_data = self._rng.integers(
            0,
            256,
            (num_frames, height, width, 3),
            dtype=np.uint8,
        )
        buffer = io.BytesIO()
        writer_kwargs = {
            "format": "mp4",
            "fps": 30,
            "codec": "libx264",
            "quality": 7,
            "pixelformat": "yuv420p",
            "macro_block_size": 16,
            "ffmpeg_params": [
                "-preset",
                "medium",
                "-crf",
                "23",
                "-movflags",
                "+faststart",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                f"scale={width}:{height}",
            ],
        }

        with imageio.get_writer(buffer, **writer_kwargs) as writer:
            for frame_idx in range(num_frames):
                writer.append_data(video_data[frame_idx])
        buffer.seek(0)
        video_bytes = buffer.read()

        return {
            "bytes": video_bytes,
        }

    def map_config_to_modality(self, config: tuple[int, int, int]) -> str:
        """Map the configuration to the modality."""
        if config[0] == 0:
            return "audio"
        elif config[-1] == 1:
            return "image"
        elif config[-1] > 1:
            return "video"
        else:
            raise ValueError(f"Invalid multimodal item configuration: {config}")

@dataclass
class ServeGenSampleRequest(SampleRequest):
    time_stamp:float=0.0
class ServeGenDataSet(OmniRandomMultiModalDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.disable_shuffle = True
        self.load_data()
    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        # self.data will be a list of dictionaries
        # e.g., [{"prompt": "What is the capital of India?"}, ...]
        # This will be the standardized format which load_data()
        # has to convert into depending on the filetype of dataset_path.
        # sample() will assume this standardized format of self.data
        self.data = []

        # Load the JSONL file
        if self.dataset_path.endswith(".jsonl"):
            jsonl_data = pd.read_json(path_or_buf=self.dataset_path, lines=True)

            # check if the JSONL file has a 'prompt' column
            # if "prompt" not in jsonl_data.columns:
            #     raise ValueError("JSONL file must contain a 'prompt' column.")

            # Convert each row to a dictionary and append to self.data
            # This will convert the DataFrame to a list of dictionaries
            # where each dictionary corresponds to a row in the DataFrame.
            # This is the standardized format we want for self.data
            for _, row in jsonl_data.iterrows():
                self.data.append(row.to_dict())
        else:
            raise NotImplementedError(
                "Only JSONL format is supported for CustomDataset."
            )
    def sample(
        self,
        tokenizer: TokenizerLike,
        num_requests: int|None = None,
        request_id_prefix: str = "",
        prefix_len: int = 0,
        **kwargs,
    ) -> list[ServeGenSampleRequest]:
        self.num_available_samples = len(self.data)
        if num_requests is None:
            num_requests=len(self.data)
        if num_requests <= 0:
            num_requests = self.num_available_samples
            logger.info(
                "num_requests is set to 0 or negative, "
                "so using all available samples: %d",
                num_requests,
            )
        logger.info("Need Sampling %d requests from the dataset", num_requests)
        sample_requests:list[ServeGenSampleRequest] = []
        prohibited_tokens = list(
            tok_id
            for tok_id, token in tokenizer.added_tokens_decoder.items()
            if token.special
        )
        vocab_size = tokenizer.vocab_size
        all_tokens = np.arange(vocab_size)
        allowed_tokens = np.array(list(set(all_tokens) - set(prohibited_tokens)))
        prefix_token_ids=self.get_prefix(tokenizer,allowed_tokens,prefix_len)
        token_mismatch_total=0
        for i,item in enumerate(self.data[:num_requests]):
            prompt,total_input_len,token_mismatch=self.generate_token_sequence(
                tokenizer=tokenizer,
                prefix_token_ids=prefix_token_ids,
                prefix_len=prefix_len,
                vocab_size=vocab_size,
                input_len=self.data[i]["text_tokens"],
                offset=0,
                index=i,
                allowed_tokens=allowed_tokens,
            )
            token_mismatch_total+=token_mismatch
            mm_item_list=[]
            for mm in self.data[i]["mm_items"]:
                if mm["modality"] == "image":
                    mm_item_list.append(self.generate_mm_item((mm["h"],mm["w"],1)))
                elif mm["modality"] == "video":
                    mm_item_list.append(self.generate_mm_item((mm["h"],mm["w"],int(mm["t"]*mm["fps"]))))
                elif mm["modality"] == "audio":
                    mm_item_list.append(self.generate_mm_item((0,mm["duration_s"],mm["num_channels"])))
            sample_request=ServeGenSampleRequest(
                prompt=prompt,
                prompt_len=total_input_len,
                expected_output_len=self.data[i]["output_tokens"],
                multi_modal_data=mm_item_list,
                request_id=f"{request_id_prefix}{i}",
                time_stamp=self.data[i]['timestamp']
            )
            sample_requests.append(sample_request)
        if token_mismatch_total != 0:
            sign = "more" if token_mismatch_total > 0 else "fewer"
            logger.warning(
                "Across all generated prompts, there were %d %s tokens "
                "than expected after decoding and re-encoding. This is "
                "expected due to the imperfect nature of the sampling "
                "procedure.",
                abs(token_mismatch_total),
                sign,
            )
        return sample_requests 