# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to use the multi-LoRA functionality
for offline inference.

Requires HuggingFace credentials for access to Llama2.
"""

from typing import Optional

from huggingface_hub import snapshot_download

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest
system_prompt='''Question: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?
Choices:
A. dry palms
B. wet palms
C. palms covered with oil
D. palms covered with lotion
Answer: A

Question: Which of the following statements best explains why magnets usually stick to a refrigerator door?
Choices:
A. The refrigerator door is smooth.
B. The refrigerator door contains iron.
C. The refrigerator door is a good conductor.
D. The refrigerator door has electric wires in it.
Answer: B

Question: A fold observed in layers of sedimentary rock most likely resulted from the
Choices:
A. cooling of flowing magma.
B. converging of crustal plates.
C. deposition of river sediments.
D. solution of carbonate minerals.
Answer: B

George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?Choices:
A. dry palms
B. wet palms
C. palms covered with oil
D. palms covered with lotion
Answer:'''

def create_test_prompts(
        lora_path: str
) -> list[tuple[str, SamplingParams, Optional[LoRARequest]]]:
    """Create a list of test prompts with their sampling parameters.

    2 requests for base model, 4 requests for the LoRA. We define 2
    different LoRA adapters (using the same model for demo purposes).
    Since we also set `max_loras=1`, the expectation is that the requests
    with the second LoRA adapter will be ran after all requests with the
    first adapter have finished.
    """
    return [
        (system_prompt,
         SamplingParams(temperature=0.0,
                        top_p=0.95,
                        max_tokens=1224), None),
        (
            system_prompt,
            SamplingParams(temperature=0.0,
                           top_p=0.95,
                           max_tokens=1224,),
            LoRARequest("arc-lora", 1, lora_path)),
    ]


def process_requests(engine: LLMEngine,
                     test_prompts: list[tuple[str, SamplingParams,
                                              Optional[LoRARequest]]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params, lora_request = test_prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               lora_request=lora_request)
            request_id += 1

        request_outputs: list[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)


def initialize_engine() -> LLMEngine:
    """Initialize the LLMEngine."""
    # max_loras: controls the number of LoRAs that can be used in the same
    #   batch. Larger numbers will cause higher memory usage, as each LoRA
    #   slot requires its own preallocated tensor.
    # max_lora_rank: controls the maximum supported rank of all LoRAs. Larger
    #   numbers will cause higher memory usage. If you know that all LoRAs will
    #   use the same rank, it is recommended to set this as low as possible.
    # max_cpu_loras: controls the size of the CPU LoRA cache.
    engine_args = EngineArgs(model="meta-llama/Llama-2-7b-hf",
                             enable_lora=True,
                             max_loras=1,
                             max_lora_rank=8,
                             max_cpu_loras=2,
                             max_num_seqs=256,
                             enforce_eager=True)
    return LLMEngine.from_engine_args(engine_args)


def main():
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine()
    lora_path = snapshot_download(repo_id="mogmyij/llama-arc-challenge-full-lora")
    test_prompts = create_test_prompts(lora_path)
    process_requests(engine, test_prompts)


if __name__ == '__main__':
    main()
