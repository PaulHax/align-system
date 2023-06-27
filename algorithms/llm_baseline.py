from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch import nn
import re


def _filtered_to_cuda(model: nn.Module) -> None:
    """Does model.cuda(), but doesn't reassign tensors that are
    already on gpu"""

    def conditional_cuda(x):
        if x.device.type != "cuda":
            return x.to("cuda:0")
        else:
            return x

    return model._apply(conditional_cuda)


PATIENT_ID_REGEXP = re.compile(r'patient_[0-9a-f]{8}\b-[0-9a-f]{4}\b-[0-9a-f]{4}\b-[0-9a-f]{4}\b-[0-9a-f]{12}', re.I)  # noqa


def select_first_mentioned_patient(generated_text):
    mentioned_patients = re.findall(PATIENT_ID_REGEXP, generated_text)

    if len(mentioned_patients) == 0:
        return None
    else:
        return mentioned_patients[0]


# May need to pass additional information (e.g. patient info)
def prepare_prompt(scenario, probe):
    scenario_description = scenario.state.unstructured
    question = probe.prompt

    return f"""
Here is a scenario related to diaster relief. We need to find the next step with right moral reasoning and give reason why : {scenario_description}

{question}"""


# May need to pass additional information (e.g. patient info)
def prepare_prompt_instruct_gpt_j(scenario, probe):
    scenario_description = scenario.state.unstructured
    question = probe.prompt

    return f"""
Here is a scenario related to diaster relief : {scenario_description}

{question}"""


class LLMBaseline:
    def __init__(self, device="cuda", model_use="gpt-j", distributed=False):
        self.device = device
        self.model_use = model_use
        self.distributed = distributed

        self.model = None
        self.tokenizer = None
        self.model_loaded = False

    def load_model(self):
        if self.model_use == 'gpt-j':
            self.model = AutoModelForCausalLM.from_pretrained(
                "EleutherAI/gpt-j-6B", torch_dtype=torch.float16)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/gpt-j-6B")
        if self.model_use == 'instruct-gpt-j':
            self.model = AutoModelForCausalLM.from_pretrained(
                "nlpcloud/instruct-gpt-j-fp16", torch_dtype=torch.float16)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "nlpcloud/instruct-gpt-j-fp16")

        if self.distributed:
            self.model = _filtered_to_cuda(self.model.half())
        else:
            self.model.to(self.device)

        self.model_loaded = True

    def run_inference(self, prompt):
        input_ids = self.tokenizer(
            prompt, return_tensors="pt").input_ids.cuda()

        len_context = input_ids.shape[-1] + 100

        gen_tokens = self.model.generate(
            input_ids,
            do_sample=True,
            temperature=0.001,
            max_length=len_context,
        )
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]

        return gen_text
