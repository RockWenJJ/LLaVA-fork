from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

model_path = "liuhaotian/llava-v1.5-7b"
prompt = "Describe the underwater image with its semantic content, including its main objects, the background, its turbidity, and the color cast. Follow the answering format like: 'This image shows <your description>. The water visibility appears to be <bad/poor/fair/good/excellent>. The color cast of the environment is <blue/green/yellow/black/red/no color cast>'"


image_file = "/mnt/03_Data/01_enhancement/01_underwater/UIE_Benchmark/Enhanced/all/tuda_wb/1.png"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "load_8bit": False,
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)