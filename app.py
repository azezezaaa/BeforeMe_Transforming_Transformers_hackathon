from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

import gradio as gr

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from fairseq.utils import move_to_cuda


model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    "facebook/fastspeech2-en-ljspeech",
    arg_overrides={"vocoder": "hifigan", "fp16": True}
)

model1 = models[0]
model1 = model1.to(device)

TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator(models, cfg)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


def inference(image_paths):
  images = []
  
  #for image_path in image_paths:
  i_image = Image.fromarray(image_paths)
  if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

  pixel_values = feature_extractor(images=i_image, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  preds = ' '.join(str(e) for e in preds)
  #print(preds)

  sample = TTSHubInterface.get_model_input(task, preds)
  sample = move_to_cuda(sample)


  wav, rate = TTSHubInterface.get_prediction(task, model1, generator, sample)
  wav = wav.to("cpu")
  
  return wav


interface = gr.Interface(inference, gr.Image(), "audio")
interface.launch()