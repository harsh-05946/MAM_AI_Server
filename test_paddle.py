from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# 1. Load Model (3B is the perfect size for your needs)
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

# 2. Prepare the Image
image_path = "test6.jpg" # This could be your crowded photo or a black image

# 3. Use a "Strict" Prompt to avoid jargon and hallucinations
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            #{"type": "text", "text": "give me the general emotion of all the people in the image , Provide a single word for the emotion"}
            {"type": "text", "text": "give me the text in the image, without any other text and explanation and do not translate the text"}
            #{"type": "text", "text": "give me the scene description of the image, provide a brief and strict description of the image with max 1-3 lines"}
        ],
    }
]

# 4. Process and Generate
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=512)
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text[0].split("assistant\n")[-1])