import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# 1. Configure 4-bit quantization using bitsandbytes
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 2. Load the model and the processor
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

# 3. Define the multimodal message payload
# You can pass a direct local path to a video file (.mp4, .avi, etc.)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/home/ubuntu/MAM_AI_Server/test.mp4",
                "fps": 0.5,  # Sample 1 frame per second to keep performance blazing fast
            },
            {
                "type": "text", 
                "text": "Provide a detailed description of the action taking place across these video frames."
            }
        ]
    }
]

# 4. Prepare inputs using qwen_vl_utils helper
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)
inputs = inputs.to("cuda")

# 5. Generate video frame descriptions
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    
    # Trim the prompt tokens out of the response output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

print("Video Analysis Output:\n", output_text[0])
