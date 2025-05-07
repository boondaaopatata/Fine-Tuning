from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("facebook/opt-350m")
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
model.config.pad_token_id = model.config.eos_token_id

user_input = input("Enter your prompt: ")
inputs = tok(user_input, return_tensors="pt")
out = model.generate(**inputs, max_length=188, do_sample=True, top_p=0.9)
print(tok.decode(out[0], skip_special_tokens=True))
