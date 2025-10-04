from django.shortcuts import render
from django.http import JsonResponse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load GPT-2 model and tokenizer (small GPT-2 for demo)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def chatbot_page(request):
    return render(request, "chatbot.html")

def chatbot_response(request):
    user_message = request.GET.get("message", "")
    if not user_message:
        return JsonResponse({"error": "No message provided"})

    # Encode input and generate response
    inputs = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean response (remove user message)
    response = response.replace(user_message, "").strip()
    
    return JsonResponse({"response": response})
