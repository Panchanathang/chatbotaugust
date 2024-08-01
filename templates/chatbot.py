from flask import Flask, request, jsonify, render_template
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import os

app = Flask(__name__)

# Load the Blenderbot model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    inputs = tokenizer(user_message, return_tensors='pt')
    outputs = model.generate(**inputs)
    response_message = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'response': {'message': response_message}})

if __name__ == '__main__':
    app.run(debug=True)
