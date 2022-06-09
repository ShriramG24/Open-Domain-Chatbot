import random
import json
import torch
from model import FeedForwardModel
from preprocess import tokenize, stem, bag_of_words

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('dialogues.json', 'r') as file:
    dialogues = json.load(file)

FILE = 'model.pth'
bot = torch.load(FILE)
input_size, output_size, hidden_size = bot['input_size'], bot['output_size'], bot['hidden_size']
model_state, word_bank, intents = bot['model_state'], bot['word_bank'], bot['intents']
model = FeedForwardModel(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Nick"
print("Let's chat! Type 'quit' to exit.")
while True:
    sentence = input('You: ')
    if sentence == 'quit':
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, word_bank)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    intent = intents[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for dialogue in dialogues:
            if dialogue["intent"] == intent:
                print(f'{bot_name}: {random.choice(dialogue["responses"])}')
    else:
        print(f'{bot_name}: I do not understand...')


