import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import FeedForwardModel
from preprocess import tokenize, stem, bag_of_words

with open('dialogues.json', 'r') as file:
    dataset = json.load(file)

word_bank = []
intents =[]
x_y = []

punc = [',', '!', '.', '?', '(', ')', '\'', '\"']

for dialogue in dataset:
    intents.append(dialogue["intent"])
    for input in dialogue["text"]:
        words = tokenize(input)
        word_bank.extend(words)
        x_y.append((words, dialogue["intent"]))

word_bank = sorted(set([stem(word) for word in word_bank if word not in punc]))
intents = sorted(set(intents))

x_train, y_train = [], []

for (words, intent) in x_y:
    x_train.append(bag_of_words(words, word_bank))
    y_train.append(intents.index(intent))

x_train, y_train = np.array(x_train), np.array(y_train)

class ChatbotData(Dataset):
    def __init__(self):
        self.num_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.num_samples
    
batch_size = 8
hidden_size = 8
output_size = len(intents)
input_size = len(word_bank)
learning_rate = 0.001
num_epochs = 1000

dataset = ChatbotData()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FeedForwardModel(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def main():
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device).long()

            # Forward Pass
            outputs = model(words)
            loss = criterion(outputs, labels)

            # Gradient Descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss = {loss.item():.4f}...')

    print(f'Final Loss = {loss.item():.4f}')

    to_save = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "word_bank": word_bank,
        "intents": intents,
    }

    FILE = "model.pth"
    torch.save(to_save, FILE)

    print(f'Training complete. Model saved to {FILE}.')

if __name__ == '__main__':
    main()


