import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

image_size = 29 * 28
hidden_size = 128
num_classes = 10
num_epochs = 100
batch_size = 1
learning_rate = 0.001



train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)


def forward_forward_loss(outputs, is_positive, theta = 2):
    prob_positive = torch.sigmoid((torch.pow(outputs, 2) - theta).sum(dim=-1))
    target = torch.ones_like(prob_positive) if is_positive else torch.zeros_like(prob_positive)
    return torch.nn.functional.binary_cross_entropy(prob_positive, target)

def append_label_to_data(data, label):
    # appends a row at the top of the image with the one hot label
    reshaped = data.reshape(28, 28)
    onehot = torch.zeros(1, 28)
    onehot[0, label] = 1
    return torch.cat([onehot, reshaped], dim=0).reshape(29 * 28)

class ForwardForwardLayer(nn.Module):
    def __init__(self, input_size, output_size, activation=None):
        super(ForwardForwardLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.actfn = nn.ReLU() if activation == 'relu' else nn.Identity()
    
    def forward_train(self, data):
        images, positive = data['images'], data['positive']
        losses = []
        rets = []
        for image, pos in zip(images, positive):
            ret = self.fc(image)
            ret = self.actfn(ret)
            losses.append(forward_forward_loss(ret, pos))
            rets.append(ret)

        return {
            'loss': torch.stack(losses).sum(),
            'data': rets
        }
        
    def forward_predict(self, x):
        x = self.fc(x)
        x = self.actfn(x)
        return x

class ForwardForwardNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ForwardForwardNet, self).__init__()
        self.layer1 = ForwardForwardLayer(input_size, hidden_size, activation='relu')
        self.layer2 = ForwardForwardLayer(hidden_size, num_classes)

    def forward_train(self, data):
        positive, images = data['positive'], data['images']
        r = self.layer1.forward_train({
            'images': images,
            'positive': positive
        })
        images = r['data']
        images = list(map(lambda x: torch.nn.functional.normalize(x, dim=0), images))
        x = self.layer2.forward_train({
            'images': images,
            'positive': positive
        })
        loss, data = x['loss'], x['data']
        return {
            'losses': [r['loss'], loss],
        }
    
    def forward_predict(self, x):
        # get the outptus for every possible label
        outputs = []
        for possible_label in range(10):
            labeled = append_label_to_data(x, possible_label)
            out1 = self.layer1.forward_predict(labeled)
            out1 = torch.nn.functional.normalize(out1, dim=0)
            out2 = self.layer2.forward_predict(out1)
            outputs.append(out2.sum())
        # get the max
        outputs = torch.stack(outputs)
        return outputs.argmax(dim=0)

learning_rate = 0.001
model = ForwardForwardNet(image_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


sss = 2
for epoch in range(num_epochs):
    example_count = 0
    step_count = 0
    for image, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
        images = []
        positive = []
        # generate 10 images
        for possible_label in range(1):
            rand_label = None
            while rand_label is None or rand_label == label:
                rand_label = torch.randint(0, 10, (1,)).item()
            images.append(append_label_to_data(image, rand_label))
            positive.append(False)

        images.append(append_label_to_data(image, label)) # 1 for each possible label
        positive.append(True)
        outputs = model.forward_train({
            'images': images,
            'positive': positive
        })
        # print(outputs['losses'])
        loss = sum(outputs['losses']) / sss

        loss.backward()

        step_count += 1
        if step_count % sss == 0:
            optimizer.step()
            optimizer.zero_grad()

        example_count += 1
        if example_count >= 1000:
            # test
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.view(-1, 28 * 28)
                    outputs = model.forward_predict(images)
                    predicted = outputs
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    if total >= 1000:
                        break
            example_count = 0
            print(f'Accuracy of the network on the 100 test images: {100 * correct / total} %')