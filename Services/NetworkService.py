import os
import torch
import torch.nn as nn
import torch.optim as optim

from Network import Netrowk

MODEL_PATH = 'vocal_detector.pt'

class NetworkService:
    def __init__(self, need_train):
        self.network = Netrowk()
        self.network.cuda()
        if(not need_train and self.is_model_pretrained()):
            self.__load_model()

    def train_network(
        self,
        train_loader, 
        num_epochs = 1, 
        learning_rate = 0.0001):

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.network.parameters(), lr = learning_rate)

        total_steps = len(train_loader)
        loss_list = []

        for epoch in range(num_epochs):
            for i, (fragments, labels) in enumerate(train_loader):
                fragments = fragments.cuda()
                labels = labels.cuda()

                out = self.network(fragments)
                out = torch.reshape(out, (1, -1))
                loss = criterion(out, labels)
                loss_list.append(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.10f}, Mask summ: {:.3f}'
                          .format(epoch + 1, num_epochs, i + 1, total_steps, loss.item(), torch.sum(labels).tolist()))

        self.__save_model()

        open('loss.txt', 'w').write(';'.join(list(map(lambda x: str(x.tolist()), loss_list))))
        return loss_list


    def get_fragments_masks(self, fragments):
        masks = []
        for fragment in fragments:
            fragment = fragment.cuda()
            fragment_mask = self.network(fragment).data.cpu()
            masks.append(fragment_mask)

        return masks

    def is_model_pretrained(self):
        return os.path.exists(MODEL_PATH)

    def __save_model(self):
        torch.save(self.network.state_dict(), MODEL_PATH)

    def __load_model(self):
        self.network.load_state_dict(torch.load(MODEL_PATH))