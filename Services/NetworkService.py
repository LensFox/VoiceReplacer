import os
import torch
import torch.nn as nn
import torch.optim as optim
from time import time

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
        global_epoch_index,
        global_epoches_count,
        local_epoch_index,
        local_epoches_count,
        block_index,
        blocks_count,
        learning_rate = 0.0001):
        print('training started')

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.network.parameters(), lr = learning_rate)

        start_time = time()

        total_steps = len(train_loader)
        loss_list = []

        epoch_index = local_epoch_index * local_epoches_count * global_epoch_index
        epoches_count = local_epoches_count * global_epoches_count

        for step_index, (fragments, labels) in enumerate(train_loader):
            fragments = fragments.cuda()
            labels = labels.cuda()

            out = self.network(fragments)
            loss = criterion(out, labels)
            loss_list.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            if (step_index + 1) % 10 == 0:
                print('Epoch [{}/{}], Block [{}/{}], Step [{}/{}], Loss: {:.10f}, Mask summ: {:.3f}'
                        .format(
                           epoch_index + 1, epoches_count,
                           block_index + 1, blocks_count, 
                           step_index + 1, total_steps, 
                           loss.item(), 
                           torch.sum(labels).tolist()))

        self.__save_model()

        end_time = time()

        print('training ended for {} s'.format(end_time - start_time))

        open('loss.txt', 'w').write(';'.join(list(map(lambda x: str(x.tolist()), loss_list))))
        return loss_list


    def get_fragments_masks(self, fragments):
        sigmoid = nn.Sigmoid()

        masks = []
        clear_values = []
        for fragment in fragments:
            fragment = fragment.cuda()

            fragment_mask = self.network(fragment).data.cpu()
            fragment_mask = sigmoid(fragment_mask)

            masks.append(fragment_mask)

            clear_mask = fragment_mask.tolist()
            clear_values.extend(clear_mask)

        return masks

    def is_model_pretrained(self):
        return os.path.exists(MODEL_PATH)

    def __save_model(self):
        torch.save(self.network.state_dict(), MODEL_PATH)

    def __load_model(self):
        self.network.load_state_dict(torch.load(MODEL_PATH))