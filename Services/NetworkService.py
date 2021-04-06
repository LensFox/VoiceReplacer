import os
import torch
import torch.nn as nn
import torch.optim as optim
from time import time
import random

from Network import Netrowk

MODEL_PATH = 'vocal_detector.pt'

FILES_PER_BLOCK = 5
EPOCHES_COUNT = 2
EPOCHES_PER_BLOCK = 1
GLOBAL_EPOCHES_COUNT = EPOCHES_COUNT // EPOCHES_PER_BLOCK

class NetworkService:
    def __init__(self, need_train):
        self.network = Netrowk()
        self.network.cuda()
        if(not need_train and self.is_model_pretrained()):
            self.__load_model()

    def train_network(
        self, 
        dataset_preparer_service,
        feature_file_names,
        mask_file_names,
        learning_rate = 0.0001):
        print('training started')

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.network.parameters(), lr = learning_rate)

        file_names = [(feature_file_names[i], mask_file_names[i]) for i in range(len(feature_file_names))]
        start_time = time()
        
        file_block_count = len(file_names) // FILES_PER_BLOCK + 1
        global_loss = []
        # train N-times in different files with downloading features
        for global_epoch_index in range(GLOBAL_EPOCHES_COUNT):
            random.shuffle(file_names)
            
            # train each block
            for block_index in range(file_block_count):
                file_block = file_names[block_index * FILES_PER_BLOCK: (block_index + 1) * FILES_PER_BLOCK]

                data_loader = dataset_preparer_service.prepare_data_to_train(file_block)

                # train each block M-times to prevent many readings from disk
                for epoch_number in range(EPOCHES_PER_BLOCK):
                    random.shuffle(data_loader)

                    loss = self.__one_train_block(
                        data_loader, 
                        global_epoch_index, 
                        GLOBAL_EPOCHES_COUNT,
                        epoch_number,
                        EPOCHES_PER_BLOCK,
                        block_index,
                        file_block_count,
                        criterion,
                        optimizer)
                    global_loss += loss
    
        open('loss.txt', 'w').write(';'.join(list(map(lambda x: str(x.tolist()), global_loss))))
        
        self.__save_model()

        end_time = time()

        print('training ended for {} s'.format(end_time - start_time))

    def __one_train_block(
        self,
        train_loader, 
        global_epoch_index,
        global_epoches_count,
        local_epoch_index,
        local_epoches_count,
        block_index,
        blocks_count,
        criterion,
        optimizer):
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