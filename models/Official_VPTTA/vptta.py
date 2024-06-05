import torch
import numpy as np
from numpy.linalg import norm
import torch
import torch.nn as nn
import torch.nn.functional as F


class Memory(object):
    """
        Create the empty memory buffer
    """

    def __init__(self, size, dimension=1 * 3 * 512 * 512):
        self.memory = {}
        self.size = size
        self.dimension = dimension

    def reset(self):
        self.memory = {}

    def get_size(self):
        return len(self.memory)

    def push(self, keys, logits):
        for i, key in enumerate(keys):
            if len(self.memory.keys()) > self.size:
                self.memory.pop(list(self.memory)[0])

            self.memory.update(
                {key.reshape(self.dimension).tobytes(): (logits[i])})

    def _prepare_batch(self, sample, attention_weight):
        attention_weight = np.array(attention_weight / 0.2)
        attention_weight = np.exp(attention_weight) / (np.sum(np.exp(attention_weight)))
        ensemble_prediction = sample[0] * attention_weight[0]
        for i in range(1, len(sample)):
            ensemble_prediction = ensemble_prediction + sample[i] * attention_weight[i]

        return torch.FloatTensor(ensemble_prediction)

    def get_neighbours(self, keys, k):
        """
        Returns samples from buffer using nearest neighbour approach
        """
        samples = []

        keys = keys.reshape(len(keys), self.dimension)
        total_keys = len(self.memory.keys())
        self.all_keys = np.frombuffer(
            np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(total_keys, self.dimension)

        for key in keys:
            similarity_scores = np.dot(self.all_keys, key.T) / (norm(self.all_keys, axis=1) * norm(key.T))

            K_neighbour_keys = self.all_keys[np.argpartition(similarity_scores, -k)[-k:]]
            neighbours = [self.memory[nkey.tobytes()] for nkey in K_neighbour_keys]

            attention_weight = np.dot(K_neighbour_keys, key.T) / (norm(K_neighbour_keys, axis=1) * norm(key.T))
            batch = self._prepare_batch(neighbours, attention_weight)
            samples.append(batch)

        return torch.stack(samples), np.mean(similarity_scores)



class Prompt(nn.Module):
    def __init__(self, prompt_alpha=0.01, image_size=512):
        super().__init__()
        self.prompt_size = int(image_size * prompt_alpha) if int(image_size * prompt_alpha) > 1 else 1
        self.padding_size = (image_size - self.prompt_size)//2
        self.init_para = torch.ones((1, 3, self.prompt_size, self.prompt_size))
        self.data_prompt = nn.Parameter(self.init_para, requires_grad=True)
        self.pre_prompt = self.data_prompt.detach().cpu().data

    def update(self, init_data):
        with torch.no_grad():
            self.data_prompt.copy_(init_data)

    def iFFT(self, amp_src_, pha_src, imgH, imgW):
        # recompose fft
        real = torch.cos(pha_src) * amp_src_
        imag = torch.sin(pha_src) * amp_src_
        fft_src_ = torch.complex(real=real, imag=imag)

        src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1), s=[imgH, imgW]).real
        return src_in_trg

    def forward(self, x):
        _, _, imgH, imgW = x.size()

        fft = torch.fft.fft2(x.clone(), dim=(-2, -1))

        # extract amplitude and phase of both ffts
        amp_src, pha_src = torch.abs(fft), torch.angle(fft)
        amp_src = torch.fft.fftshift(amp_src)

        # obtain the low frequency amplitude part
        prompt = F.pad(self.data_prompt, [self.padding_size, imgH - self.padding_size - self.prompt_size,
                                          self.padding_size, imgW - self.padding_size - self.prompt_size],
                       mode='constant', value=1.0).contiguous()

        amp_src_ = amp_src * prompt
        amp_src_ = torch.fft.ifftshift(amp_src_)

        amp_low_ = amp_src[:, :, self.padding_size:self.padding_size+self.prompt_size, self.padding_size:self.padding_size+self.prompt_size]

        src_in_trg = self.iFFT(amp_src_, pha_src, imgH, imgW)
        return src_in_trg, amp_low_