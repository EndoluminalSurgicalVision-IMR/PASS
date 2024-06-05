import numpy as np
from collections import deque
from scipy.spatial.distance import cdist
from models.unet import *
from numpy.linalg import norm


def LPF(img, radius = 32): 
	_, _, rows,cols = img.shape
	mrows = int(rows/2)
	mcols = int(cols/2)
	mask = torch.zeros((1, 1, rows,cols))
	mask[:, :, mrows-radius:mrows+radius,mcols-radius:mcols+radius] = 1
	return mask	


def fft_tensor(x_i, low_freq_range=32):
    # Perform FFT transformation
    freq_domain = torch.fft.fft2(x_i)

    shifted_freq_domain = torch.fft.fftshift(freq_domain) 

    # Get magnitude and phase
    magnitude = torch.abs(shifted_freq_domain)
    phase = torch.angle(shifted_freq_domain)

    # Get low-frequency component
    mask = LPF(magnitude, radius=low_freq_range).to(x_i.device)
    low_freq = shifted_freq_domain * mask
    low_freq_amp = torch.abs(low_freq)
    low_freq_pha = torch.angle(low_freq)

    return magnitude, phase, low_freq_amp, low_freq_pha


def mix_freq_prompt_tensor(amp_i, pha_i, prompt_i):
    # Use 1 for padding, size same as amp_i
   
    padded_prompt_i = torch.nn.functional.pad(prompt_i, (0, amp_i.shape[-2] - prompt_i.shape[-2],
                                                         0, amp_i.shape[-1] - prompt_i.shape[-1]), mode="constant",value=1)
    if len(padded_prompt_i.shape) == 3:
        padded_prompt_i = padded_prompt_i.unsqueeze(0).float()

    # Combine amplitude and phase to form complex tensor
    complex_spectrum = (padded_prompt_i * amp_i * torch.exp(1j * pha_i))
    
    # shift the frequency domain back
    complex_spectrum = torch.fft.ifftshift(complex_spectrum)

    # Perform inverse FFT
    x_ada_i = torch.fft.ifft2(complex_spectrum).real

    return x_ada_i



def mapping_freq2spatical(freq_i, pha_i):

    # Combine amplitude and phase to form complex tensor
    complex_spectrum = (freq_i  * torch.exp(1j * pha_i))
    
    # shift the frequency domain back
    complex_spectrum = torch.fft.ifftshift(complex_spectrum)

    # Perform inverse FFT
    x_spa_i = torch.fft.ifft2(complex_spectrum).real
    return x_spa_i


class UNet_VPTTA(nn.Module):
    def __init__(self, device, pretrained_path, patch_size=(512, 512), resnet='resnet34', num_classes=2, pretrained=False):
        super().__init__()

        freq_prompt = torch.ones((3, int(patch_size[-1]*0.01), int(patch_size[-2]*0.01)))
        self.freq_prompt = nn.Parameter(freq_prompt)
        self.device = device

        self.unet = UNet(resnet=resnet, num_classes=num_classes, pretrained=pretrained)
        pretrained_params = torch.load(pretrained_path)
        self.unet.load_state_dict(pretrained_params['model_state_dict'])
        self.bn_f = [SaveFeatures(self.unet.rn[0]),
                     SaveFeatures(self.unet.rn[4][0].conv1), SaveFeatures(self.unet.rn[4][0].conv2),
                     SaveFeatures(self.unet.rn[4][1].conv1), SaveFeatures(self.unet.rn[4][1].conv2),
                     SaveFeatures(self.unet.rn[4][2].conv1), SaveFeatures(self.unet.rn[4][2].conv2),  # 7
                     SaveFeatures(self.unet.rn[5][0].conv1), SaveFeatures(self.unet.rn[5][0].conv2),
                     SaveFeatures(self.unet.rn[5][0].downsample[0]),
                     SaveFeatures(self.unet.rn[5][1].conv1), SaveFeatures(self.unet.rn[5][1].conv2),
                     SaveFeatures(self.unet.rn[5][2].conv1), SaveFeatures(self.unet.rn[5][2].conv2),
                     SaveFeatures(self.unet.rn[5][3].conv1), SaveFeatures(self.unet.rn[5][3].conv2),  # 16
                     SaveFeatures(self.unet.rn[6][0].conv1), SaveFeatures(self.unet.rn[6][0].conv2),
                     SaveFeatures(self.unet.rn[6][0].downsample[0]),
                     SaveFeatures(self.unet.rn[6][1].conv1), SaveFeatures(self.unet.rn[6][1].conv2),
                     SaveFeatures(self.unet.rn[6][2].conv1), SaveFeatures(self.unet.rn[6][2].conv2),
                     SaveFeatures(self.unet.rn[6][3].conv1), SaveFeatures(self.unet.rn[6][3].conv2),
                     SaveFeatures(self.unet.rn[6][4].conv1), SaveFeatures(self.unet.rn[6][4].conv2),
                     SaveFeatures(self.unet.rn[6][5].conv1), SaveFeatures(self.unet.rn[6][5].conv2),  # 29
                     SaveFeatures(self.unet.rn[7][0].conv1), SaveFeatures(self.unet.rn[7][0].conv2),
                     SaveFeatures(self.unet.rn[7][0].downsample[0]),
                     SaveFeatures(self.unet.rn[7][1].conv1), SaveFeatures(self.unet.rn[7][1].conv2),
                     SaveFeatures(self.unet.rn[7][2].conv1), SaveFeatures(self.unet.rn[7][2].conv2),  # 36
                     SaveFeatures(self.unet.up1.tr_conv), SaveFeatures(self.unet.up1.x_conv),
                     SaveFeatures(self.unet.up2.tr_conv),  SaveFeatures(self.unet.up2.x_conv),
                     SaveFeatures(self.unet.up3.tr_conv),  SaveFeatures(self.unet.up3.x_conv),
                     SaveFeatures(self.unet.up4.tr_conv),  SaveFeatures(self.unet.up4.x_conv),
                     ]
        for name, param in self.unet.named_parameters():
            param.requires_grad = False

    def init_freq_prompt(self, prompt_i=None):
        if prompt_i is not None:
            self.freq_prompt.data = prompt_i
        else:
            self.freq_prompt.data.fill_(1)

        self.freq_prompt.requires_grad = True

    def forward(self, amp, pha, training=False, get_bottleneck_fea=False):
        img_ada = mix_freq_prompt_tensor(amp, pha, self.freq_prompt)
        output = self.unet(img_ada, get_bottleneck_fea=get_bottleneck_fea)
        if training:
            return output, self.bn_f, img_ada
        else:
            return output, img_ada, self.freq_prompt
        
class UNet_BN(nn.Module):
    def __init__(self,  pretrained_path, patch_size=(512, 512), resnet='resnet34', num_classes=2, pretrained=False):
        super().__init__()

        self.unet = UNet(resnet=resnet, num_classes=num_classes, pretrained=pretrained)
        pretrained_params = torch.load(pretrained_path)
        self.unet.load_state_dict(pretrained_params['model_state_dict'])
        self.bn_f = [SaveFeatures(self.unet.rn[0]),
                     SaveFeatures(self.unet.rn[4][0].conv1), SaveFeatures(self.unet.rn[4][0].conv2),
                     SaveFeatures(self.unet.rn[4][1].conv1), SaveFeatures(self.unet.rn[4][1].conv2),
                     SaveFeatures(self.unet.rn[4][2].conv1), SaveFeatures(self.unet.rn[4][2].conv2),  # 7
                     SaveFeatures(self.unet.rn[5][0].conv1), SaveFeatures(self.unet.rn[5][0].conv2),
                     SaveFeatures(self.unet.rn[5][0].downsample[0]),
                     SaveFeatures(self.unet.rn[5][1].conv1), SaveFeatures(self.unet.rn[5][1].conv2),
                     SaveFeatures(self.unet.rn[5][2].conv1), SaveFeatures(self.unet.rn[5][2].conv2),
                     SaveFeatures(self.unet.rn[5][3].conv1), SaveFeatures(self.unet.rn[5][3].conv2),  # 16
                     SaveFeatures(self.unet.rn[6][0].conv1), SaveFeatures(self.unet.rn[6][0].conv2),
                     SaveFeatures(self.unet.rn[6][0].downsample[0]),
                     SaveFeatures(self.unet.rn[6][1].conv1), SaveFeatures(self.unet.rn[6][1].conv2),
                     SaveFeatures(self.unet.rn[6][2].conv1), SaveFeatures(self.unet.rn[6][2].conv2),
                     SaveFeatures(self.unet.rn[6][3].conv1), SaveFeatures(self.unet.rn[6][3].conv2),
                     SaveFeatures(self.unet.rn[6][4].conv1), SaveFeatures(self.unet.rn[6][4].conv2),
                     SaveFeatures(self.unet.rn[6][5].conv1), SaveFeatures(self.unet.rn[6][5].conv2),  # 29
                     SaveFeatures(self.unet.rn[7][0].conv1), SaveFeatures(self.unet.rn[7][0].conv2),
                     SaveFeatures(self.unet.rn[7][0].downsample[0]),
                     SaveFeatures(self.unet.rn[7][1].conv1), SaveFeatures(self.unet.rn[7][1].conv2),
                     SaveFeatures(self.unet.rn[7][2].conv1), SaveFeatures(self.unet.rn[7][2].conv2),  # 36
                     SaveFeatures(self.unet.up1.tr_conv), SaveFeatures(self.unet.up1.x_conv),
                     SaveFeatures(self.unet.up2.tr_conv),  SaveFeatures(self.unet.up2.x_conv),
                     SaveFeatures(self.unet.up3.tr_conv),  SaveFeatures(self.unet.up3.x_conv),
                     SaveFeatures(self.unet.up4.tr_conv),  SaveFeatures(self.unet.up4.x_conv),
                     ]
        for name, param in self.unet.named_parameters():
            param.requires_grad = False

    def forward(self, img_ada, training=False, get_bottleneck_fea=False):
        output = self.unet(img_ada, get_bottleneck_fea=get_bottleneck_fea)
        if training:
            return output, self.bn_f, img_ada
        else:
            return output
        

class MemoryBank:
    def __init__(self, image_shape,K=8, S=20):
        self.S = S
        self.image_shape = image_shape
        self.memory_bank = deque(maxlen=S)
        self.K = min(S, K) 
        self.prompt_i = None
        self.init_queue()
    
    def get_size(self):
        return len(self.memory_bank)

    def init_queue(self):
        self.memory_bank.clear()

    def update_queue(self, low_freq_i, prompt_i):
        self.memory_bank.append((low_freq_i.clone(), prompt_i.clone()))

    def generate_prompt(self, low_freq_i):
        len_bank = self.get_size()
        if len_bank >= self.K:
            # Compute the cosine similarity between the low frequency components of the input image and the keys in the memory bank
            similarities = [F.cosine_similarity(low_freq_i.flatten(), pair[0].flatten(), dim=0) for pair in self.memory_bank]
           
            # Retrieve the most similar K pairs using the tensor indices directly
            top_k_indices = torch.argsort(torch.stack(similarities))[-self.K:]
            top_k_pairs = [self.memory_bank[idx] for idx in top_k_indices]
            top_k_similarities = torch.stack([similarities[idx] for idx in top_k_indices])
            top_k_similarities = top_k_similarities / torch.sum(top_k_similarities)

            # Generate the prompt by taking a weighted average of the K most similar values
            weighted_prompts = [pair[1] * similarity for pair, similarity in zip(top_k_pairs, top_k_similarities)]
            self.prompt_i = torch.sum(torch.stack(weighted_prompts), dim=0).detach()
        else:
            self.prompt_i = None
       
        return self.prompt_i
    
class Memory(object):
    def __init__(self, S=20, K=8, image_shape=(3, 512, 512)):

        self.memory = {}
        self.S = S
        self.K = K
        self.dimension = np.prod(image_shape)

    def get_size(self):
        return len(self.memory)

    def push(self, keys, prompts):

        for i, key in enumerate(keys):

            if len(self.memory.keys()) >= self.S:
                self.memory.pop(list(self.memory)[0])

            self.memory.update(
                {key.reshape(self.dimension).tobytes(): (prompts[i])})

    def _prepare_batch(self, prompts, attention_weight):

        attention_weight = np.array(attention_weight / 0.2)
        attention_weight = np.exp(attention_weight) / (np.sum(np.exp(attention_weight)))
        ensemble_prompt = prompts[0] * attention_weight[0]
        for i in range(1, len(prompts)):
            ensemble_prompt = ensemble_prompt + prompts[i] * attention_weight[i]

        return torch.FloatTensor(ensemble_prompt)

    def get_neighbours(self, keys):
        """
        Returns prompts from buffer using nearest neighbour approach
        """
        prompts = []

        dimension = keys.shape[-1] * keys.shape[-2] * keys.shape[-3]
        keys = keys.reshape(len(keys), dimension)
        total_keys = len(self.memory.keys())
        self.all_keys = np.frombuffer(
            np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(total_keys, dimension)
        for key in keys:
            similarity_scores = np.dot(self.all_keys, key.T) / (norm(self.all_keys, axis=1) * norm(key.T))
            K_neighbour_keys = self.all_keys[np.argpartition(
                similarity_scores, -self.K)[-self.K:]]
            neighbours = [self.memory[nkey.tobytes()]
                          for nkey in K_neighbour_keys]

            attention_weight = np.dot(K_neighbour_keys, key.T) / (norm(K_neighbour_keys, axis=1) * norm(key.T))
            batch = self._prepare_batch(neighbours, attention_weight)
            prompts.append(batch)

        return torch.stack(prompts)


    
