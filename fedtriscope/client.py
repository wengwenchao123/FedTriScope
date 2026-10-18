"""City-local model and server-owned shared backbone."""
import copy
import torch

class Server:
    def __init__(self, shared_model):
        self.model = copy.deepcopy(shared_model)

    def get_weights(self):
        return {n: p.detach().clone() for n, p in self.model.state_dict().items()}

class Client:
    def __init__(self, model, data, args):
        self.model = model.to(args.device)
        self.data, self.device = data, args.device
        self.criterion = torch.nn.L1Loss()
        self.loss_settings = {'loss_func': 'mae', 'units': 'original observation scale'}

    def set_weights(self, weights, interpolation='overwrite'):
        if interpolation != 'overwrite':
            raise ValueError('FedTriScope uses full shared-backbone replacement')
        self.model.shared_model.load_state_dict(weights, strict=True)

    def predict(self, x, start):
        return self.model(x.to(self.device), start.to(self.device))*self.data.std+self.data.mean
