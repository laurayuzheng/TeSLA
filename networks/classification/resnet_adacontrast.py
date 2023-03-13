import logging
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, arch, num_classes, checkpoint_path=None):
        super().__init__()
        model = None

        # 1) ResNet backbone (up to penultimate layer)
        if not self.use_bottleneck:
            model = models.__dict__[arch.replace("_adacontrast", "")](pretrained=True)
            modules = list(model.children())[:-1]
            self.encoder = nn.Sequential(*modules)
            self._output_dim = model.fc.in_features
        # 2) ResNet backbone + bottlenck (last fc as bottleneck)
        else:
            model = models.__dict__[arch.replace("_adacontrast", "")](pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, 256)
            bn = nn.BatchNorm1d(256)
            self.encoder = nn.Sequential(model, bn)
            self._output_dim = 256

        self.fc = nn.Linear(self.output_dim, num_classes)

        if self.use_weight_norm:
            self.fc = nn.utils.weight_norm(self.fc, dim=0)

        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

    def forward(self, x, return_feats=False):
        # 1) encoder feature
        feat = self.encoder(x)
        feat = torch.flatten(feat, 1)

        logits = self.fc(feat)

        if return_feats:
            return feat, logits
        return logits

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = dict()
        for name, param in checkpoint["state_dict"].items():
            # get rid of 'module.' prefix brought by DDP
            name = name.replace("module.", "")
            state_dict[name] = param
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info(
            f"Loaded from {checkpoint_path}; missing params: {msg.missing_keys}"
        )

    def get_params(self):
        """
        Backbone parameters use 1x lr; extra parameters use 10x lr.
        """
        backbone_params = []
        extra_params = []
        # case 1)
        if not self.use_bottleneck:
            backbone_params.extend(self.encoder.parameters())
        # case 2)
        else:
            resnet = self.encoder[0]
            for module in list(resnet.children())[:-1]:
                backbone_params.extend(module.parameters())
            # bottleneck fc + (bn) + classifier fc
            extra_params.extend(resnet.fc.parameters())
            extra_params.extend(self.encoder[1].parameters())
            extra_params.extend(self.fc.parameters())

        # exclude frozen params
        backbone_params = [param for param in backbone_params if param.requires_grad]
        extra_params = [param for param in extra_params if param.requires_grad]

        return backbone_params, extra_params

    @property
    def num_classes(self):
        return self.fc.weight.shape[0]

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def use_bottleneck(self):
        return True

    @property
    def use_weight_norm(self):
        return True


class AdaptSupCEResNet3(nn.Module):
    def __init__(self,  ckpt_path, name='resnet50', num_classes=10):
        super(AdaptSupCEResNet3, self).__init__()
        classifier = Classifier(name, num_classes, ckpt_path)
        self.encoder =classifier.encoder
        self.fc = classifier.fc

    def forward(self, x, return_feats=False):
        feats = self.encoder(x)
        scores = self.fc(feats)

        if return_feats:
            return scores, feats

        return scores
