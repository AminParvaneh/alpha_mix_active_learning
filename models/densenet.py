import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.utils import build_mlp


class DenseNetClassifier(nn.Module):
    def __init__(self, arch_name='resnet50', n_label=10, pretrained=True, dropout=0.,
                 fine_tune_layers=1, emb_size=256, in_channels=1):
        super(DenseNetClassifier, self).__init__()

        self.n_label = n_label

        model = getattr(models, arch_name)
        densenet = model(pretrained=pretrained)

        # Remove linear layers
        modules = list(densenet.features.children())
        if modules[0].in_channels != in_channels:
            conv = modules[0]
            modules[0] = nn.Conv2d(in_channels=in_channels, out_channels=conv.out_channels,
                                   kernel_size=conv.kernel_size, stride=conv.stride,
                                   padding=conv.padding, bias=conv.bias)
            pretrained = False

        self.features = nn.Sequential(*modules)

        if pretrained:
            self.fine_tune(fine_tune_layers)

        input_size = densenet.classifier.in_features
        if emb_size <= 0 or emb_size == input_size:
            self.embedding_size = input_size
            self.hidden_layers = None
        else:
            self.embedding_size = emb_size
            self.hidden_layers = build_mlp(input_size, (), emb_size, dropout=dropout, use_batchnorm=False,
                                     add_dropout_after=False)

        self.classifier = build_mlp(self.embedding_size, (), n_label,
                                    dropout=dropout,
                                    use_batchnorm=False,
                                    add_dropout_after=False)

    def forward(self, x, embedding=False):
        if embedding:
            embd = x
        else:
            embd = self.features(x)
            embd = F.relu(embd, inplace=True)
            embd = F.adaptive_avg_pool2d(embd, (1, 1))
            embd = torch.flatten(embd, 1)

            if self.hidden_layers:
                embd = self.hidden_layers(embd)

        out = self.classifier(embd)

        return out, embd

    def get_embedding_dim(self):
        return self.embedding_size

    def fine_tune(self, fine_tune_layers):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune_layers: How many convolutional layers to be fine-tuned (negative value means all)
        """
        for p in self.features.parameters():
            p.requires_grad = False

        # Last convolution layers to be fine-tuned
        for c in list(self.features.children())[
                 0 if fine_tune_layers < 0 else len(list(self.features.children())) - (1 + fine_tune_layers):]:
            for p in c.parameters():
                p.requires_grad = True

    def get_classifier(self):
        return self.classifier[-1]
