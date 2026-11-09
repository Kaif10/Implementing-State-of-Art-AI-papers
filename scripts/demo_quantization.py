"""Convert CPU demo linear layers one at a time to bound temporary memory use."""
import torch


def quantize_cpu_linears(model):
    def visit(parent):
        for name in list(parent._modules):
            child = parent._modules[name]
            if isinstance(child, torch.nn.Linear):
                container = torch.nn.Sequential(child.float())
                converted = torch.ao.quantization.quantize_dynamic(
                    container, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
                setattr(parent, name, converted[0])
            elif child is not None:
                visit(child)
    visit(model)
    # Embeddings and normalization remain floating point; quantized linear
    # kernels consume and return float32 activations.
    return model.float().eval()
