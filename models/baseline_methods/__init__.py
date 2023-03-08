import models.baseline_methods.models.segmentation as segmentation
import importlib
import torch


def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    modellib = importlib.import_module('models.baseline_methods.models.segmentation')

    model = None
    for name, cls in modellib.__dict__.items():
        if name == model_name and issubclass(cls, torch.nn.Module):
            model = cls
            break

    if model is None:
        print("There should be a subclass of BaseModel with class name that matches %s in lowercase." % (
            model_name))
        exit(0)

    return model
