import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import yaml
import random
import torchmetrics
from trainers.evaluate import SaveFullModelEveryNEpochs

torch.set_printoptions(sci_mode=False, precision=4)


def load_config(config_path):
    """
    YAML 파일 로드 함수
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_model(model_path, device):
    """
    model.pt (전체 모델) 로드 함수
    """
    model = torch.load(model_path, map_location=device)
    try:

        for module in model.modules():
            if isinstance(module, torchmetrics.Metric) and not hasattr(
                module, "_dtype_convert"
            ):
                module._dtype_convert = True
    except ImportError:
        pass


    model.to(device)
    model.eval()

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)

    if missing_keys:
        print(f"⚠️ Warning: Missing Keys ({len(missing_keys)}): {missing_keys}")

    else:
        print("✅ No missing keys found.")

    if unexpected_keys:
        print(f"⚠️ Warning: Unexpected Keys ({len(unexpected_keys)}): {unexpected_keys}")

    else:
        print("✅ No unexpected keys found.")

    return model

def preprocess_data(image_path, tabular_data, config, device):
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    categorical_mapping = config["categorical_mapping"]

    def get_categorical_key(column, value):
        mapping = categorical_mapping[column]
        for key, val in mapping.items():
            if val == value:
                return key

        return 0

    encoded_categorical = []
    for col in config["categorical_columns"]:
        if col in tabular_data:
            encoded_categorical.append(get_categorical_key(col, tabular_data[col]))

        else:
            encoded_categorical.append(0)

    encoded_categorical = (
        torch.tensor(encoded_categorical, dtype=torch.float32).unsqueeze(0).to(device)
    )


    continuous_features = []
    for col in config["table_mean"].keys():
        if col in tabular_data:
            val = tabular_data[col]

        else:
            val = config["table_mean"][col]

        continuous_features.append(
            (val - config["table_mean"][col]) / config["table_std"].get(col, 1)
        )

    continuous_features = (
        torch.tensor(continuous_features, dtype=torch.float32).unsqueeze(0).to(device)
    )

    print(f"Categorical shape: {encoded_categorical.shape}")
    print(f"Continuous shape: {continuous_features.shape}")

    return image, encoded_categorical, continuous_features

def infer_sales_amount_and_features(model, image_path, tabular_data, config, device):
    image, categorical, continuous = preprocess_data(
        image_path, tabular_data, config, device
    )

    tabular_features = torch.cat([categorical, continuous], dim=1)
    mask = torch.ones(tabular_features.shape, device=tabular_features.device)


    model_input = [image, tabular_features, mask]

    with torch.no_grad():
        try:
            sales_amt, *predicted_features = model(model_input)
        except TypeError as e:
            print(f"TypeError 발생: {e}")
            return None, None

    sales_amt = (sales_amt.item() * config["sales_std"]) + config["sales_mean"]

    denorm_continuous = {}
    for i, col in enumerate(config["table_mean"].keys()):
        norm_val = continuous[0][i].item()
        orig_val = norm_val * config["table_std"][col] + config["table_mean"][col]
        orig_val = round(orig_val, 1)
        denorm_continuous[col] = orig_val


    decoded_categorical = {}
    for i, col in enumerate(config["categorical_columns"]):
        key = int(categorical[0][i].item())
        cat_val = config["categorical_mapping"][col].get(key, None)
        decoded_categorical[col] = cat_val

    return sales_amt, denorm_continuous, decoded_categorical

