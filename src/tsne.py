import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchinfo import summary
import h5py
import datetime as dt
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import math
from sklearn.manifold import TSNE
import random
import hydra
from omegaconf import DictConfig

# Tambahkan path src ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

class_names = ['Normal', 'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema',
               'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

def generate_feature_maps_before(model, dataloader, device):
    res_dict = {'feats':[], 'labels':[]}
    timestamp1 = dt.datetime.now()
    model.eval()
    model.to(device)
    i = 0
    for inputs, classes in dataloader:
        inputs = inputs.to(device)
        labels = classes.to(device)
        outputs = model(inputs)
        for j in range(len(outputs)):
            res_dict['feats'].append(outputs[j].cpu().detach().numpy())
            res_dict['labels'].append(class_names[classes[j]])
        i += 1
        print(f"Processed {i} batches . . .")
    
    timestamp2 = dt.datetime.now()
    print(f"The process took {(timestamp2 - timestamp1).total_seconds()} seconds")
    return res_dict

def initialize_model(cfg):
    torch.manual_seed(cfg.model.seed)
    if cfg.model.backbone == "resnet":
        from models.ResNet50 import ResNet50_Model as net
    elif cfg.model.backbone == "densenet":
        from models.DenseNet121 import DenseNet121_Model as net
    elif cfg.model.backbone == "vit":
        from models.VisionTransformer import ViT_Model as net
    else:
        raise TypeError(cfg.model.backbone)
    model_cfg_data = cfg.model
    if 'unfrozen_blocks' not in model_cfg_data:
        model_cfg_data['unfrozen_blocks'] = 0
    return net(model_cfg_data)

def calc_inertia(x, y):
    cent = (x.mean(), y.mean())
    d_points = list(map(lambda a, b: (a, b), x, y))
    inert = np.mean(list(map(lambda a: math.dist(a, cent), d_points)))
    return d_points, cent, inert

def evaluate_clusters(df, label_col):
    metric_df = pd.DataFrame(columns=['centroid_tsne_1', 'centroid_tsne_2', 'inertia'])
    
    for i in df[label_col].unique():
        cur_df = df[df[label_col] == i]
        _, cent, inert = calc_inertia(cur_df['tsne_1'], cur_df['tsne_2'])
        metric_df = pd.concat([metric_df, pd.DataFrame({'centroid_tsne_1': [cent[0]],
                                                        'centroid_tsne_2': [cent[1]],
                                                        'inertia': [inert]}, index=[i])])
    return metric_df

@hydra.main(version_base=None, config_path="../config", config_name="tsne-config")
def main(cfg: DictConfig):
    seed_value = cfg.seed_value
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.use_deterministic_algorithms(False)
    torch.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = initialize_model(cfg)
    torch.save(model.state_dict(), 'model_checkpoint.pt')
    model.eval()

    BASE_EXP_DIR = './'
    model_filename = "model_checkpoint.pt"
    target_path = os.path.join(BASE_EXP_DIR, model_filename)

    if os.path.exists(target_path):
        model.load_state_dict(torch.load(target_path, map_location=torch.device('cpu')))
        model.fc = nn.Identity()
    else:
        print(f"Checkpoint file '{model_filename}' not found in the specified directory.")

    model.eval()

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset_dir = 'dataset'
    image_datasets = datasets.ImageFolder(dataset_dir, data_transforms)

    dataloaders = torch.utils.data.DataLoader(image_datasets, 
                                              batch_size=4,
                                              shuffle=False, 
                                              num_workers=4,
                                              generator=g)

    inputs, classes = next(iter(dataloaders))
    inputs, classes

    model.to(device)
    res_list = generate_feature_maps_before(model, dataloaders, device)

    f = h5py.File('feature_maps.hdf5', 'w')
    ds_feats = f.create_dataset("feature_maps", data=res_list['feats'], dtype='float64')
    ds_labs = f.create_dataset("labels", data=res_list['labels'])

    f.close()

    file_dir = 'feature_maps.hdf5'
    f = h5py.File(file_dir, 'r')

    feature_maps = np.array(f['feature_maps'])
    original_shape = feature_maps.shape
    print(f"Original shape of feature_maps: {original_shape}")

    feat_data = np.array(f.get('feature_maps')).reshape(original_shape)

    tsne = TSNE(n_components=cfg.tsne.n_components,
                verbose=cfg.tsne.verbose,
                perplexity=cfg.tsne.perplexity,
                n_iter=cfg.tsne.n_iter,
                random_state=cfg.seed_value)
    tsne_results = tsne.fit_transform(feat_data)
    print("t-SNE completed.")
    print(tsne_results)

    tsne_file_dir = 'tsne_results.hdf5'
    with h5py.File(tsne_file_dir, 'w') as tsne_file:
        tsne_file.create_dataset("tsne_results", data=tsne_results)
        print(f"t-SNE results saved to {tsne_file_dir}")
    tsne_file.close()

    labels = list(map(lambda st: st.decode("utf-8"), list(f.get('labels'))))

    sns.set(font_scale=1.2)

    palette = sns.color_palette("tab20", len(class_names))
    
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=labels,
        palette=palette,
        s=50
    )

    plt.savefig("t-sne.png", dpi=100)

    df = pd.DataFrame(data={'tsne_1': tsne_results[:, 0], 
                            'tsne_2': tsne_results[:, 1], 
                            'labels': labels})

    df.head()

    metric_df = evaluate_clusters(df, 'labels')
    metric_df.index

    sns.set(font_scale=1.5)

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=labels,
        palette=palette,
        s=50
    )

    for i in range(metric_df.shape[0]):
        ax.add_patch(
            plt.Circle((metric_df['centroid_tsne_1'].iloc[i], metric_df['centroid_tsne_2'].iloc[i]),
                       metric_df['inertia'].iloc[i], alpha=0.2, 
                       edgecolor='k', linestyle='--'))

    plt.savefig("t-sne2.png", dpi=300)

    f.close()

if __name__ == '__main__':
    main()