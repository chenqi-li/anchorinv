import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.conformer import Conformer_backbone
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn_extra.cluster import KMedoids
from scipy.spatial import cKDTree
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.backends import cudnn
import random
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sn
import re
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

# torch.use_deterministic_algorithms(True)
os.environ["OMP_NUM_THREADS"] = "1"


# Adapted from https://github.com/wangkiw/TEEN

def pdist(x,y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return dist

class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()

class ProtoNetInc(nn.Module):
    
    def __init__(self,num_classes,backbone='conformer',temperature=16, softmax_t=16, shift_weight=0.5, feat_size=None):
        
        super().__init__()
        
        self.backbone = backbone

        if backbone == 'conformer':
            self.feature_extractor = Conformer_backbone(feat_size=feat_size)
            self.dim = feat_size
        else:
            print('Define the backbone')

        self.fc = nn.Linear(self.dim, num_classes, bias=False)
        self.temperature = temperature
        self.softmax_t = softmax_t
        self.shift_weight = shift_weight
        self.mode = 'proto'

    def get_feature_vector(self,inp):
        
        batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp)
        feature_vector = feature_map.view(batch_size,self.dim)
        
        return feature_vector

    def forward(self,inp):
        if self.mode == 'proto':
            feat = self.get_feature_vector(inp) 
            x = F.linear(F.normalize(feat, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            logits = self.temperature * x
        if self.mode == 'encode':
            feat = self.get_feature_vector(inp)
            logits = None
        return feat, logits
    
    def update_fc(self,eeg,label):
        data=self.get_feature_vector(eeg).detach()
        new_fc = self.update_fc_avg(data, label)

    def update_fc_avg(self,data,label):
        new_fc=[]
        for class_index in np.unique(label.cpu()):
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc
    
    def update_fc_with_tensor(self, fc_weight):
        self.fc.weight.data = fc_weight

    def soft_calibration(self, args, label_support, tau, alpha):
        base_protos = self.fc.weight.data[:len(args.source_class)].detach().cpu().data
        base_protos = F.normalize(base_protos, p=2, dim=-1)
        
        inc_classes = np.unique(label_support.cpu())
        cur_protos = self.fc.weight.data[inc_classes[0]:inc_classes[-1]+1].detach().cpu().data
        cur_protos = F.normalize(cur_protos, p=2, dim=-1)
        
        weights = torch.mm(cur_protos, base_protos.T) * tau
        norm_weights = torch.softmax(weights, dim=1)
        delta_protos = torch.matmul(norm_weights, base_protos)

        delta_protos = F.normalize(delta_protos, p=2, dim=-1)
        
        updated_protos = (1-alpha) * cur_protos + alpha * delta_protos

        self.fc.weight.data[inc_classes[0]:inc_classes[-1]+1] = updated_protos
    
def replace_base_fc(train_loader, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()
    model.module.mode = 'encode'

    embedding_list = []
    label_list = []
    with torch.no_grad():
        for i, (eeg, label, file) in enumerate(train_loader):
            embedding, logits = model(eeg)
            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(len(args.source_class)):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.weight.data[:len(args.source_class)] = proto_list
    model.module.mode = 'proto'

    return model

def calculate_sigma_points(m, P):
    n = len(m)  # Dimensionality of the distribution
    sigma_points = np.zeros((2 * n + 1, n))  # Array to store sigma points
    kappa=3.0 - len(m)

    # Calculate the matrix square root of (n+kappa)*P using Cholesky decomposition
    sqrt_P = cholesky((n + kappa) * P, lower=True)

    # First sigma point is the mean itself
    sigma_points[0] = m

    # Remaining sigma points
    for i in range(n):
        sigma_points[i + 1] = m + sqrt_P[:, i]
        sigma_points[i + 1 + n] = m - sqrt_P[:, i]
    

    return torch.tensor(sigma_points, dtype=torch.float32)

def model_inv(train_loader, model,output_dir,args,chcor_matrix=None):
    # Retrieve the device and set seeds
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_n = args.seed
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    # Initialize tracking variables
    loss_hist = []

    # Identify (embedding) targets for inversion. Outputs need to be on GPU.
    # (AnchorInv) Output: label_replay (inverted input label for finetuning later) and feat_target (the desired embedding output from the inverted input)
    if args.inv_base_select in ['sigma']: # use sigma points
        if len(train_loader)>1: # base stage
            # Save EEG embedding and corresponding label
            model = model.eval()
            model.module.mode = 'encode'
            embedding_list = []
            label_list = []
            with torch.no_grad():
                for i, (eeg, label, file) in enumerate(train_loader):
                    embedding, logits = model(eeg)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())
            embedding_list = torch.cat(embedding_list, dim=0)
            label_list = torch.cat(label_list, dim=0)
            model = model.eval()
            model.module.mode = 'proto'
            # Generate sigma points for each class
            sigma_dict = {}
            for class_index in np.unique(label_list.cpu()):
                data_index = (label_list == class_index).nonzero()
                embedding_this = embedding_list[data_index.squeeze(-1)].numpy()
                embedding_mean = np.mean(embedding_this, axis=0)
                embedding_cova = np.cov(embedding_this, rowvar=False)
                sigma_points = calculate_sigma_points(embedding_mean, embedding_cova)
                sigma_dict[class_index] = sigma_points
            # Sample sigma points to be used for inversion
            feat_target_list = []
            label_replay_list = []
            for key, value in sigma_dict.items():
                # Randomly select rows from the current Nx2360 array
                N, _ = value.shape
                indices = np.random.choice(N, args.inv_n_base_samples, replace=False)
                selected_rows = value[indices]
                # Append the selected rows and the corresponding key to the lists
                feat_target_list.append(selected_rows)
                label_replay_list.append([key] * args.inv_n_base_samples)
            feat_target = torch.tensor(np.concatenate(feat_target_list, axis=0)).to(device)
            label_replay = torch.tensor(np.concatenate(label_replay_list)).to(device)
        elif len(train_loader)==1: # incremental stage
            # Retrieve embedding of 50 samples from each base class
            with torch.no_grad():
                for i, (eeg, label, file) in enumerate(train_loader):
                    model.eval()
                    embedding, logits = model(eeg[:,:,:,:]) # [bs, 1, 22, 1000] or [bs, 1, 8, 2360]
                    feat_target = embedding
                    label_replay = label[:] # [bs]
                    break
    elif args.inv_base_select in ['kmeans', 'kmedoids']: # use cluster centroids
        if len(train_loader)>1: # base stage
            # Save EEG embedding and corresponding label
            model = model.eval()
            model.module.mode = 'encode'
            embedding_list = []
            label_list = []
            with torch.no_grad():
                for i, (eeg, label, file) in enumerate(train_loader):
                    embedding, logits = model(eeg)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())
            embedding_list = torch.cat(embedding_list, dim=0).numpy()
            label_list = torch.cat(label_list, dim=0).numpy()
            model = model.eval()
            model.module.mode = 'proto'
            # Identify the cluster centroid
            label_replay_list = []
            feat_target_list = []
            for lb in np.unique(label_list):
                indices = np.where(label_list == lb)[0]
                data = embedding_list[indices]
                if args.inv_base_select == 'kmeans':
                    kmeans = KMeans(n_clusters=args.inv_n_cluster, random_state=args.seed)
                    kmeans.fit(data)
                    centroids = kmeans.cluster_centers_
                elif args.inv_base_select == 'kmedoids':
                    kmedoids = KMedoids(n_clusters=args.inv_n_cluster, random_state=args.seed)
                    kmedoids.fit(data)
                    centroids = kmedoids.cluster_centers_
                feat_target_list.append(centroids)
                label_replay_list.append([lb] * args.inv_n_cluster)
            # Repeat the centroids to invert enough samples
            multiplier = int(args.inv_n_base_samples / args.inv_n_cluster)
            feat_target_list = [item for item in feat_target_list for _ in range(multiplier)]
            label_replay_list = [item for item in label_replay_list for _ in range(multiplier)]
            feat_target = torch.tensor(np.concatenate(feat_target_list, axis=0)).to(device)
            label_replay = torch.tensor(np.concatenate(label_replay_list)).to(device)
        elif len(train_loader)==1: # incremental stage
            # Retrieve embedding of 50 samples from each incremental class
            with torch.no_grad():
                for i, (eeg, label, file) in enumerate(train_loader):
                    model.eval()
                    embedding, logits = model(eeg[:,:,:,:]) # [bs, 1, 22, 1000] or [bs, 1, 8, 2360]
                    feat_target = embedding
                    label_replay = label[:] # [bs]
                    break
    elif args.inv_base_select in ['kmeans_basemix', 'kmedoids_basemix']: # use cluster centroids
        if len(train_loader)>1: # base stage
            # Save EEG embedding and corresponding label
            model = model.eval()
            model.module.mode = 'encode'
            embedding_list = []
            label_list = []
            with torch.no_grad():
                for i, (eeg, label, file) in enumerate(train_loader):
                    embedding, logits = model(eeg)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())
            embedding_list = torch.cat(embedding_list, dim=0).numpy()
            label_list = torch.cat(label_list, dim=0).numpy()
            model = model.eval()
            model.module.mode = 'proto'
            # Identify the cluster centroid
            label_replay_list = []
            feat_target_list = []
            if args.inv_base_select == 'kmeans_basemix':
                kmeans = KMeans(n_clusters=args.inv_n_cluster, random_state=args.seed)
                kmeans.fit(embedding_list)
                centroids = kmeans.cluster_centers_
            elif args.inv_base_select == 'kmedoids_basemix':
                kmedoids = KMedoids(n_clusters=args.inv_n_cluster, random_state=args.seed)
                kmedoids.fit(embedding_list)
                centroids = kmedoids.cluster_centers_
            # Find the appropriate label for the centroids  
            tree = cKDTree(embedding_list)
            distances, indices = tree.query(centroids, k=1)
            feat_target_list.append(centroids)
            label_replay_list.append(label_list[indices])
            # Repeat the centroids to invert enough samples
            multiplier = int(args.inv_n_base_samples * len(args.source_class) / args.inv_n_cluster)
            feat_target_list = [item for item in feat_target_list for _ in range(multiplier)]
            label_replay_list = [item for item in label_replay_list for _ in range(multiplier)]
            feat_target = torch.tensor(np.concatenate(feat_target_list, axis=0)).to(device)
            label_replay = torch.tensor(np.concatenate(label_replay_list)).to(device)
        elif len(train_loader)==1: # incremental stage
            # Retrieve embedding of 50 samples from each base class
            with torch.no_grad():
                for i, (eeg, label, file) in enumerate(train_loader):
                    model.eval()
                    embedding, logits = model(eeg[:,:,:,:]) # [bs, 1, 22, 1000] or [bs, 1, 8, 2360]
                    feat_target = embedding
                    label_replay = label[:] # [bs]
                    break
    elif args.inv_base_select in ['closest', 'farthest','spaced', 'closest_cos', 'farthest_cos', 'spaced_cos', 'closest_0.3_random', 'closest_0.5_random', 'closest_0.7_random', 'correct_random']: # use embeddings closest to class prototype
        if len(train_loader)>1: # base stage
            # Save EEG embedding and corresponding label
            model = model.eval()
            model.module.mode = 'proto'
            embedding_list = []
            label_list = []
            logits_list = []
            with torch.no_grad():
                for i, (eeg, label, file) in enumerate(train_loader):
                    embedding, logits = model(eeg)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())
                    logits_list.append(logits.cpu())
            embedding_list = torch.cat(embedding_list, dim=0).numpy()
            label_list = torch.cat(label_list, dim=0).numpy()
            logits_list = torch.cat(logits_list, dim=0).numpy()
            model = model.eval()
            model.module.mode = 'proto'
            # Identify the closest samples to prototype
            label_replay_list = []
            feat_target_list = []
            for lb in np.unique(label_list):
                indices = np.where(label_list == lb)[0]
                data = embedding_list[indices]
                label_class = label_list[indices]
                logits_class = logits_list[indices]
                class_prototype = data.mean(0)
                if 'cos' in args.inv_base_select:
                    distances = F.linear(F.normalize(torch.from_numpy(data), p=2, dim=-1), F.normalize(torch.from_numpy(class_prototype), p=2, dim=-1))
                    distances = distances * -1 # because in euclidean, lower is closer, but in cosine, higher is closer
                else:
                    distances = np.linalg.norm(data - class_prototype, axis=1)
                if args.inv_base_select in ['closest', 'closest_cos']:
                    selected_indices = np.argsort(distances)[:args.inv_n_base_samples]
                elif args.inv_base_select in ['closest_0.3_random','closest_0.5_random','closest_0.7_random']:
                    eligible_indices = np.argsort(distances)[:int(float(re.search(r'\d+\.\d+', args.inv_base_select).group(0))*len(indices))]
                    selected_indices = np.random.choice(eligible_indices, size=min(args.inv_n_base_samples, len(eligible_indices)), replace=False)
                elif args.inv_base_select in ['farthest', 'farthest_cos']:
                    selected_indices = np.argsort(distances)[-args.inv_n_base_samples:]
                elif args.inv_base_select in ['spaced', 'spaced_cos']:
                    sorted_indices = np.argsort(distances)
                    selected_indices = sorted_indices[np.linspace(0, len(sorted_indices) - 1, args.inv_n_base_samples, dtype=int)]
                elif args.inv_base_select in ['correct_random']:
                    predicted_labels = np.argmax(logits_class, axis=1)
                    eligible_indices = np.where(predicted_labels == label_class)[0]
                    selected_indices = np.random.choice(eligible_indices, size=min(args.inv_n_base_samples, len(eligible_indices)), replace=False)
                selected_rows = data[selected_indices]
                feat_target_list.append(selected_rows)
                label_replay_list.append([lb] * args.inv_n_base_samples)
            feat_target = torch.tensor(np.concatenate(feat_target_list, axis=0)).to(device)
            label_replay = torch.tensor(np.concatenate(label_replay_list)).to(device)
        elif len(train_loader)==1: # incremental stage
            # Retrieve embedding of 50 samples from each base class
            with torch.no_grad():
                for i, (eeg, label, file) in enumerate(train_loader):
                    model.eval()
                    embedding, logits = model(eeg[:,:,:,:]) # [bs, 1, 22, 1000] or [bs, 1, 8, 2360]
                    feat_target = embedding
                    label_replay = label[:] # [bs]
                    break
    elif args.inv_base_select in ['tsne']:
        if len(train_loader)>1: # base stage
            # Save EEG embedding and corresponding label
            model = model.eval()
            model.module.mode = 'encode'
            embedding_list = []
            label_list = []
            with torch.no_grad():
                for i, (eeg, label, file) in enumerate(train_loader):
                    embedding, logits = model(eeg)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())
            embedding_list = torch.cat(embedding_list, dim=0).numpy()
            label_list = torch.cat(label_list, dim=0).numpy()
            model = model.eval()
            model.module.mode = 'proto'
            for selection in ['closest', 'random']: #['closest', 'farthest', 'spaced', 'random']:
                # Identify the closest samples to prototype
                all_label_list = []
                all_feat_list = []
                all_shape_list = []
                all_size_list = []
                # Add existing ones
                all_feat_list.append(embedding_list)
                all_label_list.append(label_list)
                all_shape_list.append(['o']*label_list.shape[0])
                all_size_list.append([10]*label_list.shape[0])
                # Calulate the prototype and selected indices
                for lb in np.unique(label_list):
                    indices = np.where(label_list == lb)[0]
                    data = embedding_list[indices]
                    class_prototype = data.mean(0)
                    distances = np.linalg.norm(data - class_prototype, axis=1)
                    if selection in ['random']:
                        selected_indices = np.random.choice(indices, size=min(args.inv_n_base_samples, len(indices)), replace=False)
                        selected_rows = embedding_list[selected_indices]
                    elif selection in ['closest']:
                        selected_indices = np.argsort(distances)[:args.inv_n_base_samples]
                        selected_rows = data[selected_indices]
                    elif selection in ['farthest']:
                        selected_indices = np.argsort(distances)[-args.inv_n_base_samples:]
                        selected_rows = data[selected_indices]
                    elif selection in ['spaced']:
                        sorted_indices = np.argsort(distances)
                        selected_indices = sorted_indices[np.linspace(0, len(sorted_indices) - 1, args.inv_n_base_samples, dtype=int)]
                        selected_rows = data[selected_indices]
                    # Change the shape and size for selected rows in embedding_list
                    retrieved_indices = []
                    for row in selected_rows:
                        # Check where each row of selected_rows matches a row in embedding_list
                        matches = np.all(embedding_list == row, axis=1)
                        if np.any(matches):  # Check if there is a match
                            index = np.where(matches)[0][0]  # Get the index of the match
                            retrieved_indices.append(index)
                    for retrieved_index in retrieved_indices:
                        all_shape_list[0][retrieved_index] = 'P'
                        all_size_list[0][retrieved_index] = 100
                    # all_feat_list.append(selected_rows)
                    # all_label_list.append([lb] * args.inv_n_base_samples)
                    # all_shape_list.append(['P']*args.inv_n_base_samples)
                    # all_size_list.append([100]*args.inv_n_base_samples)
                    all_feat_list.append(class_prototype.reshape(1,-1))
                    all_label_list.append([lb])
                    all_shape_list.append(['*'])
                    all_size_list.append([500])

                    # Plot histogram of distances
                    plt.hist(distances)
                    # Show plot
                    plt.show()
                    img_path = os.path.join(output_dir+'_loadevaluate'+'_'+str(args.eval_suffix),'distance_hist_plots')
                    if not os.path.exists(img_path):
                        os.makedirs(img_path)
                    plt.savefig(os.path.join(img_path, f'class_{lb}'))
                    plt.close()
                all_feat = np.concatenate(all_feat_list, axis=0)
                all_label = np.concatenate(all_label_list)
                all_shape = np.concatenate(all_shape_list)
                all_size = np.concatenate(all_size_list)
                # TSNE visualization
                perplexity_list = [30] #, 5, 10, 20, 1, 40, 50, 100
                n_iter_list = [1000]
                for perplexity in perplexity_list:
                    for n_iter in n_iter_list:
                        tsne = TSNE(n_components=2, verbose=1, random_state=123, n_iter=n_iter, perplexity=perplexity)
                        z = tsne.fit_transform(all_feat)
                        df = pd.DataFrame()
                        df["y"] = all_label.squeeze()
                        df["comp-1"] = z[:,0]
                        df["comp-2"] = z[:,1]
                        # df["pred"] = epoch_eval_pred_all.squeeze()
                        df["shapes"] = np.array(all_shape).squeeze()
                        df["sizes"] = np.array(all_size).squeeze()

                        # Plot with color based on GT label
                        # plt.figure(figsize=(12, 12))
                        # figure = sn.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), style=df.shapes.tolist(), size=df.sizes.tolist(), sizes=df.sizes.tolist(),
                        #         palette=sn.color_palette("coolwarm", 4), data=df, legend='brief')
                        # Create scatter plot
                        fig, ax = plt.subplots(figsize=(18, 12))

                        # Assuming df["y"] contains numeric class labels like 0, 1, 2, ..., n
                        colors = df["y"].astype(int)  # Convert to integer if not already
                        unique_classes = np.unique(colors)

                        # Choose the appropriate colormap
                        if len(unique_classes) == 2:
                            colormap = 'coolwarm'  # Works well for binary classes
                        elif len(unique_classes) <= 10:
                            colormap = 'tab10'  # Works well for up to 10 classes

                        # Create a BoundaryNorm to ensure discrete colors for each class
                        # norm = mcolors.BoundaryNorm(boundaries=np.arange(len(unique_classes) + 1) - 0.5, ncolors=len(unique_classes))

                        # Create scatter plot
                        fig, ax = plt.subplots(figsize=(16, 12))
                        for shape in ['o', 'P', '*']:
                            subset = df[df["shapes"] == shape]
                            scatter = ax.scatter(subset["comp-1"], subset["comp-2"], s=subset["sizes"], 
                                                c=subset["y"], cmap=colormap, 
                                                label=f'Shape: {shape}', alpha=0.6, edgecolors='w', marker=shape)

                        # Adding labels and title
                        ax.set_xlabel('Component 1', fontsize=24)
                        ax.set_ylabel('Component 2', fontsize=24)
                        ax.tick_params(axis='x', labelsize=20)
                        ax.tick_params(axis='y', labelsize=20)

                        # # Adding a color bar to show the mapping of y values to colors
                        # cbar = plt.colorbar(scatter, ticks=unique_classes)
                        # cbar.set_label('Class Label', fontsize=24)
                        # cbar.ax.tick_params(labelsize=20)

                        # Create a custom legend for shapes
                        shape_legend = [plt.Line2D([0], [0], marker='*', color='w', label='Class Prototype',
                                                    markerfacecolor='black', markersize=20),
                                        plt.Line2D([0], [0], marker='P', color='w', label='Anchor Points',
                                                    markerfacecolor='black', markersize=16),
                                        plt.Line2D([0], [0], marker='o', color='w', label='Remaining Feature Vector',
                                                    markerfacecolor='black', markersize=10)]

                        # Adding the first legend in the best location
                        legend1 = ax.legend(handles=shape_legend, prop={'size': 24}, loc='best')
                        ax.add_artist(legend1)

                        # Add color legend above the plot, spread horizontally
                        colors = [plt.get_cmap(colormap)(color_i / (len(unique_classes)-1)) for color_i in range(len(unique_classes))]
                        legend_patches = [patches.Patch(color=color, label=f'{color_label}') 
                                        for color, color_label in zip(colors, unique_classes)]
                        ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=len(unique_classes), prop={'size': 20})

                        # Show plot
                        plt.tight_layout()
                        plt.show()
                        # figure.set(title="T-SNE projection")
                        # figure = figure.get_figure()
                        img_path = os.path.join(output_dir+'_loadevaluate'+'_'+str(args.eval_suffix),'tsne_plots')
                        if not os.path.exists(img_path):
                            os.makedirs(img_path)
                        fig.savefig(os.path.join(img_path, f'selection_{selection}_perplexity_{perplexity}_niter_{n_iter}'),bbox_inches='tight',dpi=300)
                        plt.close()
                        
        elif len(train_loader)==1: # incremental stage
            # Retrieve embedding of 50 samples from each base class
            with torch.no_grad():
                for i, (eeg, label, file) in enumerate(train_loader):
                    model.eval()
                    embedding, logits = model(eeg[:,:,:,:]) # [bs, 1, 22, 1000] or [bs, 1, 8, 2360]
                    feat_target = embedding
                    label_replay = label[:] # [bs]
                    break
    elif args.inv_base_select in ['kmeans_silhouette', 'kmedoids_silhouette']: # only for choosing number of clusters, does not invert
        if len(train_loader)>1: # base stage
            # Save EEG embedding and corresponding label
            model = model.eval()
            model.module.mode = 'encode'
            embedding_list = []
            label_list = []
            with torch.no_grad():
                for i, (eeg, label, file) in enumerate(train_loader):
                    embedding, logits = model(eeg)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())
            embedding_list = torch.cat(embedding_list, dim=0).numpy()
            label_list = torch.cat(label_list, dim=0).numpy()
            model = model.eval()
            model.module.mode = 'proto'
            # Identify the cluster centroid
            label_replay_list = []
            feat_target_list = []
            for lb in np.unique(label_list):
                indices = np.where(label_list == lb)[0]
                data = embedding_list[indices]
                for n_clusters in range(2,51):
                    if args.inv_base_select == 'kmeans_silhouette':
                        kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed)
                        cluster_labels = kmeans.fit_predict(data)
                        silhouette_avg = silhouette_score(data, cluster_labels)
                        print(
                            "For n_clusters =",
                            n_clusters,
                            "The average silhouette_score is :",
                            silhouette_avg,
                        )
                        centroids = kmeans.cluster_centers_
                    elif args.inv_base_select == 'kmedoids_silhouette':
                        kmedoids = KMedoids(n_clusters=n_clusters, random_state=args.seed)
                        cluster_labels = kmedoids.fit_predict(data)
                        silhouette_avg = silhouette_score(data, cluster_labels)
                        print(
                            "For n_clusters =",
                            n_clusters,
                            "The average silhouette_score is :",
                            silhouette_avg,
                        )
                        centroids = kmedoids.cluster_centers_
            # feat_target = torch.tensor(np.concatenate(feat_target_list, axis=0)).to(device) #torch.Size([100, 2360]) <class 'torch.Tensor'> torch.float32
            # label_replay = torch.tensor(np.concatenate(label_replay_list)) #torch.Size([100]) <class 'torch.Tensor'> torch.int64
    elif args.inv_base_select in ['random_sample']: # randomly select min(args.inv_n_base_samples, available samples) real samples to invert
        if len(train_loader)>1: # base stage
            with torch.no_grad():
                for i, (eeg, label, file) in enumerate(train_loader):
                    # Pick args.inv_n_base_samples for each class
                    unique_labels = np.unique(label.cpu())
                    selected_samples = []
                    for lb in unique_labels:
                        indices = np.where(label.cpu() == lb)[0]
                        selected_indices = np.random.choice(indices, size=min(args.inv_n_base_samples, len(indices)), replace=False)
                        selected_samples.extend(selected_indices)
                    # Pass the selected samples through the model to get its embedding, also save its label
                    model.eval()
                    embedding, logits = model(eeg[selected_samples,:,:,:]) # [bs, 1, 22, 1000] or [bs, 1, 8, 2360]
                    feat_target = embedding
                    label_replay = label[selected_samples] # [bs]
                    print(f'Files selected are: {np.array(file)[selected_samples]}')
                    break
        elif len(train_loader)==1: # incremental stage
            # Retrieve embedding of 50 samples from each base class
            with torch.no_grad():
                for i, (eeg, label, file) in enumerate(train_loader):
                    model.eval()
                    embedding, logits = model(eeg[:,:,:,:]) # [bs, 1, 22, 1000] or [bs, 1, 8, 2360]
                    feat_target = embedding
                    label_replay = label[:] # [bs]
                    break
    
    # Copy the replay to prepare for breakdown of inversion into chunks
    label_replay_full = deepcopy(label_replay.cpu().detach())
    feat_target_full = deepcopy(feat_target.cpu().detach())
    torch.cuda.empty_cache()
    input_tensor_list = []

    # Breakdown the inversion into chunks for base session of GRABM
    if 'GRABMdataset' in args.dataloader_dir and len(train_loader)>1:
        replay_skip = args.inv_n_base_samples*2
        replay_breakdown = range(0, label_replay_full.size()[0], replay_skip)
    # Otherwise, do inversion for the full batch
    else:
        replay_skip = label_replay_full.size()[0]
        replay_breakdown = range(0, label_replay_full.size()[0], replay_skip)

    # Inversion process
    for replay_index in replay_breakdown:
        label_replay = deepcopy(label_replay_full[replay_index:replay_index+replay_skip].cpu().detach()).to(device)
        feat_target = deepcopy(feat_target_full[replay_index:replay_index+replay_skip].cpu().detach()).to(device)
        # Initialize the input tensor
        if (args.inv_base_select in ['sigma', 'kmedoids', 'kmeans', 'random_sample', 'closest', 'farthest', 'spaced', 'closest_cos', 'farthest_cos', 'spaced_cos', 'condensation', 'kmeans_basemix', 'closest_0.3_random', 'closest_0.5_random', 'closest_0.7_random', 'correct_random']):
            if 'NHIEdataset' in args.dataloader_dir:
                if args.inv_init == 'randn':
                    input_tensor = torch.randn([label_replay.size()[0], 1, 8, 3840], requires_grad=True, device=device)
                elif args.inv_init == 'zeros':
                    input_tensor = torch.zeros([label_replay.size()[0], 1, 8, 3840], requires_grad=True, device=device)
            elif 'BCIdataset' in args.dataloader_dir:
                if args.inv_init == 'randn':
                    input_tensor = torch.randn([label_replay.size()[0], 1, 22, 1000], requires_grad=True, device=device)
                elif args.inv_init == 'zeros':
                    input_tensor = torch.zeros([label_replay.size()[0], 1, 22, 1000], requires_grad=True, device=device)
            elif 'GRABMdataset' in args.dataloader_dir:
                if args.inv_init == 'randn':
                    input_tensor = torch.randn([label_replay.size()[0], 1, 28, 1280], requires_grad=True, device=device)
                elif args.inv_init == 'zeros':
                    input_tensor = torch.zeros([label_replay.size()[0], 1, 28, 1280], requires_grad=True, device=device)

        # Define the loss function (e.g., Mean Squared Error)
        if args.inv_loss == 'MAE':
            loss_fn = torch.nn.L1Loss()
        elif args.inv_loss == 'MSE':
            loss_fn = torch.nn.MSELoss()
        elif args.inv_loss == 'cos':
            loss_fn = torch.nn.CosineEmbeddingLoss()
        # Auxiliary losses
        if args.inv_l2_scale == 0:
            loss_l2 = 0
        if args.inv_tv_l1_scale == 0:
            loss_var_l1 = 0
        if args.inv_tv_l2_scale == 0:
            loss_var_l2 = 0
        if args.inv_bn_l_scale == 0:
            loss_r_feature = 0
        if args.inv_chcor_scale == 0:
            loss_chcor = 0
        if args.inv_laplacian_scale == 0:
            loss_laplacian = 0
        ## (Optional BN Loss) Create hooks for feature statistics
        if args.inv_bn_l_scale > 0:
            loss_r_feature_layers = []
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    loss_r_feature_layers.append(DeepInversionFeatureHook(module))

        # Define the optimizer (e.g., SGD)
        if args.inv_optim == 'SGD':
            optimizer = torch.optim.SGD([input_tensor], lr=args.inv_lr) #lr=10
        elif args.inv_optim == 'Adam':
            optimizer = torch.optim.Adam([input_tensor], lr=args.inv_lr) #lr=0.01
        
        # Invert for args.inv_steps steps
        for i in tqdm(range(args.inv_steps)):
            # Forward pass
            feat, logits = model(input_tensor)
            # Calculate the losses
            if args.inv_loss in ['MAE', 'MSE']:
                loss_target = loss_fn(feat.squeeze(), feat_target)
            elif args.inv_loss in ['cos']:
                loss_target = loss_fn(feat.squeeze(), feat_target, torch.ones(feat_target.size()[0]).to(device))
            if args.inv_l2_scale > 0: # L2 norm loss
                loss_l2 = torch.norm(input_tensor[:,:,:,:].view(input_tensor.size()[0], -1)[:,:], dim=1).mean() # L2 norm
            if args.inv_tv_l1_scale > 0 or args.inv_tv_l2_scale > 0: # total variance loss
                diff1 = torch.diff(input_tensor[:,:,:,:], dim=-1)
                if args.inv_tv_l2_scale > 0: # total variance L2
                    loss_var_l2 = torch.norm(diff1) 
                if args.inv_tv_l1_scale > 0: # total variance L1
                    loss_var_l1 = torch.mean(diff1.abs())
            if args.inv_bn_l_scale > 0: # batch norm loss
                rescale = [args.inv_bn_l_scale] + [1. for _ in range(len(loss_r_feature_layers)-1)] # bn loss scaling
                loss_r_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)]) # bn regularization
            if args.inv_chcor_scale > 0: #channel correlation loss
                # Find all pairs with correlation greater than 0.95
                threshold = 0.95
                pairs = []
                loss_chcor = 0
                # Iterate over the upper triangle of the matrix
                for ch_row in range(chcor_matrix.shape[0]):
                    for ch_col in range(ch_row + 1, chcor_matrix.shape[1]):
                        if chcor_matrix[ch_row, ch_col] > threshold:
                            pairs.append((ch_row, ch_col))
                for pair in pairs:
                    channel_1, channel_2 = pair
                    difference = torch.sum(input_tensor[:, :, channel_1, :] - input_tensor[:, :, channel_2, :])
                    loss_chcor += abs(difference)
            if args.inv_laplacian_scale > 0: # laplacian loss
                loss_laplacian = torch.mean((input_tensor[:,:,:,2:] - 2*input_tensor[:,:,:,1:-1] + input_tensor[:,:,:,:-2])**2)
            loss = loss_target + args.inv_l2_scale*loss_l2 + args.inv_tv_l1_scale*loss_var_l1 + args.inv_tv_l2_scale*loss_var_l2 \
                    + args.inv_bn_l_scale*loss_r_feature +args.inv_chcor_scale*loss_chcor + args.inv_laplacian_scale*loss_laplacian
            loss_hist.append(loss.item())
            # print(f'Totalloss: {loss.item()}, Main loss: {loss_target.item()}, L2 loss: {loss_l2}, Variance L2 loss: {loss_var_l2}, Variance L1 loss: {loss_var_l1}, BN Loss: {loss_r_feature}') # PRINT TO CHOOSE LOSS SCALE
            # Backward pass: compute the gradient of the loss with respect to the input
            loss.backward()
            # Update the input using the optimizer
            optimizer.step()
            # Zero the gradients for the next iteration
            optimizer.zero_grad()
        # Detach the hoook after inversion process
        if args.inv_bn_l_scale > 0:
            for layer in loss_r_feature_layers:
                layer.close()

        # Append input_tensor to the full list
        input_tensor_list.append(deepcopy(input_tensor.cpu().detach()))

    # Accumulate all input_tensor
    input_tensor = torch.cat(input_tensor_list, axis=0).to(device)
    label_replay = deepcopy(label_replay_full)

    # Print inversion result
    feat, logits = model(input_tensor)
    print(f'Inversion at step {i} Total loss: {loss.item()}, Target loss: {loss_target.item()}, L2 scale: {args.inv_l2_scale}, L2 loss: {loss_l2}, '
        f'TVL2 scale: {args.inv_tv_l2_scale}, Variance L2 loss: {loss_var_l2}, TVL1 scale: {args.inv_tv_l1_scale}, Variance L1 loss: {loss_var_l1}, '
        f'BN scale {args.inv_bn_l_scale}, BN Loss, {loss_r_feature}, ChCor scale {args.inv_chcor_scale}, ChCor Loss, {loss_chcor}, '
        f'Laplacian scale {args.inv_laplacian_scale}, Laplacian Loss, {loss_laplacian}')
    print(f'Inverted input yields {logits.size()} output logits {logits}')
    print(f'Target classes of shape {label_replay.size()} are {label_replay.squeeze()}')
    print(f'Inverted input yields {feat.size()} model embedding {feat.squeeze()}')
    print(f'Original input yields {feat_target_full.size()} model embedding {feat_target_full.squeeze()}')        



    # Plot loss curve and example inverted EEG
    if args.n_trial>1 and args.incremental_shots>1:
        # Plot loss curve and example inverted EEG
        offset_scale = 25
        plt.figure(figsize=(15, 20))
        if 'NHIEdataset' in args.dataloader_dir:
            offset = np.linspace(-offset_scale, offset_scale, 8).reshape(8,1)
        elif 'BCIdataset' in args.dataloader_dir:
            offset = np.linspace(-offset_scale, offset_scale, 22).reshape(22,1)
        elif 'GRABMdataset' in args.dataloader_dir:
            offset = np.linspace(-offset_scale, offset_scale, 28).reshape(28,1)
        inverted_input = input_tensor.cpu().detach().numpy().squeeze()+offset
        real_input = eeg.cpu().detach().numpy().squeeze()+offset
        plt.subplot(2, 1, 1)
        plt.plot(inverted_input[0,:,:200].squeeze().T)
        plt.subplot(2, 1, 2)
        plt.plot(real_input[0,:,:200].squeeze().T)
        # plt.show()
        img_path = os.path.join(output_dir+'_loadevaluate'+'_'+str(args.eval_suffix),'InvImg')
        if not os.path.exists(img_path):
            os.makedirs(img_path)
    
    # Save the inverted input (eeg_replay) and label of the inverted input (label_replay) to the model
    eeg_replay = deepcopy(input_tensor.cpu().detach())
    if hasattr(model, 'eeg_replay') == False:
        model.eeg_replay = nn.Parameter(deepcopy(eeg_replay.cpu().detach()), requires_grad=False)
        model.label_replay = nn.Parameter(deepcopy(label_replay.cpu().detach()), requires_grad=False)
    elif hasattr(model, 'eeg_replay') == True:
        cur_session_replay_eeg = torch.cat((deepcopy(model.eeg_replay), eeg_replay.cpu().detach()), dim=0)
        cur_session_replay_label = torch.cat((deepcopy(model.label_replay), label_replay.cpu().detach()), dim=0)
        model.eeg_replay = nn.Parameter(deepcopy(cur_session_replay_eeg), requires_grad=False)
        model.label_replay = nn.Parameter(deepcopy(cur_session_replay_label), requires_grad=False)
    # Print status of replay
    print(f'EEG replay is now size {model.eeg_replay.size()}, dtype {model.eeg_replay.dtype}')
    print(f'Label replay is now size {model.label_replay.size()}, dtype {model.label_replay.dtype}, labels {np.unique(model.label_replay.cpu())}')
    return model