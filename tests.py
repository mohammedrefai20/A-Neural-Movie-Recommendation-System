import torch.nn as nn
import torch
import visualtorch
from collections import defaultdict
import matplotlib.pyplot as plt



class NCF(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim=32, mlp_layers=[64, 32, 16]):
        super(NCF, self).__init__()

        # Embeddings for the MLP branch
        self.user_embedding_mlp = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding_mlp = nn.Embedding(n_movies, embedding_dim)

        # Embeddings for the Matrix-Factorization (dot product) branch
        self.user_embedding_mf = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding_mf = nn.Embedding(n_movies, embedding_dim)

        # MLP layers
        mlp_modules = []
        input_size = embedding_dim * 2
        for layer_size in mlp_layers:
            mlp_modules.append(nn.Linear(input_size, layer_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(0.2))
            input_size = layer_size
        self.mlp = nn.Sequential(*mlp_modules)

        # Final layer: combines MF branch output (1 value) + MLP branch output
        self.output_layer = nn.Linear(mlp_layers[-1] + embedding_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for emb in [self.user_embedding_mlp, self.movie_embedding_mlp,
                    self.user_embedding_mf, self.movie_embedding_mf]:
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, user_idx, movie_idx):
        user_idx = user_idx.long()
        movie_idx = movie_idx.long()

        user_mf = self.user_embedding_mf(user_idx)
        movie_mf = self.movie_embedding_mf(movie_idx)
        mf_vector = user_mf * movie_mf

        user_mlp = self.user_embedding_mlp(user_idx)
        movie_mlp = self.movie_embedding_mlp(movie_idx)
        mlp_input = torch.cat([user_mlp, movie_mlp], dim=-1)
        mlp_vector = self.mlp(mlp_input)

        combined = torch.cat([mlp_vector, mf_vector], dim=-1)
        output = self.output_layer(combined)

        return output.squeeze()



model = NCF(600, 5000, 128)




model.eval()
# input_shape = input_shape = (1, 1, 28, 28)
color_map: dict = defaultdict(dict)
color_map[nn.Conv2d]["fill"] = "#E69F00"
color_map[nn.BatchNorm2d]["fill"] = "#009E73"
color_map[nn.ReLU]["fill"] = "#56B4E9"


# One shape per forward() argument: (image, vector).
input_shape = ((1, 128), (1, 128))

color_map: dict = defaultdict(dict)
color_map[nn.Conv2d]["fill"] = "#E69F00"
color_map[nn.Linear]["fill"] = "#56B4E9"

img = visualtorch.render(
    model,
    input_shape,
    style="flow",
    color_map=color_map,
    scale_xy=3,
    spacing=15,
    show_dimension=True,
)

dpi = 150  # rendered at 2x this in the final doc build (savefig.dpi=300 in conf.py)
plt.figure(figsize=(img.width / dpi, img.height / dpi), dpi=dpi)
plt.imshow(img)
plt.axis("off")
plt.tight_layout()
plt.show()