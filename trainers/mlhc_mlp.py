import torch
import torch.nn as nn

class BinaryClassificationMLP(nn.Module):
    def __init__(self, mlp_input_size, mlp_hidden_size1, mlp_hidden_size2, mlp_output_size):
        super(BinaryClassificationMLP, self).__init__()
        self.input_layer = nn.Linear(mlp_input_size, mlp_hidden_size1)
        self.hidden_layer1 = nn.Linear(mlp_hidden_size1, mlp_hidden_size2)
        self.hidden_layer2 = nn.Linear(mlp_hidden_size2, mlp_output_size)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer1(x))
        output = torch.sigmoid(self.hidden_layer2(x))  # Sigmoid activation for binary classification
        return output

class TransformerClassifier(nn.Module):
    def __init__(self, t_input_size, t_hidden_size, t_num_attention_heads, t_num_hidden_layers, t_output_size):
        super(TransformerClassifier, self).__init__()
        
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=t_input_size,
                nhead=t_num_attention_heads,
                dim_feedforward=t_hidden_size,
                # num_layers=t_num_hidden_layers
            ),
            num_layers=t_num_hidden_layers
        )
        
        # Output layer
        self.output_layer = nn.Linear(t_input_size, t_output_size)
        
    def forward(self, x):
        # Input x: (B, 512) embedding
        
        # Apply transformer
        transformer_output = self.transformer(x)
        # Apply output layer
        output = self.output_layer(transformer_output)  # Take only the first token's embedding
        
        # Apply softmax for binary classification soft labels
        output_softmax = torch.softmax(output, dim=1)
        
        return output

