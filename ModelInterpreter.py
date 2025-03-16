import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

class ModelInterpreter:
    def __init__(self, model="gpt2-small", release="gpt2-small-res-jb", sae_id="blocks.8.hook_resid_pre"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the transformer model
        self.model = HookedTransformer.from_pretrained(model).to(self.device)
        self.release = release
        self.sae_id = sae_id

        # Load the sparse autoencoder
        self.sae, self.cfg_dict, self.sparsity = SAE.from_pretrained(
            release=self.release,
            sae_id=self.sae_id,
            device=self.device
        )
        
        self.stored_activations = []
    
    def store_hook(self, act, hook):
        self.stored_activations.append(act.detach())
        return act  # Return unchanged to not affect the forward pass

    def get_activations(self, text):
        # Tokenize input
        tokens = self.model.to_tokens(text).to(self.device)
        
        # Clear previous activations
        self.stored_activations = []
        
        # Pass through GPT-2 and extract activations
        with torch.no_grad():
            self.model.run_with_hooks(tokens, fwd_hooks=[(self.sae_id, self.store_hook)])
        
        # Return the stored activations
        return self.sae.encode(self.stored_activations[0] if self.stored_activations else None)
    
    def get_activations_np(self, text):
        encoded_activations = self.get_activations(text)
        return encoded_activations.detach().squeeze().cpu().numpy()
        
def main():
    # Example usage
    interpreter = ModelInterpreter()
    text = "Let's activate some neurons to understand what the model is doing!"
    act_np = interpreter.get_activations_np(text)

    # X axis is the feature index, Y axis is the token position
    # Let's grab the mean of the activations for each feature
    feature_means = act_np.mean(axis=0)

    # Sort by mean activation and include feature index
    sorted_indices = sorted(enumerate(feature_means), key=lambda x: x[1], reverse=True)

    # Print the top 50 features
    print(sorted_indices[:50])

if __name__ == "__main__":
    main()