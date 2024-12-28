from pathlib import Path
from typing import List, Set
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import numpy as np

class DocumentAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 768, encoding_dim: int = 32) -> None:
        """
        Sparse autoencoder for document embedding and tag generation.
        
        Args:
            input_dim (int): Dimension of input embeddings (BERT default)
            encoding_dim (int): Dimension of the sparse encoding
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim),
            nn.Sigmoid()  # Forces sparsity
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class NotionTagger:
    def __init__(self, num_clusters: int = 20) -> None:
        """
        Initialize the tagger with BERT, autoencoder, and clustering.
        
        Args:
            num_clusters (int): Number of tag clusters to generate
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
        self.autoencoder = DocumentAutoencoder().to(self.device)
        self.kmeans = KMeans(n_clusters=num_clusters)
        self.base_tags = list({  # Convert to list for indexed access
            '[[Artificial Intelligence]]', '[[Computer Science]]', '[[writing]]',
            '[[book]]', '[[boot.dev]]', '[[general]]', '[[Internet Subculture]]',
            '[[Journal]]', '[[Parasocial Grief]]', '[[Personal Writing]]',
            '[[Personal]]', '[[Scraping]]', '[[türkçe]]', '[[work]]', '[[yzf]]',
            '[[Aiden]]'
        })

    def train_autoencoder(self, documents: List[str], epochs: int = 10) -> None:
        """
        Train the autoencoder on the document corpus.
        
        Args:
            documents (List[str]): List of document contents
            epochs (int): Number of training epochs
        """
        self.autoencoder.train()
        optimizer = torch.optim.Adam(self.autoencoder.parameters())
        criterion = nn.MSELoss()
        
        embeddings = []
        for doc in documents:
            emb = self.get_document_embedding(doc)
            embeddings.append(emb)
        
        embeddings_tensor = torch.stack(embeddings).to(self.device)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            encoded, decoded = self.autoencoder(embeddings_tensor)
            loss = criterion(decoded, embeddings_tensor)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
            
        # Train KMeans on encoded representations
        self.autoencoder.eval()
        with torch.no_grad():
            encoded_docs, _ = self.autoencoder(embeddings_tensor)
            encoded_docs = encoded_docs.cpu().numpy()
        self.kmeans.fit(encoded_docs)

    def get_document_embedding(self, text: str) -> torch.Tensor:
        """
        Get BERT embedding for a document.
        
        Args:
            text (str): Document text
            
        Returns:
            torch.Tensor: Document embedding
        """
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()

    def analyze_document(self, content: str) -> Set[str]:
        """
        Analyze document content using autoencoder and clustering.
        
        Args:
            content (str): Document content
            
        Returns:
            Set[str]: Generated tags
        """
        self.autoencoder.eval()
        with torch.no_grad():
            embedding = self.get_document_embedding(content)
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            encoded, _ = self.autoencoder(embedding)
            encoded = encoded.cpu().numpy()
            
        cluster = self.kmeans.predict(encoded)[0]
        
        tags = set()
        encoding_threshold = 0.5
        # Handle tensor dimensions properly
        active_neurons = np.where(encoded[0] > encoding_threshold)[0]
        
        for neuron in active_neurons:
            if neuron < len(self.base_tags):
                tags.add(self.base_tags[neuron])
        
        tags.add(f"[[Cluster_{cluster}]]")
        return tags

    def update_file_with_tags(self, file_path: Path) -> None:
        """
        Updates file with autoencoder-generated tags.
        
        Args:
            file_path (Path): Path to the file to update
        """
        try:
            content = file_path.read_text(encoding='utf-8')
            
            if content.startswith('Tags:'):
                return
                
            tags = self.analyze_document(content)
            tags_line = f"Tags: {' '.join(sorted(tags))}\n\n"
            
            file_path.write_text(tags_line + content, encoding='utf-8')
            print(f"Updated {file_path} with tags: {' '.join(sorted(tags))}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

def main() -> None:
    """
    Process markdown files using autoencoder-based tagging.
    """
    tagger = NotionTagger()
    current_dir = Path('.')
    md_files = list(current_dir.glob('*.md'))
    
    # Train autoencoder on all documents first
    documents = []
    for file_path in md_files:
        try:
            content = file_path.read_text(encoding='utf-8')
            documents.append(content)
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
    
    print("Training autoencoder...")
    tagger.train_autoencoder(documents)
    
    print("Processing files...")
    for file_path in md_files:
        tagger.update_file_with_tags(file_path)

if __name__ == "__main__":
    main() 