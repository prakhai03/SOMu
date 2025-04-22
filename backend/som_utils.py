import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib
# Set the backend to Agg for non-interactive environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from typing import Dict, List, Tuple, Optional
import json

class SOMAnalyzer:
    def __init__(self, data: pd.DataFrame, target_variable: Optional[str] = None, algorithm: str = 'MiniSom', **params):
        self.data = data
        self.target_variable = target_variable
        self.algorithm = algorithm.lower()  # Convert to lowercase for case-insensitive comparison
        self.params = params
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.som = None  # Initialize som as None
        
        try:
            # Prepare data
            if target_variable:
                self.X = self.data.drop(columns=[target_variable])
                self.y = self.data[target_variable]
                # Check if target is categorical
                self.is_categorical = pd.api.types.is_categorical_dtype(self.y) or pd.api.types.is_object_dtype(self.y)
                if self.is_categorical:
                    # Encode categorical target
                    self.y_encoded = self.label_encoder.fit_transform(self.y)
                else:
                    self.y_encoded = self.y
            else:
                self.X = self.data
                self.y = None
                self.y_encoded = None
                self.is_categorical = False
                
            self.X_scaled = self.scaler.fit_transform(self.X)
            
            # Initialize SOM based on algorithm
            self._initialize_som()
        except Exception as e:
            raise Exception(f"Error initializing SOMAnalyzer: {str(e)}")
        
    def _initialize_som(self):
        """Initialize the SOM with the given parameters."""
        try:
            input_len = self.X_scaled.shape[1]
            
            # Default parameters
            x_size = self.params.get('x', 10)
            y_size = self.params.get('y', 10)
            sigma = self.params.get('sigma', 1.0)
            learning_rate = self.params.get('learning_rate', 0.5)
            random_seed = self.params.get('random_seed', 42)
            
            if self.algorithm == 'minisom':
                self.som = MiniSom(
                    x=x_size,
                    y=y_size,
                    input_len=input_len,
                    sigma=sigma,
                    learning_rate=learning_rate,
                    random_seed=random_seed
                )
                # Train the SOM
                self.som.train(self.X_scaled, num_iteration=1000)
            # elif self.algorithm == 'sompY':
            #     # For SOMPY, we'll use MiniSom with adjusted parameters
            #     # SOMPY typically uses different default parameters
            #     self.som = MiniSom(
            #         x=x_size,
            #         y=y_size,
            #         input_len=input_len,
            #         sigma=sigma * 2,  # SOMPY typically uses larger sigma
            #         learning_rate=learning_rate * 0.5,  # SOMPY typically uses smaller learning rate
            #         random_seed=random_seed
            #     )
            #     # Train the SOM with SOMPY-like parameters
            #     self.som.train(self.X_scaled, num_iteration=2000)  # SOMPY typically uses more iterations
            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}. Supported algorithms are 'minisom' and 'sompY'")
        except Exception as e:
            raise Exception(f"Error initializing SOM: {str(e)}")
        
    def get_winning_neurons(self) -> pd.DataFrame:
        """Get winning neurons for each data point."""
        if self.som is None:
            raise Exception("SOM has not been initialized")
            
        try:
            winners = []
            for x in self.X_scaled:
                winner = self.som.winner(x)
                winners.append(winner)
            
            # Create DataFrame with results
            result_df = pd.DataFrame({
                'Neuron': [f"({w[0]},{w[1]})" for w in winners],
                'Data_Points': [self.X.iloc[i].values.tolist() for i in range(len(self.X))],
                'Row_Numbers': range(len(self.X))
            })
            
            # Group by neuron
            grouped = result_df.groupby('Neuron').agg({
                'Data_Points': list,
                'Row_Numbers': list
            }).reset_index()
            
            return grouped
        except Exception as e:
            raise Exception(f"Error getting winning neurons: {str(e)}")
    
    def get_u_matrix(self) -> str:
        """Generate U-Matrix visualization."""
        if self.som is None:
            raise Exception("SOM has not been initialized")
            
        try:
            plt.figure(figsize=(10, 10))
            plt.clf()  # Clear the current figure
            distance_map = self.som.distance_map().T
            mappable = plt.imshow(distance_map, cmap='bone_r')
            plt.colorbar(mappable)
            plt.title('U-Matrix')
            
            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()  # Close the figure to free memory
            
            return base64.b64encode(image_png).decode()
        except Exception as e:
            raise Exception(f"Error generating U-Matrix: {str(e)}")
    
    def get_quantization_error(self) -> float:
        """Calculate quantization error."""
        if self.som is None:
            raise Exception("SOM has not been initialized")
            
        try:
            return self.som.quantization_error(self.X_scaled)
        except Exception as e:
            raise Exception(f"Error calculating quantization error: {str(e)}")
    
    def get_topographic_error(self) -> float:
        """Calculate topographic error."""
        if self.som is None:
            raise Exception("SOM has not been initialized")
            
        try:
            return self.som.topographic_error(self.X_scaled)
        except Exception as e:
            raise Exception(f"Error calculating topographic error: {str(e)}")
    
    def get_cluster_purity(self) -> float:
        """Calculate cluster purity if target variable exists."""
        if self.som is None:
            raise Exception("SOM has not been initialized")
            
        if self.target_variable is None or not self.is_categorical:
            return None
            
        try:
            winners = np.array([self.som.winner(x) for x in self.X_scaled])
            clusters = [f"({w[0]},{w[1]})" for w in winners]
            
            purity = 0
            unique_clusters = np.unique(clusters)
            
            for cluster in unique_clusters:
                cluster_indices = np.where(clusters == cluster)[0]
                if len(cluster_indices) > 0:
                    cluster_labels = self.y.iloc[cluster_indices]
                    most_common = cluster_labels.mode()[0]
                    purity += np.sum(cluster_labels == most_common)
                    
            return purity / len(self.y)
        except Exception as e:
            raise Exception(f"Error calculating cluster purity: {str(e)}")
    
    def get_neuron_hit_map(self) -> str:
        """Generate neuron hit map visualization."""
        if self.som is None:
            raise Exception("SOM has not been initialized")
            
        try:
            plt.figure(figsize=(10, 10))
            plt.clf()  # Clear the current figure
            activation_map = self.som.activation_response(self.X_scaled).T
            mappable = plt.imshow(activation_map, cmap='Blues')
            plt.colorbar(mappable)
            plt.title('Neuron Hit Map')
            
            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()  # Close the figure to free memory
            
            return base64.b64encode(image_png).decode()
        except Exception as e:
            raise Exception(f"Error generating neuron hit map: {str(e)}")
    
    def predict_clusters(self) -> Dict:
        """Predict clusters using BMUs."""
        if self.som is None:
            raise Exception("SOM has not been initialized")
            
        try:
            winners = np.array([self.som.winner(x) for x in self.X_scaled])
            clusters = [f"({w[0]},{w[1]})" for w in winners]
            
            result = {
                'cluster_assignments': clusters,
                'unique_clusters': list(set(clusters))
            }
            
            return result
        except Exception as e:
            raise Exception(f"Error predicting clusters: {str(e)}")
    
    def get_confusion_matrix(self) -> Optional[str]:
        """Generate confusion matrix if target variable exists and is categorical."""
        if self.som is None:
            raise Exception("SOM has not been initialized")
            
        if self.target_variable is None or not self.is_categorical:
            return None
            
        try:
            winners = np.array([self.som.winner(x) for x in self.X_scaled])
            clusters = [f"({w[0]},{w[1]})" for w in winners]
            
            plt.figure(figsize=(10, 10))
            plt.clf()  # Clear the current figure
            
            # Use encoded labels for confusion matrix
            cm = confusion_matrix(self.y_encoded, clusters)
            mappable = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Clusters')
            plt.ylabel('True Labels')
            
            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()  # Close the figure to free memory
            
            return base64.b64encode(image_png).decode()
        except Exception as e:
            raise Exception(f"Error generating confusion matrix: {str(e)}")
    
    def analyze(self) -> Dict:
        """Perform complete SOM analysis."""
        if self.som is None:
            raise Exception("SOM has not been initialized")
            
        try:
            return {
                'winning_neurons': self.get_winning_neurons().to_dict('records'),
                'u_matrix': self.get_u_matrix(),
                'quantization_error': self.get_quantization_error(),
                'topographic_error': self.get_topographic_error(),
                'cluster_purity': self.get_cluster_purity(),
                'neuron_hit_map': self.get_neuron_hit_map(),
                'clusters': self.predict_clusters(),
                'confusion_matrix': self.get_confusion_matrix()
            }
        except Exception as e:
            raise Exception(f"Error during analysis: {str(e)}") 