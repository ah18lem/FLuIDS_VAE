import os

# This guide can only be run with the TensorFlow backend.
os.environ["KERAS_BACKEND"] = "tensorflow"
import flwr as fl
from sklearn.calibration import LabelEncoder
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import keras
import numpy as np
from sklearn.metrics import  confusion_matrix
import flwr as fl
from flwr.common import  Scalar
from typing import  Dict, List, Optional, Tuple
from sklearn.metrics import classification_report
import pandas as pd
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
import client
import model
import functions
import constantes
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# Enable GPU growth in your main process



train_data_server, train_labels_server, test_data, test_labels, client_data, client_labels,combined_data,combined_labels=functions.split_train_server_clients()
balancedDatasets, FakeLabels=functions.split_balanced_datasets_clients(client_data,client_labels)
inputdim= train_data_server.shape[1]
test_data_tensor = tf.convert_to_tensor(test_data, dtype=tf.float32)

test_labels_one_hot_encoded=functions.one_hot_encode_column(test_labels)


for i in range(len(FakeLabels)):
    # Convert the fake_label to a DataFrame
    fake_label_df = pd.DataFrame(FakeLabels[i])
    
    # Perform one-hot encoding
    encoded_label = functions.one_hot_encode_column(fake_label_df)
    print(encoded_label.shape)
    
    # Replace the item in the FakeLabels list with the one-hot encoded DataFrame
    FakeLabels[i] = pd.DataFrame(encoded_label)

if(constantes.MULTICLASS):
        # Get the actual class names before encoding
        class_names = np.unique(test_labels)
        # Initialize LabelEncoder
        label_encoder = LabelEncoder()
        # Fit LabelEncoder to your string labels
        label_encoder.fit(test_labels)
        # Transform string labels to class indices
        test_labels_encoded = label_encoder.transform(test_labels)
else:
        # For binary or other cases
        class_names = ['Class 0', 'Class 1']  # Provide class names according to your dataset
        test_labels_encoded=test_labels

class modifiedFedAVG(fl.server.strategy.FedAvg):
    
    def configure_fit(
        self, server_round:int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        # Au lieu de retourner aux clients les parametres aggregated , on entraine dabord
        net =model.AutoencoderWithClassifier(inputdim,isServer=True, vae=constantes.VAE)
        net.model.set_weights(parameters_to_ndarrays(parameters))     
        net.train(train_data_server, train_labels_server, constantes.EPOCHS_SERVEUR)
        updated_param = net.get_parameters()
        fit_ins = FitIns(ndarrays_to_parameters(updated_param), config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]
    
net =model.AutoencoderWithClassifier(inputdim,isServer=True, vae=constantes.VAE)
net.model.summary()
def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar],
             test_data: np.ndarray, test_labels: np.ndarray) -> Optional[Tuple]:
    
    net.model.set_weights(parameters)
    predictions, _ = net.call(test_data_tensor)
    print(predictions.shape)
    print(test_labels.shape)
    classification_loss = tf.keras.losses.categorical_crossentropy(test_labels_one_hot_encoded, predictions)
    mean_classification_loss = tf.reduce_mean(classification_loss)
    predictions = np.argmax(predictions, axis=1)

    print("Round number : " ,server_round)
    report=classification_report(test_labels_encoded, predictions, target_names=class_names,zero_division=1) 
    # Print confusion matrix
    cm = confusion_matrix(test_labels_encoded, predictions)
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, columns=class_names, index=class_names))  # Print confusion matrix with class names
    print ("classification report")
    print(report)
    return mean_classification_loss, {
        
    }

strategy = modifiedFedAVG(
 fraction_fit=constantes.FRACTION_FIT,  
    fraction_evaluate=constantes.FRACTION_EVALUATE,  
    min_fit_clients=constantes.MIN_FIT_CLIENTS,  
    min_evaluate_clients=constantes.MIN_EVALUATE_CLIENTS, 
    min_available_clients=constantes.MIN_AVAILABLE_CLIENTS,
   evaluate_fn=lambda server_round, parameters, config: evaluate(server_round, parameters, config, test_data, test_labels),
  
)
client_resources = {"num_cpus": 1, "num_gpus": 0.0}


fl.simulation.start_simulation(
    client_fn=client.get_client_fn(inputdim,balancedDatasets,FakeLabels,test_data),
    num_clients=constantes.NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=constantes.NUM_ROUNDS),
    strategy=strategy,
    client_resources=client_resources,


)
