import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from swotted import swottedModule, swottedTrainer
from swotted.loss_metrics import *
from swotted.utils import Subset, success_rate

from src.trajectoryclusteringanalysis.plotting import *

class MultidimensionalAnalyzer:

    def __init__(self, data, index_col='patient_id', time_col='time', event_col='care_event'):
        self.K = None  #: number of individuals
        self.N = None  #: number of events
        self.T = None  #: length of time points
        self.X = None  #: tensor of shape (K, N, T) with K individuals, N events, and T time points
        self.data = data
        self.index_col = index_col
        self.time_col = time_col
        self.event_col = event_col
        self.model = None  #: SWoTTeD model
    
    def has_time_event_structure(self):
        """

        """
        if not isinstance(self.data, pd.DataFrame):
            return False
        if self.index_col not in self.data.columns or self.time_col not in self.data.columns or self.event_col not in self.data.columns:
            return False
        # if not (np.issubdtype(self.data[time_col].dtype, np.number) and np.issubdtype(self.data[event_col].dtype, np.object)):
        #     return False
        return True

    def transform_time_event_structure_to_tensor(self):
        """

        """
        unique_individuals = self.data[self.index_col].unique()
        unique_events = self.data[self.event_col].unique()
        unique_time_points = self.data[self.time_col].unique()

        patient_to_index = {patient: idx for idx, patient in enumerate(unique_individuals)}

        self.K = len(unique_individuals)
        self.N = len(unique_events)
        self.T = len(unique_time_points)
        
        tensor = np.zeros((self.K, self.N, self.T), dtype=int)
        for _, row in self.data.iterrows():
            patient_idx = patient_to_index[row[self.index_col]]
            event_idx = np.where(unique_events == row[self.event_col])[0][0]
            time_idx = np.where(unique_time_points == row[self.time_col])[0][0]
            tensor[patient_idx, event_idx, time_idx] = 1

        self.X = torch.tensor(tensor, dtype=torch.float32)

    def get_tensor_shape(self):
        """
        Returns the shape of the tensor.
        """
        if self.X is not None:
            return self.X.shape
        else:
            raise ValueError("Tensor has not been initialized. Please call time_event_structure_to_tensor first.")
    
    def get_tensor(self):
        """
        Returns the tensor.
        """
        if self.X is not None:
            return self.X
        else:
            raise ValueError("Tensor has not been initialized. Please call time_event_structure_to_tensor first.")
        
    def fit_swotted_decomposition(self, tensor, rank, time_window_length, reg_term_ns=0.5, reg_term_s=0.5, metric='Bernoulli', learning_rate=1e-2, n_epochs=100, ):
        """

        """
        params = {}
        params['model']={}
        params['model']['non_succession']=reg_term_ns
        params['model']['sparsity']=reg_term_s
        params['model']['rank']=rank
        params['model']['twl']=time_window_length
        params['model']['N']=self.N
        params['model']['metric']=metric

        #some additional parameters of the trainer
        params['training']={}
        params['training']['lr']=learning_rate

        #some additional parameters for the projection (decomposition on new sequences)
        params['predict']={}
        params['predict']['nepochs']=n_epochs
        params['predict']['lr']=learning_rate

        config=OmegaConf.create(params)

        # define the model
        self.model = swottedModule(config)
        # train the model
        trainer = swottedTrainer(max_epochs=n_epochs, accelerator='cpu', devices=1, logger=None, enable_progress_bar=True)

        train_loader = DataLoader(
        Subset(tensor, np.arange(len(self.X))),
            batch_size=15,
            # num_workers=31,
            shuffle=False,
            collate_fn=lambda x: x
        )

        trainer.fit(model=self.model, train_dataloaders=train_loader)
        self.W = self.model(self.X)

    def get_decomposition_results(self, labels, id):  
        """
        Returns the decomposition result.
        """
        if hasattr(self, 'model') and self.model is not None:
            
            print(f"Decomposed into {len(self.W)} pathways with rank {self.model.rank} and time window length {self.model.twl}")
            Ph = self.model.Ph.detach().clone().requires_grad_(True)
            rPh, rW = self.model.reorderPhenotypes(gen_pheno=Ph, Wk=None, tw=self.model.twl)

            X_pred = []
            for p in range(self.K):
                X_pred.append(self.model.model.reconstruct(self.W[p], Ph)) 
            X_pred = torch.stack(X_pred)
            
            fit_metric = 1 - (torch.norm(self.X - X_pred) / torch.norm(self.X))
            print(f"FIT metric for the entire dataset: {fit_metric.item():.4f}")

            print(f"success_rate :{success_rate(self.X, X_pred)}")

            plot_discovered_phenotypes(rPh, self.model.rank, labels)
            plot_discovered_pathways(rW, id)
            plot_reconstructed_matrix(X_pred, id, labels)

        else:
            raise ValueError("Decomposition has not been performed. Please call fit_swotted_decomposition first.")
        
    def to_phenotype_intensity(self, scaler=MinMaxScaler()):
        """
        Converts the decomposition result to phenotype intensity.
        """
        if hasattr(self, 'model') and self.W is not None:
            W_all = torch.stack(self.W)
            phenotype_intensity = W_all.sum(axis=2).detach().numpy()

            phenotype_summary = pd.DataFrame(phenotype_intensity, columns=[f'Phenotype_{i+1}' for i in range(phenotype_intensity.shape[1])])
            phenotype_summary = pd.DataFrame(scaler.fit_transform(phenotype_summary), columns=phenotype_summary.columns)
            phenotype_summary[self.index_col] = self.data[self.index_col].unique()
            
            return phenotype_summary
        else:
            raise ValueError("Decomposition has not been performed. Please call fit_swotted_decomposition first.")

def main():
    # Example usage
    data = pd.read_excel('data/multidimensional_data.xlsx')
    print("Data shape:", data.shape)

    analyzer = MultidimensionalAnalyzer(data, index_col='ID_PATIENT', time_col='Months_Since_First_Events', event_col='Lib_traitement')
    if analyzer.has_time_event_structure():
        print("Data has the required time-event structure.")
        analyzer.transform_time_event_structure_to_tensor()
        print("Tensor shape:", analyzer.get_tensor_shape())
        tensor = analyzer.get_tensor()
        
        # Example decomposition
        rank = 3
        time_window_length = 3
        reg_term_ns = 0.5
        reg_term_s = 0.5
        metric = 'Bernoulli'
        learning_rate = 1e-2
        n_epochs = 10

        
        analyzer.fit_swotted_decomposition(tensor, rank, time_window_length, reg_term_ns, reg_term_s, metric, learning_rate, n_epochs)
        analyzer.get_decomposition_result()
        

        # Plotting the discovered phenotypes
        id = 100
        plot_input_matrix(tensor, id, analyzer.data['Lib_traitement'].unique())
        
        # plot_discovered_phenotypes(analyzer.model, rank)

    
    else:
        print("Data does not have the required time-event structure.")

if __name__ == "__main__":
    main()
