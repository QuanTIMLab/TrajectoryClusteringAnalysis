import numpy as np
import pandas as pd
import time
import copy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from swotted import swottedModule, swottedTrainer
from swotted.loss_metrics import *
from swotted.utils import Subset, success_rate
try:
    from lightning.pytorch.callbacks import Callback
    from lightning.pytorch.callbacks import EarlyStopping
except ImportError:
    from pytorch_lightning.callbacks import Callback
    from pytorch_lightning.callbacks import EarlyStopping

from trajectoryclusteringanalysis.plotting import *


def identity_collate(batch):
    return batch


class FitMetricCallback(Callback):
    """
    Lightweight epoch monitoring.

    By default, prints training loss each epoch with negligible overhead.
    Optional full FIT metric can be enabled every N epochs (costly).
    """

    def __init__(
        self,
        analyzer,
        fit_metric_every_n_epochs=0,
        fit_metric_eval_max_patients=None,
        restore_best_fit_checkpoint=True,
        early_stopping_monitor='train_loss_total',
        early_stopping_min_delta=1e-4,
        early_stopping_mode='min',
        early_stopping_patience=10,
    ):
        self.analyzer = analyzer
        self.fit_metric_every_n_epochs = int(fit_metric_every_n_epochs)
        self.fit_metric_eval_max_patients = fit_metric_eval_max_patients
        self.restore_best_fit_checkpoint = bool(restore_best_fit_checkpoint)
        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.early_stopping_mode = early_stopping_mode
        self.early_stopping_patience = early_stopping_patience
        self._best_early_stop_score = None
        self._no_improve_count = 0
        self._epoch_start_time = None
        self._fit_eval_indices = None
        self._fit_eval_tensor = None
        self._fit_eval_norm = None
        self._fit_full_norm = None
        self._best_fit_metric_for_checkpoint = None
        self._best_fit_epoch_for_checkpoint = None
        self._best_model_state_dict = None

        self._prepare_fit_eval_state()

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start_time = time.perf_counter()

    def _get_train_losses(self, trainer):
        metrics = trainer.callback_metrics
        train_loss_ph = metrics.get('train_loss_Ph', None)
        train_loss_w = metrics.get('train_loss_W', None)
        return train_loss_ph, train_loss_w

    def _prepare_fit_eval_state(self):
        if self.analyzer.X is None or self.analyzer.K is None:
            return

        self._fit_full_norm = torch.norm(self.analyzer.X)

        if self.fit_metric_eval_max_patients is None:
            self._fit_eval_indices = None
            self._fit_eval_tensor = self.analyzer.X
            self._fit_eval_norm = self._fit_full_norm
            return

        n_patients = int(self.analyzer.K)
        n_eval = max(1, min(int(self.fit_metric_eval_max_patients), n_patients))
        if n_eval >= n_patients:
            self._fit_eval_indices = None
            self._fit_eval_tensor = self.analyzer.X
            self._fit_eval_norm = self._fit_full_norm
            return

        # Evenly spaced subset keeps coverage across the cohort.
        self._fit_eval_indices = torch.linspace(0, n_patients - 1, steps=n_eval).long()
        self._fit_eval_tensor = self.analyzer.X[self._fit_eval_indices]
        self._fit_eval_norm = torch.norm(self._fit_eval_tensor)

    def _compute_fit_metric(self, pl_module, use_full_dataset=False):
        if self.analyzer.X is None or self.analyzer.K is None:
            return None

        x_target = self.analyzer.X if use_full_dataset else self._fit_eval_tensor
        if x_target is None:
            x_target = self.analyzer.X

        denom = self._fit_full_norm if use_full_dataset else self._fit_eval_norm
        if denom is None:
            denom = torch.norm(x_target)

        with torch.no_grad():
            w_epoch = pl_module(x_target)
            ph_epoch = pl_module.Ph.detach()

            x_pred = []
            n_items = len(w_epoch)
            for p in range(n_items):
                x_pred.append(pl_module.model.reconstruct(w_epoch[p], ph_epoch))
            x_pred = torch.stack(x_pred)

            if denom.item() == 0:
                return 0.0
            return float((1 - (torch.norm(x_target - x_pred) / denom)).item())

    def on_train_epoch_end(self, trainer, pl_module):

        epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs
        epoch_duration = None
        if self._epoch_start_time is not None:
            epoch_duration = time.perf_counter() - self._epoch_start_time

        train_loss_ph, train_loss_w = self._get_train_losses(trainer)
        loss_ph_value = None
        loss_w_value = None

        if train_loss_ph is not None:
            try:
                loss_ph_value = float(train_loss_ph.detach().cpu().item())
            except Exception:
                loss_ph_value = float(train_loss_ph)

        if train_loss_w is not None:
            try:
                loss_w_value = float(train_loss_w.detach().cpu().item())
            except Exception:
                loss_w_value = float(train_loss_w)

        loss_total_value = None
        if loss_ph_value is not None and loss_w_value is not None:
            loss_total_value = loss_ph_value + loss_w_value
            trainer.callback_metrics['train_loss_total'] = torch.tensor(loss_total_value)

        fit_metric_value = None
        checkpoint_message = None
        should_compute_fit = (
            self.fit_metric_every_n_epochs > 0
            and (epoch == 1 or (epoch % self.fit_metric_every_n_epochs) == 0)
            and self.analyzer.X is not None
            and self.analyzer.K is not None
        )

        if should_compute_fit:
            fit_metric_value = self._compute_fit_metric(pl_module, use_full_dataset=False)
            trainer.callback_metrics['fit_metric'] = torch.tensor(fit_metric_value)

            if (
                self._best_fit_metric_for_checkpoint is None
                or fit_metric_value > self._best_fit_metric_for_checkpoint
            ):
                self._best_fit_metric_for_checkpoint = float(fit_metric_value)
                self._best_fit_epoch_for_checkpoint = int(epoch)
                # Keep the best model in memory to restore it at fit end.
                self._best_model_state_dict = copy.deepcopy(pl_module.state_dict())

                if hasattr(self.analyzer, 'training_history') and isinstance(self.analyzer.training_history, dict):
                    self.analyzer.training_history['best_fit_metric'] = self._best_fit_metric_for_checkpoint
                    self.analyzer.training_history['best_fit_metric_epoch'] = self._best_fit_epoch_for_checkpoint

                checkpoint_message = (
                    f"    checkpoint: New best FIT metric = {self._best_fit_metric_for_checkpoint:.4f} "
                    f"at epoch {self._best_fit_epoch_for_checkpoint}."
                )

        if hasattr(self.analyzer, 'training_history') and isinstance(self.analyzer.training_history, dict):
            self.analyzer.training_history['epoch'].append(epoch)
            self.analyzer.training_history['train_loss_Ph'].append(loss_ph_value)
            self.analyzer.training_history['train_loss_W'].append(loss_w_value)
            self.analyzer.training_history['train_loss_total'].append(loss_total_value)
            self.analyzer.training_history['fit_metric'].append(fit_metric_value)

            if fit_metric_value is not None:
                self.analyzer.training_history['fit_metric_points_epoch'].append(epoch)
                self.analyzer.training_history['fit_metric_points_value'].append(fit_metric_value)

        parts = [f"Epoch {epoch}/{max_epochs} :"]
        if loss_ph_value is not None:
            parts.append(f"train_loss_Ph: {loss_ph_value:.6f}")
        if loss_w_value is not None:
            parts.append(f"train_loss_W: {loss_w_value:.6f}")
        if loss_total_value is not None:
            parts.append(f"train_loss_total: {loss_total_value:.6f}")
        if epoch_duration is not None:
            parts.append(f"duration: {epoch_duration:.2f}s")
        if fit_metric_value is not None:
            parts.append(f"FIT metric: {fit_metric_value:.4f}")

        # Keep the first metric directly after ':' for easier reading.
        out = " - ".join(parts)
        # for marker in ["train_loss_Ph", "train_loss_W", "duration", "FIT metric"]:
        #     out = out.replace(f" - {marker}", f" {marker}", 1)
        print(out)

        if self.early_stopping_patience is None or self.early_stopping_patience <= 0:
            if checkpoint_message is not None:
                print(checkpoint_message)
            return

        # For FIT-based monitoring, only run early-stopping checks when a new
        # FIT value is actually computed. Otherwise, callback_metrics may keep
        # the previous FIT value and artificially consume patience every epoch.
        if self.early_stopping_monitor == 'fit_metric':
            monitored_value = fit_metric_value
            if monitored_value is None:
                if checkpoint_message is not None:
                    print(checkpoint_message)
                return
        else:
            monitored_value = trainer.callback_metrics.get(self.early_stopping_monitor, None)
            if monitored_value is None:
                if checkpoint_message is not None:
                    print(checkpoint_message)
                return

        try:
            monitored_value = float(monitored_value.detach().cpu().item())
        except Exception:
            monitored_value = float(monitored_value)

        if self._best_early_stop_score is None:
            self._best_early_stop_score = monitored_value
            print(
                f"    early stopping: Metric {self.early_stopping_monitor} initialized at "
                f"{self._best_early_stop_score:.3f}."
            )
            if checkpoint_message is not None:
                print(checkpoint_message)
            return

        if self.early_stopping_mode == 'max':
            improvement = monitored_value - self._best_early_stop_score
        else:
            improvement = self._best_early_stop_score - monitored_value

        if improvement >= self.early_stopping_min_delta:
            self._best_early_stop_score = monitored_value
            self._no_improve_count = 0
            print(
                f"    early stopping: Metric {self.early_stopping_monitor} improved by "
                f"{improvement:.3f} >= min_delta = {self.early_stopping_min_delta}. "
                f"New best score: {self._best_early_stop_score:.3f}"
            )
        else:
            self._no_improve_count += 1
            if self.early_stopping_patience is not None and self._no_improve_count >= self.early_stopping_patience:
                print(
                    f"    early stopping: No improvement for {self._no_improve_count} checks on "
                    f"{self.early_stopping_monitor}. Stopping training."
                )
                trainer.should_stop = True

        if checkpoint_message is not None:
            print(checkpoint_message)

    def on_fit_end(self, trainer, pl_module):
        if self.restore_best_fit_checkpoint and self._best_model_state_dict is not None:
            pl_module.load_state_dict(self._best_model_state_dict)
            print(
                f"\n -> Best-FIT checkpoint restored from epoch {self._best_fit_epoch_for_checkpoint} "
                f"(FIT metric: {self._best_fit_metric_for_checkpoint:.4f})."
            )

        train_elapsed_s = getattr(self.analyzer, 'training_duration_s', None)
        if train_elapsed_s is not None:
            print(f" -> Training finished (time: {train_elapsed_s:.2f}s)")
        else:
            print(" -> Training complete.")

        print("\n -> Computing FIT metric on full tensor ...")
        fit_start_time = time.perf_counter()
        final_fit = self._compute_fit_metric(pl_module, use_full_dataset=True)
        fit_elapsed_s = time.perf_counter() - fit_start_time
        if final_fit is not None:
            fit_source_label = (
                "checkpoint saved"
                if self.restore_best_fit_checkpoint and self._best_model_state_dict is not None
                else "last epoch model"
            )
            print(
                f" -> Global FIT metric computed from {fit_source_label}: {final_fit:.4f} "
                f"(time: {fit_elapsed_s:.2f}s)"
            )
            if hasattr(self.analyzer, 'training_history') and isinstance(self.analyzer.training_history, dict):
                self.analyzer.training_history['final_fit_metric'] = final_fit

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
        Checks if the data has a valid time-event structure.
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
        Transforms the time-event structure to a tensor.
        """
        unique_individuals = self.data[self.index_col].unique()
        unique_events = np.sort(self.data[self.event_col].unique())
        unique_time_points = np.sort(self.data[self.time_col].unique())

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
            
        
    def fit_swotted_decomposition(
        self,
        tensor,
        rank,
        time_window_length,
        reg_term_ns=0.5,
        reg_term_s=0.5,
        metric='Bernoulli',
        learning_rate=1e-2,
        n_epochs=100,
        fit_metric_every_n_epochs=0,
        fit_metric_eval_max_patients=None,
        restore_best_fit_checkpoint=True,
        num_workers=0,
        early_stopping_patience=10,
        early_stopping_min_delta=1e-4,
        early_stopping_monitor='train_loss_total',
    ):
        """
        Fits the SWoTTeD decomposition model to the tensor.

        Parameters:
        - tensor: the input tensor of shape (K, N, T)       
        - rank: the rank of the decomposition (number of phenotypes)
        - time_window_length: the length of the time window for the non-succession regularization
        - reg_term_ns: the regularization term for non-succession (default: 0.5)
        - reg_term_s: the regularization term for sparsity (default: 0.5)
        - metric: the metric to use for the reconstruction loss (default: 'Bernoulli')
        - learning_rate: the learning rate for training (default: 1e-2) 
        - n_epochs: the number of epochs for training (default: 100)

        Returns:
        - model: the fitted SWoTTeD model
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

        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")

        # define the model
        self.model = swottedModule(config)
        self.training_history = {
            'epoch': [],
            'train_loss_Ph': [],
            'train_loss_W': [],
            'train_loss_total': [],
            'fit_metric': [],
            'fit_metric_points_epoch': [],
            'fit_metric_points_value': [],
            'best_fit_metric': None,
            'best_fit_metric_epoch': None,
            'final_fit_metric': None,
        }

        early_stopping_mode = 'max' if early_stopping_monitor == 'fit_metric' else 'min'

        callbacks = [
            FitMetricCallback(
                self,
                fit_metric_every_n_epochs=fit_metric_every_n_epochs,
                fit_metric_eval_max_patients=fit_metric_eval_max_patients,
                restore_best_fit_checkpoint=restore_best_fit_checkpoint,
                early_stopping_monitor=early_stopping_monitor,
                early_stopping_min_delta=early_stopping_min_delta,
                early_stopping_mode=early_stopping_mode,
                early_stopping_patience=early_stopping_patience,
            )
        ]
        if (
            early_stopping_patience is not None
            and early_stopping_patience > 0
            and early_stopping_monitor != 'fit_metric'
        ):
            callbacks.append(
                EarlyStopping(
                    monitor=early_stopping_monitor,
                    min_delta=early_stopping_min_delta,
                    patience=early_stopping_patience,
                    mode=early_stopping_mode,
                    strict=False,
                    verbose=False,
                )
            )

        # train the model
        trainer = swottedTrainer(
            max_epochs=n_epochs,
            accelerator='cpu',
            devices='auto',
            logger=None,
            enable_progress_bar=True,
            callbacks=callbacks,
        )

        train_loader = DataLoader(
            Subset(tensor, np.arange(len(self.X))),
            batch_size=15,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=identity_collate,
            persistent_workers=True if num_workers > 0 else False,
        )

        print("\n -> Starting training")
        train_start_time = time.perf_counter()
        trainer.fit(model=self.model, train_dataloaders=train_loader)
        train_elapsed_s = time.perf_counter() - train_start_time
        self.training_duration_s = train_elapsed_s
        
        print("\n -> Computing final decomposition on full tensor ...")
        decomp_start_time = time.perf_counter()
        self.W = self.model(self.X)
        decomp_elapsed_s = time.perf_counter() - decomp_start_time
        print(f" -> Decomposition finished (time: {decomp_elapsed_s:.2f}s)")

    def get_decomposition_results(self, labels, id, time_unit="Months"):  
        """
        Returns the decomposition result.

        Parameters:
        - labels: the labels of the events
        - id: the ID of the individual
        - time_unit: the unit of time for plotting (default: "Months") 

        Returns:
        - The plot of the discovered phenotypes
        - The plot of the discovered pathways
        - The plot of the reconstructed matrix
        - The plot of the input matrix

        """
        if hasattr(self, 'model') and self.model is not None:
            
            print(f"Decomposed into {len(self.W)} pathways with rank {self.model.rank} and time window length {self.model.twl}")
            Ph = self.model.Ph.detach().clone().requires_grad_(True)
            rPh, rW = self.model.reorderPhenotypes(gen_pheno=Ph, Wk=None, tw=self.model.twl)

            X_pred = []
            for p in range(self.K):
                X_pred.append(self.model.model.reconstruct(self.W[p], Ph)) 
            X_pred = torch.stack(X_pred)
            
            print(f"Success rate for the entire dataset: {success_rate(self.X, X_pred):.4f}")

            plot_discovered_phenotypes(rPh, self.model.rank, labels, time_unit=time_unit)
            plot_discovered_pathways(rW, id, title=f"Discovered Pathways of individual {id}", time_unit=time_unit)
            plot_reconstructed_matrix(X_pred, id, labels, title=f"Reconstructed Matrix of individual {id}", time_unit=time_unit)
            plot_input_matrix(self.X, id, labels, title=f"Input Matrix of individual {id}", time_unit=time_unit)

        else:
            raise ValueError("Decomposition has not been performed. Please call fit_swotted_decomposition first.")
        
    def compute_reconstruction(self):
        """
        Returns X_pred after fit_swotted_decomposition.
        """
        if not (hasattr(self, 'model') and self.model is not None):
            raise ValueError("Decomposition has not been performed. Please call fit_swotted_decomposition first.")
    
        Ph = self.model.Ph.detach().clone().requires_grad_(True)
        X_pred = torch.stack([self.model.model.reconstruct(self.W[p], Ph)
                              for p in range(self.K)])
        return X_pred
        
    def to_phenotype_intensity(self, scaler=MinMaxScaler()):
        """
        Converts the decomposition result to phenotype intensity.

        Parameters:
        - scaler: the scaler to use for normalizing the phenotype intensity (default: MinMaxScaler)

        Returns:
        - A DataFrame containing the phenotype intensity for each individual.
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
        plot_input_matrix(tensor, id, analyzer.data['Lib_traitement'].unique(), time_unit="Days")
        
        # plot_discovered_phenotypes(analyzer.model, rank)

    
    else:
        print("Data does not have the required time-event structure.")

if __name__ == "__main__":
    main()
