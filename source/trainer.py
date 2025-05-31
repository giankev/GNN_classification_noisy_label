import logging
import os
from datetime import datetime
from typing import List, Dict

import pandas as pd
import numpy as np
import torch

from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def warm_up_lr(epoch, num_epoch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params['lr'] = (epoch+1)**3 * init_lr / num_epoch_warm_up**3

class ModelTrainer:
    def __init__(self, config: 'ModelConfig', device: str):
        self.config = config
        self.device = device
        self.models: List[str] = []
        self.pretrain_models: List[str] = []
        self.best_f1_scores = []

        self._setup_directories()
        self._setup_logging()
        self.criterion = torch.nn.CrossEntropyLoss()

    def predict(self, model, device, loader):
        model.eval()
        y_pred = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                z, mu, logvar, class_logits = model(data.x, data.edge_index, data.edge_attr, data.batch, eps=0.0)
                pred = class_logits.argmax(dim=1)
                y_pred.extend(pred.tolist())
        return y_pred

    def _setup_directories(self):
        self.output_dir = '../output'
        os.makedirs(self.output_dir, exist_ok=True)
        for d in ['file_checkpoints', 'best', 'file_log']:
            os.makedirs(os.path.join(self.output_dir, d), exist_ok=True)


    def _setup_logging(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = os.path.join(self.output_dir, 'file_log', f'training_{self.config.folder_name}_{timestamp}.log')
    
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
    
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_filename, mode='w'),
                logging.StreamHandler()
            ]
        )
        
        logging.info("Logger created.")

    def load_pretrained(self):
        if self.config.pretrain_paths is not None:
            path = self.config.pretrain_paths
        if path.endswith('.pth'):
            self.pretrain_models = [path]
        else:
            with open(path, 'r') as f:
                self.pretrain_models = [line.strip() for line in f if line.strip()]

    def evaluate_model(self, model: torch.nn.Module, data_loader: DataLoader) -> Dict[str, float]:
        model.eval()
        total_loss, total_samples = 0.0, 0
        all_preds, all_labels = [], []
    
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                _, _, _, logits = model(data.x, data.edge_index, data.edge_attr, data.batch)
                loss = self.criterion(logits, data.y)
                preds = logits.argmax(dim=1).cpu().numpy()
                labels = data.y.cpu().numpy()
    
                all_preds.extend(preds)
                all_labels.extend(labels)
    
                batch_size = data.y.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
    
        avg_loss = total_loss / total_samples
        f1 = f1_score(all_labels, all_preds, average='weighted')
        acc = accuracy_score(all_labels, all_preds)
        return {
            'cross_entropy_loss': avg_loss,
            'f1_score': f1,
            'accuracy': acc,
            'num_samples': total_samples
        }




    def train_single_cycle(self, cycle_num: int, train_data, val_data):


        model = EdgeVGAE(1, 7, self.config.hidden_dim,
                         self.config.latent_dim,
                         self.config.num_classes).to(self.device)

        if len(self.pretrain_models)>0:
            n = len(self.pretrain_models)
            model_file = self.pretrain_models[(cycle_num-1)%n]
            model_data = torch.load(model_file, weights_only=False,map_location=torch.device(self.device))
            model.load_state_dict(model_data['model_state_dict'])
            logging.info(f"Loaded pretrained model: {model_file}")


        train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.config.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.6,
            patience=7,
            min_lr=1e-6,
            verbose=True
        )
        warmup_epochs = self.config.warmup
        best_val_loss, best_f1, epoch_best = float('inf'), 0.0, 0
        best_model_path = None

        for epoch in range(self.config.epochs):
            if epoch < warmup_epochs:
                warm_up_lr(epoch, warmup_epochs, self.config.learning_rate, optimizer)
       

            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer)
            val_metrics = self.evaluate_model(model, val_loader)
            val_loss = val_metrics['cross_entropy_loss']
            val_acc = val_metrics['accuracy']
            val_f1 = val_metrics['f1_score']
            
            if (epoch + 1) % 10 == 0:
                logging.info(
                    f"[Epoch {epoch + 1}] "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )

            if (epoch + 1) % 5 == 0:
                ckpt_path = os.path.join(self.output_dir, 'file_checkpoints', f'ckpt_cycle_{cycle_num}_epoch_{epoch + 1}.pth')

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_f1': val_f1,
                    'train_loss': train_loss,
                    'config': self.config
                }, ckpt_path)
      
            
            print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")


            if epoch >= warmup_epochs:
                scheduler.step(val_metrics['f1_score'])

            if val_metrics['f1_score'] > best_f1:
                best_val_loss = val_metrics['cross_entropy_loss']
                best_f1 = val_metrics['f1_score']
                epoch_best = epoch

                best_model_path = os.path.join(
                    self.output_dir, 'best',
                    f"best_model_{self.config.folder_name}_cycle_{cycle_num}.pth"
                )


                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': best_val_loss,
                    'val_f1': best_f1,
                    'train_loss': train_loss,
                    'config': self.config
                }, best_model_path)


            if (epoch - epoch_best) > self.config.early_stopping_patience // 2 and epoch % 10 == 0:
                checkpoint = torch.load(best_model_path, weights_only=False, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])

            if (epoch - epoch_best) > self.config.early_stopping_patience:
                break

        self.models.append(best_model_path)
        return best_val_loss, best_f1, best_model_path

    def train_epoch(self, model, train_loader, optimizer):
        model.train()
        total_loss, total_samples = 0.0, 0
        correct = 0
    
        for data in tqdm(train_loader, desc="Training", leave=False):
            data = data.to(self.device)
            optimizer.zero_grad()
    
            z, mu, logvar, logits = model(data.x, data.edge_index, data.edge_attr, data.batch)
            recon_loss = model.recon_loss(z, data.edge_index, data.edge_attr)
            kl_loss = model.kl_loss(mu, logvar)
            class_loss = self.criterion(logits, data.y)
    
            loss = 0.15 * recon_loss + 0.1 * kl_loss + class_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
            preds = logits.argmax(dim=1)
            correct += (preds == data.y).sum().item()
    
            batch_size = data.y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
        avg_loss = total_loss / total_samples
        acc = correct / total_samples
        return avg_loss, acc


    def train_multiple_cycles(self, df, num_cycles=10):
        self.load_pretrained()
        results = []

        for cycle in range(num_cycles):
            cycle_seed = cycle + 1
            set_seed(cycle_seed)

            train_data, val_data = self.prepare_data_split(df, seed=cycle_seed)
            val_loss, val_f1, model_path = self.train_single_cycle(cycle + 1, train_data, val_data)

            results.append({'cycle': cycle + 1, 'seed': cycle_seed,
                            'val_loss': val_loss, 'val_f1': val_f1, 'model_path': model_path})


        model_paths_file = os.path.join(self.output_dir, f"model_paths_{self.config.folder_name}.txt")
        with open(model_paths_file, 'w') as f:

            f.writelines(f"{p}\n" for p in self.models)

        return results

    def get_model_loss(self, model_path: str) -> float:
        checkpoint = torch.load(model_path, weights_only=False, map_location=self.device)
        return checkpoint['val_loss']

    def prepare_data_split(self, df, seed=1):
        db_lst = df.db.unique()
        if len(db_lst) > 1:
            df_train, df_valid = pd.DataFrame(), pd.DataFrame()
            for db in db_lst:
                idx = (df.db == db)
                tmp_train, tmp_valid = train_test_split(df.loc[idx, :], test_size=0.2, shuffle=True, random_state=seed)
                df_train = pd.concat([df_train, tmp_train], ignore_index=True)
                df_valid = pd.concat([df_valid, tmp_valid], ignore_index=True)
        else:
            df_train, df_valid = train_test_split(df, test_size=0.2, shuffle=True, random_state=seed)

        return create_dataset_from_dataframe(df_train), create_dataset_from_dataframe(df_valid)

    def _compute_ensemble_weights(self, values: np.ndarray, use_loss=True) -> np.ndarray:

        if use_loss:
            weights = np.exp(-values)
        else:
            weights = np.exp(values)
        return weights / np.sum(weights)

    def _ensemble_predict(self, test_df, weight_by='loss'):
        test_dataset = create_dataset_from_dataframe(test_df, result=False)
        test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)

        all_preds, all_values = [], []

        for model_path in self.models:
            model = EdgeVGAE(1, 7, self.config.hidden_dim,
                            self.config.latent_dim,
                            self.config.num_classes).to(self.device)
            checkpoint = torch.load(model_path, weights_only=False, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])

            value = checkpoint['val_loss'] if weight_by == 'loss' else checkpoint['val_f1']
            preds = self.predict(model, self.device, test_loader)

            all_preds.append(preds)
            all_values.append(value)

        all_preds = np.array(all_preds)
        all_values = np.array(all_values)
        weights = self._compute_ensemble_weights(all_values, use_loss=(weight_by == 'loss'))


        num_samples = all_preds.shape[1]
        num_classes = self.config.num_classes
        weighted_votes = np.zeros((num_samples, num_classes))

        for i, preds in enumerate(all_preds):
            for idx, pred_class in enumerate(preds):
                weighted_votes[idx, pred_class] += weights[i]

        ensemble_preds = np.argmax(weighted_votes, axis=1)
        confidence_scores = np.max(weighted_votes, axis=1)

        unique, counts = np.unique(ensemble_preds, return_counts=True)


        return ensemble_preds, confidence_scores


    def predict_with_ensemble(self, test_df):
        return self._ensemble_predict(test_df, weight_by='loss')

    def predict_with_ensemble_score(self, test_df):
        return self._ensemble_predict(test_df, weight_by='score')

    def predict_with_threshold(self, test_df, confidence_threshold=0.5):
        preds, confidences = self.predict_with_ensemble(test_df)
        filtered_preds = np.where(confidences > confidence_threshold, preds, -1)
        return filtered_preds, confidences
