import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from PIL import Image, ImageFilter
import os
from itertools import combinations
from sklearn.metrics import roc_curve, auc

from torchvision.datasets import ImageFolder
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from torchvision.models import DenseNet



# ------------------------- Custom Dataset -------------------------

# https://docs.pytorch.org/vision/main/datasets.html
# "All datasets are subclasses of torch.utils.data.Dataset i.e, they have __getitem__ and __len__ methods implemented.
# Hence, they can all be passed to a torch.utils.data.DataLoader which can load multiple samples in parallel usin"

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


class EarlyStopper:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



class DenseNetModel:
    def __init__(self, data_dir, num_classes=3, batch_size=62, num_epochs=200, learning_rate=0.000001, train_paths = "", train_labels = "", test_paths = "", test_labels = ""):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),

            transforms.Lambda(lambda img: img.filter(ImageFilter.MedianFilter(size=3))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.data_dir = data_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        #https://docs.pytorch.org/vision/0.9/_modules/torchvision/models/densenet.html

        #self.model = DenseNet(
        #    growth_rate=32,
        #    block_config=(6, 12, 24, 16),  # esta es la config de densenet121
        #    num_init_features=64,
        #    bn_size=4,
        #    drop_rate=0,
        #    num_classes=self.num_classes
        #)
        
        self.model = models.densenet121(weights='DEFAULT')

        self.model.classifier = nn.Linear(self.model.classifier.in_features, self.num_classes)
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []


        full_train_dataset = CustomDataset(train_paths, train_labels, self.transform)

        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

        test_dataset = CustomDataset(test_paths, test_labels, self.transform)

        print(f"Train Size: {train_size}")
        print(f"Val Size: {val_size}")
        print(f"Test Size: {len(test_dataset)}")

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        print(f"Entrenando modelo_{plano}")


        self.train(self.train_loader, self.val_loader, self.num_epochs)






    def train(self, train_loader, val_loader, num_epochs):
        #early_stopper = EarlyStopper(patience=3, min_delta=0.0001)
        print (f"...Comenzando entrenamiento función train...")
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)

            train_loss = running_loss / len(train_loader)
            train_accuracy = 100 * correct_train / total_train
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)

            self.model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    correct_val += (predicted == labels).sum().item()
                    total_val += labels.size(0)

            val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct_val / total_val
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            print(f"Época [{epoch+1}/{self.num_epochs}], Pérdida: {train_loss:.4f}, "
                  f"Precisión (Entrenamiento): {train_accuracy:.2f}%, "
                  f"Precisión (Validación): {val_accuracy:.2f}%, "
                  f"Pérdida de Validación: {val_loss:.4f}")

            #if early_stopper.early_stop(val_loss):
            #    print(f"Early stopping en la época {epoch+1}")
            #    break

        print("Entrenamiento finalizado.")

    def save_model(self, filename="modelo_densenet.pth"):
        torch.save(self.model.state_dict(), filename)
        print(f"Modelo guardado como '{filename}'.")

    def plot_confusion_matrix(self, fold, name, lr, batch_size, class_names):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap='Blues', values_format='')

        plt.title(f"Matriz de Confusión - Fold {fold}")
        plt.savefig(f"Confusion_Matrix_{name}_Fold{fold}_LR{lr}_BS{batch_size}_KFOLD_{fold}.png")
        plt.savefig(f"Confusion_Matrix_{name}_Fold{fold}_LR{lr}_BS{batch_size}_KFOLD_{fold}.svg")
        plt.close()
        return cm

    def plot_metrics(self, fold, name, lr, batch_size):
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(self.train_accuracies, label='Precisión de Entrenamiento')
            plt.plot(self.val_accuracies, label='Precisión de Validación')
            plt.title('Precisión del Modelo')
            plt.ylabel('Precisión (%)')
            plt.xlabel('Época')
            plt.legend(loc='upper left')

            plt.savefig(f"Metric_{name}_acc_LR{lr}_BS{batch_size}_KFOLD_{fold}.png")  #guardar en svg
            plt.savefig(f"Metric_{name}_acc_LR{lr}_BS{batch_size}_KFOLD_{fold}.svg")
            plt.clf()


            print("Se ha guardado la metrica")


            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 2)
            plt.plot(self.train_losses, label='Pérdida de Entrenamiento')
            plt.plot(self.val_losses, label='Pérdida de Validación')
            plt.title('Pérdida del Modelo')
            plt.ylabel('Pérdida')
            plt.xlabel('Época')
            plt.legend(loc='upper right')

            plt.savefig(f"Metric_{name}_loss_LR{lr}_BS{batch_size}_KFOLD_{fold}.png")
            plt.savefig(f"Metric_{name}_loss_LR{lr}_BS{batch_size}_KFOLD_{fold}.svg")  #esta linea estaba mal --> corregida

            plt.clf()

    def get_model_scores(self):
            self.model.eval()
            all_labels = []
            all_scores = []

            with torch.no_grad():
                for images, labels in self.test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    scores = torch.softmax(outputs, dim=1)

                    all_labels.append(labels.cpu().numpy())
                    all_scores.append(scores.cpu().numpy())

            return np.concatenate(all_labels), np.concatenate(all_scores)

    #basado en https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html que me mando pablo
    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html

    #Revisar por que me da que la pifie en algun lado



    def plot_roc_curve(self,fold, name, lr, batch_size):
        y_true, y_scores = self.get_model_scores()


        label_binarizer = LabelBinarizer().fit(y_true)
        y_onehot = label_binarizer.transform(y_true)

        print(label_binarizer.classes_)
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["darkorange", "blue", "green"]
        target_to_class = {v: k for k, v in ImageFolder (data_dir).class_to_idx.items()}
        print(target_to_class)

        fpr_dict = {}
        tpr_dict = {}
        auc_dict = {}

        for i, class_name in target_to_class.items():

            fpr, tpr, _ = roc_curve(y_onehot[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)

            fpr_dict[i] = fpr
            tpr_dict[i] = tpr
            auc_dict[i] = roc_auc

            RocCurveDisplay.from_predictions(
                y_onehot[:, i], y_scores[:, i],
                name=f"{class_name} vs Rest",
                color=colors[i],
                ax=ax
            )

        fpr_grid = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(fpr_grid)

        for i in fpr_dict:
            interp_tpr = np.interp(fpr_grid, fpr_dict[i], tpr_dict[i])
            mean_tpr += interp_tpr

        mean_tpr /= len(fpr_dict)
        mean_auc = auc(fpr_grid, mean_tpr)

        plt.plot(
            fpr_grid,
            mean_tpr,
            linestyle=":",
            linewidth=3,
            color="black",
            label=f"Macro-average (AUC = {mean_auc:.2f})"
        )

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("One-vs-Rest ROC Curves")
        ax.legend()
        plt.savefig(f"Metric__{name}_LR{lr}_BS{batch_size}_KFOLD_{fold}_ROC_curve.png")
        plt.savefig(f"Metric__{name}_LR{lr}_BS{batch_size}_KFOLD_{fold}_ROC_curve.svg")

        plt.close()

        return fpr_dict, tpr_dict, auc_dict


    def binary_plot_roc_curve(self, fold, name, lr, batch_size):

          target_to_class = {v: k for k, v in ImageFolder(self.data_dir).class_to_idx.items()}

          y_true, y_scores = self.get_model_scores()

          label_binarizer = LabelBinarizer().fit(y_true)
          y_onehot = label_binarizer.transform(y_true)

          class_indices = list(range(len(label_binarizer.classes_)))
          pair_list = list(combinations(target_to_class, 2))

          print(f"Class indices {class_indices}")
          print(f"Pair list {pair_list}")

          fpr_grid = np.linspace(0, 1, 1000)
          pair_scores = []
          mean_tpr = dict()

          idx_to_classname = {v: k for k, v in target_to_class.items()}

          all_fprs = []
          all_tprs = []
          all_aucs = []

          print(f"target_to_class {target_to_class}")

          print(f"idx_to_classname {idx_to_classname}")

          for ix, (label_a, label_b) in enumerate(pair_list):
              print(label_a)
              print(label_b)

              a_mask = y_true == label_a
              b_mask = y_true == label_b
              ab_mask = np.logical_or(a_mask, b_mask)

              a_true = a_mask[ab_mask]
              b_true = b_mask[ab_mask]

              idx_a = np.flatnonzero(label_binarizer.classes_ == label_a)[0]
              idx_b = np.flatnonzero(label_binarizer.classes_ == label_b)[0]

              fpr_a, tpr_a, _ = roc_curve(a_true, y_scores[ab_mask, idx_a])
              fpr_b, tpr_b, _ = roc_curve(b_true, y_scores[ab_mask, idx_b])

              mean_tpr[ix] = np.zeros_like(fpr_grid)
              mean_tpr[ix] += np.interp(fpr_grid, fpr_a, tpr_a)
              mean_tpr[ix] += np.interp(fpr_grid, fpr_b, tpr_b)
              mean_tpr[ix] /= 2
              mean_score = auc(fpr_grid, mean_tpr[ix])
              pair_scores.append(mean_score)

              #esto es pa las medias
              #------------------------------------------

              all_fprs.append(fpr_grid)
              all_tprs.append(mean_tpr[ix])
              all_aucs.append(mean_score)

              #------------------------------------------

              fig, ax = plt.subplots(figsize=(6, 6))
              plt.plot(
                  fpr_grid,
                  mean_tpr[ix],
                  label=f"Mean {target_to_class[label_a]} vs {target_to_class[label_b]} (AUC = {mean_score:.2f})",
                  linestyle=":",
                  linewidth=4,
              )
              RocCurveDisplay.from_predictions(
                  a_true,
                  y_scores[ab_mask, idx_a],
                  ax=ax,
                  name=f"{target_to_class[label_a]} as positive class",
              )
              RocCurveDisplay.from_predictions(
                  b_true,
                  y_scores[ab_mask, idx_b],
                  ax=ax,
                  name=f"{target_to_class[label_b]} as positive class",
                  plot_chance_level=True,
                  despine=True,
              )
              ax.set(
                  xlabel="False Positive Rate",
                  ylabel="True Positive Rate",
                  title=f"{target_to_class[label_a]} vs {target_to_class[label_b]} ROC curves",
              )
              ax.legend()


              plt.savefig(f"Metric__{name}_{target_to_class[label_a]} vs {target_to_class[label_b]}_LR{lr}_BS{batch_size}_binary_ROC_curve__KFOLD_{fold}.png")
              plt.savefig(f"Metric__{name}_{target_to_class[label_a]} vs {target_to_class[label_b]}_LR{lr}_BS{batch_size}_binary_ROC_curve__KFOLD_{fold}.svg")

              plt.close()

          print(f"\nMacro-averaged One-vs-One ROC AUC score:\n{np.average(pair_scores):.2f}")
          return all_fprs, all_tprs, all_aucs, pair_list

#Dataset solo SC
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

trainin_paths = [r'C:\Users\comci\Desktop\TFG Eric\Datase_ordernado_visitas\m12\Axial', r'C:\Users\comci\Desktop\TFG Eric\Datase_ordernado_visitas\m12\Coronal', r'C:\Users\comci\Desktop\TFG Eric\Datase_ordernado_visitas\m12\Sagittal']
diagnosis_training_path = [r'C:\Users\comci\Desktop\TFG Eric\Datase_ordernado_visitas\sc\Axial', r'C:\Users\comci\Desktop\TFG Eric\Datase_ordernado_visitas\sc\Coronal', r'C:\Users\comci\Desktop\TFG Eric\Datase_ordernado_visitas\sc\Sagittal']
diagnosis_training_path_m06 = [r'C:\Users\comci\Desktop\TFG Eric\Datase_ordernado_visitas\m06\Axial', r'C:\Users\comci\Desktop\TFG Eric\Datase_ordernado_visitas\m06\Coronal', r'C:\Users\comci\Desktop\TFG Eric\Datase_ordernado_visitas\m06\Sagittal']

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#Por si se me olvida, el lr lo hardcodeamos a 10^-5 por que asi va volando el modelo.

lr = 0.000055
cont = 0  
batch_size = 64  #Este fue el que dijimos de mirar en la reunion

#==========================================================================================================
#                                       MEJORES PARAMETROS
#                                           K = 12
#                                          BS = 64
#==========================================================================================================

def get_mean_confusion_matrix(all_confusion_matrices,class_names):
            mean_cm = np.mean(np.stack(all_confusion_matrices), axis=0)
            mean_cm = np.round(mean_cm, 1)
            disp = ConfusionMatrixDisplay(confusion_matrix=mean_cm, display_labels=class_names)

            #https://stackoverflow.com/questions/65463392/sklearn-metrics-confusionmatrixdisplay-using-scientific-notation/65463511
            #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
            #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

            disp.plot(cmap='Blues', values_format='')
            plt.title("Matriz de Confusión Promedio")
            plt.savefig(f"Confusion_Matrix_Media_{plano}_LR{lr}_BS{batch_size}.png")
            plt.savefig(f"Confusion_Matrix_Media_{plano}_LR{lr}_BS{batch_size}.svg")
            plt.close()

def get_image_paths_and_labels(data_dir):
        dataset = ImageFolder(data_dir)
        image_paths = [s[0] for s in dataset.samples]
        labels = [s[1] for s in dataset.samples]
        return image_paths, labels, dataset.classes

def save_avg_metrics_plot(all_train_losses, all_val_losses, all_train_accuracies, all_val_accuracies, plano, lr, batch_size):
    # Calcular el promedio de todas las métricas a través de los folds
    avg_train_losses = [sum(epoch_losses) / len(epoch_losses) for epoch_losses in zip(*all_train_losses)]
    avg_val_losses = [sum(epoch_losses) / len(epoch_losses) for epoch_losses in zip(*all_val_losses)]
    avg_train_accuracies = [sum(epoch_accuracies) / len(epoch_accuracies) for epoch_accuracies in zip(*all_train_accuracies)]
    avg_val_accuracies = [sum(epoch_accuracies) / len(epoch_accuracies) for epoch_accuracies in zip(*all_val_accuracies)]

    # === Gráfica de Precisión Promedio ===
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    # Graficamos la precisión promedio de entrenamiento y validación
    plt.plot(avg_train_accuracies, color='blue', label='Precisión Promedio de Entrenamiento')
    plt.plot(avg_val_accuracies, color='orange', label='Precisión Promedio de Validación')
    plt.title('Precisión Promedio del Modelo (Media de los Folds)')
    plt.ylabel('Precisión (%)')
    plt.xlabel('Época')
    plt.legend(loc='upper left')

    # Guardamos la imagen en formato .png y .svg
    plt.savefig(f"Metric_avg_acc_LR{lr}_BS{batch_size}_PLANO_{plano}.png")
    plt.savefig(f"Metric_avg_acc_LR{lr}_BS{batch_size}_PLANO_{plano}.svg")
    plt.clf()

    print("Se ha guardado la métrica de precisión promedio")

    # === Gráfica de Pérdida Promedio ===
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 2)
    # Graficamos la pérdida promedio de entrenamiento y validación
    plt.plot(avg_train_losses, color='blue', label='Pérdida Promedio de Entrenamiento')
    plt.plot(avg_val_losses, color='orange', label='Pérdida Promedio de Validación')
    plt.title('Pérdida Promedio del Modelo (Media de los Folds)')
    plt.ylabel('Pérdida')
    plt.xlabel('Época')
    plt.legend(loc='upper right')

    # Guardamos la imagen en formato .png y .svg
    plt.savefig(f"Metric_avg_loss_LR{lr}_BS{batch_size}_PLANO_{plano}.png")
    plt.savefig(f"Metric_avg_loss_LR{lr}_BS{batch_size}_PLANO_{plano}.svg")
    plt.clf()

    print("Se ha guardado la métrica de pérdida promedio")


def plot_mean_roc_curves_by_class(all_fpr_ad, all_fpr_cn, all_fpr_mci,
                                  all_tpr_ad, all_tpr_cn, all_tpr_mci,
                                  all_auc_ad, all_auc_cn, all_auc_mci,
                                  plano, lr, batch_size):
    mean_fpr = np.linspace(0, 1, 100)

    # Interpolación para cada clase
    def interpolate_tprs(fprs, tprs):
        return [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fprs, tprs)]

    interp_tpr_ad = interpolate_tprs(all_fpr_ad, all_tpr_ad)
    interp_tpr_cn = interpolate_tprs(all_fpr_cn, all_tpr_cn)
    interp_tpr_mci = interpolate_tprs(all_fpr_mci, all_tpr_mci)

    mean_tpr_ad = np.mean(interp_tpr_ad, axis=0)
    mean_tpr_cn = np.mean(interp_tpr_cn, axis=0)
    mean_tpr_mci = np.mean(interp_tpr_mci, axis=0)

    mean_auc_ad = np.mean(all_auc_ad)
    mean_auc_cn = np.mean(all_auc_cn)
    mean_auc_mci = np.mean(all_auc_mci)

    # Corregimos el último punto
    mean_tpr_ad[-1] = 1.0
    mean_tpr_cn[-1] = 1.0
    mean_tpr_mci[-1] = 1.0

    # Graficamos
    plt.figure(figsize=(8, 6))
    plt.plot(mean_fpr, mean_tpr_ad, label=f"AD vs Rest (AUC = {mean_auc_ad:.2f})", color='darkorange', lw=2)
    plt.plot(mean_fpr, mean_tpr_cn, label=f"CN vs Rest (AUC = {mean_auc_cn:.2f})", color='blue', lw=2)
    plt.plot(mean_fpr, mean_tpr_mci, label=f"MCI vs Rest (AUC = {mean_auc_mci:.2f})", color='green', lw=2)
    
    # --------------------------
    # Calcular curva promedio (macro-average) de las 3 curvas 1vsRest
    # --------------------------
    mean_tpr_macro = (mean_tpr_ad + mean_tpr_cn + mean_tpr_mci) / 3
    mean_auc_macro = auc(mean_fpr, mean_tpr_macro)

    # Añadir curva promedio a la gráfica
    plt.plot(
        mean_fpr,
        mean_tpr_macro,
        label=f"Macro-average (AUC = {mean_auc_macro:.2f})",
        color="black",
        linestyle=":",
        linewidth=4,
    )


    plt.plot([0, 1], [0, 1], 'k--', label='Random')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mean ROC Curves (1 vs Rest)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()

    plt.savefig(f"Mean_ROC_curve_{plano}_LR{lr}_BS{batch_size}.png")
    plt.savefig(f"Mean_ROC_curve_{plano}_LR{lr}_BS{batch_size}.svg")
    plt.close()

    print("Se ha guardado la gráfica de las curvas ROC promedio.")


def calculate_mean_binary_roc(all_fpr_pairs, all_tpr_pairs, all_auc_pairs):


    mean_fpr_pairs = []
    mean_tpr_pairs = []
    mean_auc_pairs = []

    # Iteramos sobre los pares de clases (cada par de clases tiene sus propias métricas)
    for i in range(len(all_fpr_pairs[0])):  # Asumimos que todos los folds tienen el mismo número de pares
        # Obtener todas las métricas de cada fold para este par de clases
        fprs = [all_fpr_pairs[fold][i] for fold in range(len(all_fpr_pairs))]
        tprs = [all_tpr_pairs[fold][i] for fold in range(len(all_tpr_pairs))]
        aucs = [all_auc_pairs[fold][i] for fold in range(len(all_auc_pairs))]

        # Calcular la media de las métricas
        mean_fpr = np.mean(fprs, axis=0)
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = np.mean(aucs)

        # Almacenamos las métricas medias
        mean_fpr_pairs.append(mean_fpr)
        mean_tpr_pairs.append(mean_tpr)
        mean_auc_pairs.append(mean_auc)

    return mean_fpr_pairs, mean_tpr_pairs, mean_auc_pairs

def plot_mean_binary_roc_curve(mean_fpr_pairs, mean_tpr_pairs, mean_auc_pairs, pair_list, plano, lr, batch_size):


    plt.figure(figsize=(8, 6))
    target_to_class = {v: k for k, v in ImageFolder (data_dir).class_to_idx.items()}

    # Iteramos sobre los pares de clases
    for i, (fpr, tpr, auc_value, pair) in enumerate(zip(mean_fpr_pairs, mean_tpr_pairs, mean_auc_pairs, pair_list)):
        label_a, label_b = pair
        plt.plot(fpr, tpr, label=f"{target_to_class[label_a]} vs {target_to_class[label_b]} (AUC = {auc_value:.2f})", lw=2)
        #plt.fill_between(fpr, tpr - 0.05, tpr + 0.05, alpha=0.2)  # Para dar una visualización del intervalo de confianza

        # Línea aleatoria (Random)
        plt.plot([0, 1], [0, 1], 'k--', label='Random')

        # ------------------------------
        # Calcular curva media global (macro-average)
        # ------------------------------
        fpr_grid = np.linspace(0, 1, 100)
        mean_tpr_macro = np.zeros_like(fpr_grid)

        for tpr, fpr in zip(mean_tpr_pairs, mean_fpr_pairs):
            interp_tpr = np.interp(fpr_grid, fpr, tpr)
            mean_tpr_macro += interp_tpr

        mean_tpr_macro /= len(mean_tpr_pairs)
        mean_auc_macro = auc(fpr_grid, mean_tpr_macro)

        # Añadir la curva a la gráfica
        plt.plot(
            fpr_grid,
            mean_tpr_macro,
            label=f"Macro-average (AUC = {mean_auc_macro:.2f})",
            color="black",
            linestyle=":",
            linewidth=4,
        )



        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Mean ROC Curves (One-vs-One)')
        plt.legend(loc='lower right')
        plt.grid()
        plt.tight_layout()

        # Guardamos la gráfica
        plt.savefig(f"Mean_{plano}_{target_to_class[label_a]} vs {target_to_class[label_b]}_LR{lr}_BS{batch_size}_binary_ROC_curve.png")
        plt.savefig(f"Mean_{plano}_{target_to_class[label_a]} vs {target_to_class[label_b]}_LR{lr}_BS{batch_size}_binary_ROC_curve.svg")

        plt.close()

    print("Se ha guardado la gráfica de las curvas ROC promediadas One-vs-One.")


for path, diag_path , diag_path_m06 in zip(trainin_paths, diagnosis_training_path , diagnosis_training_path_m06):
        data_dir = path
        name = path.split('/')[-1]

        print("==================================================================")

        print(f"EJECUCION {cont}")

        print("==================================================================")



        plano = os.path.basename(name)

        image_paths, labels, class_names = get_image_paths_and_labels(data_dir)
        
        #---------- Objetivo 3 -------------------------------------------------

        sc_paths, sc_labels, classes = get_image_paths_and_labels(diag_path)
        m06_paths, m06_labels, classes = get_image_paths_and_labels(diag_path_m06)
        
        #-----------------------------------------------------------------------
        print(class_names)
        k_folds = 5
        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=0)


        all_confusion_matrices = []

        #Para obtener la metrica de la media de todas las metricas loss y acc de los folds
        all_train_losses = []
        all_val_losses = []
        all_train_accuracies = []
        all_val_accuracies = []

        #Para obtener la metrica de la media de los roc curves
        all_fpr_ad = []
        all_fpr_cn = []
        all_fpr_mci = []

        all_tpr_ad = []
        all_tpr_cn = []
        all_tpr_mci = []

        all_auc_ad = []
        all_auc_cn = []
        all_auc_mci = []

 #Para obtener la metrica de la media de los roc binatrias

        all_fpr_pairs = []
        all_tpr_pairs = []
        all_auc_pairs = []

        print (f"Entrenando modelo_{plano}_LR{lr}_BS{batch_size}")

        for fold, (train_idx, test_idx) in enumerate(kfold.split(image_paths, labels)):
            print(f"\n====== Fold {fold+1}/{k_folds} ======")

            train_paths = [image_paths[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            test_paths = [image_paths[i] for i in test_idx]
            test_labels = [labels[i] for i in test_idx]

            
            #---------- Objetivo 3 ---------------------------------------------
            
            train_paths += sc_paths + m06_paths
            train_labels += sc_labels + m06_labels
            
            #-------------------------------------------------------------------

            densenet_model = DenseNetModel(data_dir=data_dir, num_classes=3, num_epochs=5,
                                        learning_rate=lr, batch_size=batch_size, train_paths = train_paths,
                                        train_labels = train_labels, test_paths = test_paths, test_labels = test_labels)

            densenet_model.save_model(f"densenet_fold{fold+1}_Plano_{plano}.pth")

            #Le pedimos que nos genere y guarde las métricas de loss y acc
            densenet_model.plot_metrics(fold+1, plano, lr, batch_size)

            all_train_losses.append(densenet_model.train_losses)
            all_val_losses.append(densenet_model.val_losses)
            all_train_accuracies.append(densenet_model.train_accuracies)
            all_val_accuracies.append(densenet_model.val_accuracies)

            #Le pedimos que nos genere y guarde las métricas de roc 1 vs rest
            fpr, tpr, roc_auc = densenet_model.plot_roc_curve(fold+1, plano, lr, batch_size)

            # Las clases ya están ordenadas
            class_indices = sorted(fpr.keys())  # Claves ordenadas ( 0: AD, 1: CN, 2: MCI)

            # Almacenar las métricas en las listas correspondientes
            all_fpr_ad.append(fpr[0])
            all_fpr_cn.append(fpr[1])
            all_fpr_mci.append(fpr[2])

            all_tpr_ad.append(tpr[0])
            all_tpr_cn.append(tpr[1])
            all_tpr_mci.append(tpr[2])

            all_auc_ad.append(roc_auc[0])
            all_auc_cn.append(roc_auc[1])
            all_auc_mci.append(roc_auc[2])

            #Le pedimos que nos genere y guarde las métricas de roc de 1vs1

            fpr_pairs, tpr_pairs, auc_pairs, pair_list = densenet_model.binary_plot_roc_curve(fold+1, plano, lr, batch_size)

            # Almacenamos las métricas de cada fold
            all_fpr_pairs.append(fpr_pairs)
            all_tpr_pairs.append(tpr_pairs)
            all_auc_pairs.append(auc_pairs)


            cm = densenet_model.plot_confusion_matrix(fold+1, plano, lr, batch_size, class_names)
            all_confusion_matrices.append(cm)


        get_mean_confusion_matrix(all_confusion_matrices,class_names)

        # Llamada a la función para graficar las curvas ROC promedio
        plot_mean_roc_curves_by_class(
            all_fpr_ad, all_fpr_cn, all_fpr_mci,
            all_tpr_ad, all_tpr_cn, all_tpr_mci,
            all_auc_ad, all_auc_cn, all_auc_mci,
            plano, lr, batch_size
        )


        save_avg_metrics_plot(all_train_losses, all_val_losses, all_train_accuracies, all_val_accuracies, plano, lr, batch_size)

        print("All Confusion matrices")

        print("----------------------------------------------------------")

        print(all_confusion_matrices)

        print("----------------------------------------------------------")

        # Calculamos la media de las métricas
        mean_fpr_pairs, mean_tpr_pairs, mean_auc_pairs = calculate_mean_binary_roc(all_fpr_pairs, all_tpr_pairs, all_auc_pairs)

        # Graficamos las curvas ROC promediadas
        plot_mean_binary_roc_curve(mean_fpr_pairs, mean_tpr_pairs, mean_auc_pairs, pair_list, plano, lr, batch_size)

        cont+=1


#Una posible herramienta que usare mas tarde
class MetricOperations:
    def __init__(self):
        pass

