from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


@dataclass
class TrainingHistory:
    epochs: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    training_time_seconds: float = 0.0


@dataclass
class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    inference_latency_ms: float
    predictions: np.ndarray
    probabilities: np.ndarray


# ============================================================================
# 1. NumPy Convolutional Engine (Zero dependências externas pesadas para testes)
# ============================================================================

class NumpyCNNClassifier:
    """Implementação convolucional pura em NumPy (Conv2D + MaxPool + Dense + Sigmoid/Softmax).
    
    Permite executar testes e validações locais de forma determinística e ultrarrápida.
    """

    def __init__(self, input_shape: Tuple[int, int, int] = (64, 64, 3), num_classes: int = 2, random_state: int = 42):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.rng = np.random.default_rng(random_state)

        # 4 filtros convolucionais 3x3x3
        self.n_filters = 4
        self.filter_size = 3
        self.filters = self.rng.normal(0, 0.1, (self.n_filters, self.filter_size, self.filter_size, input_shape[2]))
        self.bias_conv = np.zeros(self.n_filters)

        # Pooling 2x2 reduz 64x64 para 31x31 (com conv sem padding: (64-3+1)=62 -> pool=31)
        h_conv = (input_shape[0] - self.filter_size + 1) // 2
        w_conv = (input_shape[1] - self.filter_size + 1) // 2
        self.flatten_dim = h_conv * w_conv * self.n_filters

        # Camada densa com inicialização He
        self.w_fc = self.rng.normal(0, np.sqrt(2.0 / self.flatten_dim), (self.flatten_dim, num_classes))
        self.b_fc = np.zeros(num_classes)

    def _conv_and_pool(self, image: np.ndarray) -> np.ndarray:
        """Aplica convolução 2D simples + ReLU + MaxPool (2x2)."""
        h, w, c = image.shape
        out_h = h - self.filter_size + 1
        out_w = w - self.filter_size + 1
        conv_out = np.zeros((out_h, out_w, self.n_filters), dtype=np.float32)

        for f in range(self.n_filters):
            kernel = self.filters[f]
            for i in range(0, out_h, 2):  # stride 2 simplificado para acelerar
                for j in range(0, out_w, 2):
                    patch = image[i : i + self.filter_size, j : j + self.filter_size, :]
                    conv_out[i, j, f] = np.maximum(0.0, np.sum(patch * kernel) + self.bias_conv[f])

        # Max pooling 2x2
        pool_h, pool_w = out_h // 2, out_w // 2
        pooled = np.zeros((pool_h, pool_w, self.n_filters), dtype=np.float32)
        for f in range(self.n_filters):
            for i in range(pool_h):
                for j in range(pool_w):
                    pooled[i, j, f] = np.max(conv_out[i * 2 : (i + 1) * 2, j * 2 : (j + 1) * 2, f])

        return pooled.flatten()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Calcula probabilidades para um lote de imagens."""
        n_samples = len(x)
        logits = np.zeros((n_samples, self.num_classes), dtype=np.float32)

        for i in range(n_samples):
            features = self._conv_and_pool(x[i])
            logits[i] = np.dot(features, self.w_fc) + self.b_fc

        # Softmax estável
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 15,
        lr: float = 0.01,
    ) -> TrainingHistory:
        """Treina a rede com gradiente descendente estocástico/mini-batch e rastreia histórico."""
        history = TrainingHistory()
        start_time = time.perf_counter()

        n_train = len(x_train)
        y_one_hot = np.eye(self.num_classes)[y_train]

        # Pré-computa representação latente convolucional para acelerar épocas
        features_train = np.array([self._conv_and_pool(img) for img in x_train], dtype=np.float32)
        features_val = np.array([self._conv_and_pool(img) for img in x_val], dtype=np.float32)

        for epoch in range(1, epochs + 1):
            # Forward treino
            logits_train = np.dot(features_train, self.w_fc) + self.b_fc
            exp_train = np.exp(logits_train - np.max(logits_train, axis=1, keepdims=True))
            probs_train = exp_train / np.sum(exp_train, axis=1, keepdims=True)

            # Cross-entropy loss
            loss_train = -np.mean(np.log(np.clip(probs_train[np.arange(n_train), y_train], 1e-12, 1.0)))
            preds_train = np.argmax(probs_train, axis=1)
            acc_train = float(np.mean(preds_train == y_train))

            # Gradientes camada densa
            grad_logits = (probs_train - y_one_hot) / n_train
            grad_w = np.dot(features_train.T, grad_logits)
            grad_b = np.sum(grad_logits, axis=0)

            # Atualização dos pesos
            self.w_fc -= lr * grad_w
            self.b_fc -= lr * grad_b

            # Avaliação na validação
            logits_val = np.dot(features_val, self.w_fc) + self.b_fc
            exp_val = np.exp(logits_val - np.max(logits_val, axis=1, keepdims=True))
            probs_val = exp_val / np.sum(exp_val, axis=1, keepdims=True)
            loss_val = -np.mean(np.log(np.clip(probs_val[np.arange(len(x_val)), y_val], 1e-12, 1.0)))
            preds_val = np.argmax(probs_val, axis=1)
            acc_val = float(np.mean(preds_val == y_val))

            history.epochs.append(epoch)
            history.train_loss.append(round(float(loss_train), 4))
            history.train_acc.append(round(float(acc_train), 4))
            history.val_loss.append(round(float(loss_val), 4))
            history.val_acc.append(round(float(acc_val), 4))

        history.training_time_seconds = round(time.perf_counter() - start_time, 3)
        return history

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> EvaluationMetrics:
        """Calcula métricas de classificação completas e mede latência de inferência."""
        start_time = time.perf_counter()
        probs = self.forward(x_test)
        total_time_ms = (time.perf_counter() - start_time) * 1000.0
        latency_ms = total_time_ms / max(1, len(x_test))

        preds = np.argmax(probs, axis=1)

        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        cm = confusion_matrix(y_test, preds)

        return EvaluationMetrics(
            accuracy=round(acc, 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1=round(f1, 4),
            confusion_matrix=cm,
            inference_latency_ms=round(latency_ms, 2),
            predictions=preds,
            probabilities=probs,
        )


# ============================================================================
# 2. Definição do Módulo PyTorch para Google Colab
# ============================================================================

PYTORCH_COLAB_TEMPLATE = """
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class FarmTechPyTorchCNN(nn.Module):
    \"\"\"Arquitetura Convolucional do Zero para Classificação Agrícola na FarmTech.\"\"\"
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(64 * 8 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
"""
