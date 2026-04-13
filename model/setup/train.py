"""
Training script for multi-modal emotion recognition.
Imports the RNN from rnn.py and the data from PreprocessedDataset
and trains on that.

Also contains fine_tune_iemocap() for fine-tuning a pretrained MELD
checkpoint on IEMOCAP sessions 1-4 and evaluating on session 5. This
function is really only relevant for the original thesis paper.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloader, get_iemocap_dataloaders
from rnn import MultiModalEmotionRNN
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config


class Trainer:
    """Training harness for MultiModalEmotionRNN.
    Two high-level entry points are provided:

    * :meth:`train`: full training loop on pre-extracted MELD features.
    * :meth:`fine_tune_iemocap`: loads a MELD checkpoint and fine-tunes on
      IEMOCAP sessions 1-4 with leave-one-session-out evaluation on session 5.
    """

    def __init__(self, model, device=config.DEVICE):
        """Set up the training harness.

        Args:
            model : (MultiModalEmotionRNN)
                The model to train. It is moved to ``device`` immediately.
            device : (str)
                The device to train on (e.g. ``"cpu"``, ``"cuda"``, or
                ``"mps"``). Defaults to ``config.DEVICE``.
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.writer = SummaryWriter(log_dir="runs/emotion_recognition")

    def train_epoch(self, train_loader, epoch):
        """Run one full pass over the training set and return loss and accuracy.

        Args:
            train_loader : (DataLoader)
                DataLoader yielding ``(audio, text, labels, lengths)`` batches.
            epoch : (int)
                Current epoch number; displayed in the progress bar.

        Returns:
            avg_loss : (float)
                Mean cross-entropy loss over all batches.
            accuracy : (float)
                Fraction of correctly classified samples in the epoch.
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for audio, text, labels, lengths in pbar:
            # Move to device
            audio = audio.to(self.device)
            text = text.to(self.device)
            labels = labels.to(self.device)
            lengths = lengths.to(self.device)

            # Forward pass
            logits = self.model(audio, text, lengths)
            loss = self.criterion(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            pbar.set_postfix({"loss": loss.item(), "acc": correct / total})

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, val_loader):
        """Evaluate the model on a validation or test set.

        Accumulates loss, accuracy, and all predictions and ground-truth
        labels, then delegates to :meth:`compute_class_metrics` for
        per-emotion precision, recall, and F1.

        Args:
            val_loader : (DataLoader)
                DataLoader yielding ``(audio, text, labels, lengths)`` batches.

        Returns:
            avg_loss : (float)
                Mean cross-entropy loss over all batches.
            accuracy : (float)
                Fraction of correctly classified samples.
            class_metrics : (dict)
                Per-emotion metrics dict as returned by
                :meth:`compute_class_metrics`.
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        all_predictions = []
        all_labels = []

        for audio, text, labels, lengths in tqdm(val_loader, desc="Validating"):
            # Move to device
            audio = audio.to(self.device)
            text = text.to(self.device)
            labels = labels.to(self.device)
            lengths = lengths.to(self.device)

            # Forward pass
            logits = self.model(audio, text, lengths)
            loss = self.criterion(logits, labels)

            # Metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        # Compute per-class metrics
        class_metrics = self.compute_class_metrics(all_predictions, all_labels)

        return avg_loss, accuracy, class_metrics

    def compute_class_metrics(self, predictions, labels):
        """Compute per-emotion precision, recall, F1, and support.

        Uses ``sklearn.metrics.precision_recall_fscore_support`` with
        ``zero_division=0`` so that classes with no predicted samples do not
        raise warnings. Metrics are computed for every class index in
        ``config.EMOTIONS`` regardless of whether it appears in this batch.

        Args:
            predictions : (list[int])
                Predicted class indices.
            labels : (list[int])
                Ground-truth class indices.

        Returns:
            metrics : (dict[str, dict])
                Outer keys are emotion name strings from ``config.EMOTIONS``.
                Each inner dict contains:

                  - ``"precision"`` (float) — fraction of predicted positives that are correct.
                  - ``"recall"`` (float) — fraction of actual positives predicted correctly.
                  - ``"f1"`` (float) — harmonic mean of precision and recall.
                  - ``"support"`` (int) — number of ground-truth samples for this class.
        """
        from sklearn.metrics import precision_recall_fscore_support

        precision, recall, f1, support = precision_recall_fscore_support(
            labels,
            predictions,
            labels=list(range(len(config.EMOTIONS))),
            zero_division=0,
        )
        metrics = {}
        for i, emotion in enumerate(config.EMOTIONS):
            metrics[emotion] = {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": support[i],
            }
        return metrics

    def save_checkpoint(self, epoch, val_loss, val_acc, is_best=False, suffix=""):
        """Save a model checkpoint to ``config.MODELS_DIR``.

        Always writes ``latest{suffix}.pt``. When ``is_best=True``, also
        writes ``best{suffix}.pt`` and prints a confirmation message.

        Args:
            epoch : (int)
                The epoch number to store in the checkpoint.
            val_loss : (float)
                Validation loss at this epoch.
            val_acc : (float)
                Validation accuracy at this epoch.
            is_best : (bool)
                If ``True``, the checkpoint is also saved as the best model.
                Defaults to ``False``.
            suffix : (str)
                Optional string appended to both file names, e.g. ``"_iemocap"``
                so fine-tuned checkpoints don't overwrite MELD ones.
                Defaults to ``""``.
        """
        config.MODELS_DIR.mkdir(exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        torch.save(checkpoint, config.MODELS_DIR / f"latest{suffix}.pt")
        if is_best:
            torch.save(checkpoint, config.MODELS_DIR / f"best{suffix}.pt")
            print(f"Saved best model{suffix} (val_acc: {val_acc:.4f})")

    def train(self, num_epochs=config.NUM_EPOCHS):
        """Run the full training loop on preprocessed MELD data.

        Loads the train and dev splits via :func:`~data_loader.get_dataloader`
        (neutral-class undersampling is applied to the training split by
        default), then trains for up to ``num_epochs`` epochs. After each
        epoch the validation loss is passed to the learning-rate scheduler,
        train/val metrics are logged to TensorBoard, and a checkpoint is
        saved. Training stops early if ``config.EARLY_STOPPING_PATIENCE``
        consecutive epochs pass without improvement in validation accuracy.

        Args:
            num_epochs : (int)
                Maximum number of training epochs.
                Defaults to ``config.NUM_EPOCHS``.
        """
        print("Loading data")
        train_loader = get_dataloader(split="train", shuffle=True)
        val_loader = get_dataloader(split="dev", shuffle=False)

        print(f"\nTraining for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print()

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_acc, class_metrics = self.evaluate(val_loader)

            # Schedule learning rate
            self.scheduler.step(val_loss)

            # Log
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)

            # Print results
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            # Print per-class F1 scores
            print("\nPer-class F1 scores:")
            for emotion, metrics in class_metrics.items():
                print(f"  {emotion:10s}: {metrics['f1']:.3f} (n={metrics['support']})")

            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            self.save_checkpoint(epoch, val_loss, val_acc, is_best)

            # Early stopping
            if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        self.writer.close()

    def fine_tune_iemocap(
        self,
        pretrained_path,
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE * 0.1,
    ):
        """Fine-tune a pretrained MELD checkpoint on IEMOCAP.

        Uses leave-one-session-out: trains on sessions 1-4, validates on
        session 5. Features are extracted on-the-fly (no preprocessing
        needed). Saves checkpoints with an ``'_iemocap'`` suffix so they
        don't overwrite your MELD model.

        A lower learning rate (default: 1e-5, one tenth of the MELD rate)
        is used to avoid catastrophic forgetting of what the model learned
        on MELD. A fresh Adam optimiser and ``ReduceLROnPlateau`` scheduler
        (patience 3) are created so that MELD momentum does not carry over
        into the IEMOCAP loss landscape. Per-epoch metrics and TensorBoard
        logs are written to ``runs/emotion_recognition_iemocap``.

        Args:
            pretrained_path : (str | Path)
                Path to the MELD checkpoint to start from,
                e.g. ``config.MODELS_DIR / "best.pt"``.
            num_epochs : (int)
                Maximum number of fine-tuning epochs.
                Defaults to ``config.NUM_EPOCHS``.
            learning_rate : (float)
                Learning rate for fine-tuning. Defaults to one tenth of
                ``config.LEARNING_RATE`` (typically 1e-5) to prevent
                catastrophic forgetting.
        """
        # ------------------------------------------------------------------
        # Load pretrained MELD weights
        # ------------------------------------------------------------------
        print(f"Loading pretrained checkpoint from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"  Resumed from epoch {checkpoint['epoch']} "
            f"(MELD val_acc: {checkpoint['val_acc']:.4f})"
        )

        # ------------------------------------------------------------------
        # Reset optimiser and scheduler with the lower fine-tuning LR.
        # We intentionally do NOT load the MELD optimiser state — starting
        # fresh avoids carrying over momentum that was calibrated to MELD's
        # loss landscape.
        # ------------------------------------------------------------------
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )
        self.best_val_acc = 0.0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # ------------------------------------------------------------------
        # Data — on-the-fly extraction, num_workers=0
        # ------------------------------------------------------------------
        print("Building IEMOCAP dataloaders...")
        train_loader, val_loader = get_iemocap_dataloaders(batch_size=config.BATCH_SIZE)

        print(f"\nFine-tuning on IEMOCAP for up to {num_epochs} epochs")
        print(f"Device:        {self.device}")
        print(f"Learning rate: {learning_rate}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches:   {len(val_loader)}")
        print()

        ft_writer = SummaryWriter(log_dir="runs/emotion_recognition_iemocap")

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc, class_metrics = self.evaluate(val_loader)
            self.scheduler.step(val_loss)

            ft_writer.add_scalar("Loss/train", train_loss, epoch)
            ft_writer.add_scalar("Loss/val", val_loss, epoch)
            ft_writer.add_scalar("Accuracy/train", train_acc, epoch)
            ft_writer.add_scalar("Accuracy/val", val_acc, epoch)

            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            print("\nPer-class F1 scores:")
            for emotion, metrics in class_metrics.items():
                print(f"  {emotion:10s}: {metrics['f1']:.3f} (n={metrics['support']})")

            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, val_loss, val_acc, is_best, suffix="_iemocap")

            if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

        print(f"Best IEMOCAP val accuracy: {self.best_val_acc:.4f}")
        ft_writer.close()


def main():
    """Initialise a fresh model and run the full MELD training loop.

    Prints the trainable parameter count, constructs a :class:`Trainer`,
    and calls :meth:`Trainer.train`. ``config.MODELS_DIR`` is created if
    it does not already exist.
    """
    print("Initializing model")
    model = MultiModalEmotionRNN()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    trainer = Trainer(model)
    trainer.train()
    config.MODELS_DIR.mkdir(exist_ok=True)


def main_fine_tune():
    """Initialise a model and fine-tune the best MELD checkpoint on IEMOCAP.

    Constructs a :class:`Trainer` with a fresh :class:`MultiModalEmotionRNN`
    and calls :meth:`Trainer.fine_tune_iemocap` using the best MELD
    checkpoint at ``config.MODELS_DIR / "cmvn_best.pt"``.
    """
    print("Initializing model for IEMOCAP fine-tuning")
    model = MultiModalEmotionRNN()
    trainer = Trainer(model)
    trainer.fine_tune_iemocap(
        pretrained_path=config.MODELS_DIR / "cmvn_best.pt",
    )


if __name__ == "__main__":
    main_fine_tune()
