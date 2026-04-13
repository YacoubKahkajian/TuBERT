"""
Multi-modal RNN for emotion recognition.
"""

import torch
import torch.nn as nn

import config


class AttentionLayer(nn.Module):
    """
    Attention mechanism that weights GRU hidden states by importance.
    A single linear layer maps each hidden state to an unnormalized scalar
    score.  Padding positions are masked to ``-inf`` before softmax so they
    contribute zero weight to the context vector.
    """

    def __init__(self, hidden_dim):
        """
        Initialise the attention layer. Creates a single :class:`~torch.nn.Linear`
        layer that projects each GRU hidden state from ``hidden_dim`` dimensions
        down to a single unnormalized attention score.

        Args:
            hidden_dim : (int)
                Dimensionality of the incoming GRU hidden states.  For a
                bidirectional GRU this should be ``gru_hidden_size * 2``.
        """
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, gru_output, lengths):
        """
        Compute a context vector as a length-masked, softmax-weighted sum.

        Args:
            gru_output : (torch.Tensor)
                Shape ``(batch, seq_len, hidden_dim)``.  All GRU hidden states
                for the batch, including padding frames.
            lengths : (torch.Tensor)
                Shape ``(batch,)``.  The true (unpadded) sequence length for
                each sample; used to mask padded positions to ``-inf`` before
                softmax.

        Returns:
            context : (torch.Tensor)
                Shape ``(batch, hidden_dim)``.  Weighted sum of GRU outputs
                using the attention distribution.
            weights : (torch.Tensor)
                Shape ``(batch, seq_len)``.  Softmax attention weights
                (zero on padded positions).
        """
        # Compute attention scores
        scores = self.attention(gru_output)  # (batch, seq_len, 1)
        scores = scores.squeeze(-1)  # (batch, seq_len)

        # Mask padding positions
        batch_size, max_len = scores.shape
        mask = torch.arange(max_len, device=scores.device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, float("-inf"))

        # Apply softmax
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len)

        # Compute weighted sum
        context = torch.bmm(weights.unsqueeze(1), gru_output)  # (batch, 1, hidden_dim)
        context = context.squeeze(1)  # (batch, hidden_dim)

        return context, weights


class MultiModalEmotionRNN(nn.Module):
    """
    Multi-modal emotion recognition model combining audio and text.
    The model has three main stages:

    1. **Audio pathway**: A multi-layer (optionally bidirectional) GRU
       reads the MFCC+delta+delta-delta feature sequence.  An optional
       :class:`AttentionLayer` aggregates the per-frame hidden states into a
       single vector. If attention is disabled the final hidden state is used.

    2. **Text pathway**: A two-layer FC network (``text_fc``) projects the
       DistilBERT [CLS]-token embedding into the same dimensionality as the
       GRU output so both modalities are comparable.

    3. **Fusion and classification**: Audio and text representations are
       concatenated, compressed by a two-layer fusion MLP (``fusion``), and
       mapped to per-class logits by a linear classification head
       (``classifier``).
    """

    def __init__(
        self,
        audio_dim=config.AUDIO_FEATURE_DIM,
        text_dim=config.TEXT_EMBEDDING_DIM,
        hidden_dim=config.GRU_HIDDEN_DIM,
        num_layers=config.GRU_NUM_LAYERS,
        num_classes=len(config.EMOTIONS),
        dropout=config.GRU_DROPOUT,
        bidirectional=config.BIDIRECTIONAL,
        use_attention=config.USE_ATTENTION,
    ):
        """Build all sub-modules of the multi-modal RNN.

        Args:
            audio_dim : (int)
                Feature dimensionality of each audio frame
                (``n_mfcc * 3`` for MFCC+delta+delta-delta).
                Defaults to ``config.AUDIO_FEATURE_DIM``.
            text_dim : (int)
                Dimensionality of the DistilBERT embedding
                (768 for ``distilbert-base-uncased``).
                Defaults to ``config.TEXT_EMBEDDING_DIM``.
            hidden_dim : (int)
                Number of features in the GRU hidden state and the
                intermediate FC layers.
                Defaults to ``config.GRU_HIDDEN_DIM``.
            num_layers : (int)
                Number of stacked GRU layers.  Dropout is only applied
                between layers when ``num_layers > 1``.
                Defaults to ``config.GRU_NUM_LAYERS``.
            num_classes : (int)
                Number of output emotion classes.
                Defaults to ``len(config.EMOTIONS)``.
            dropout : (float)
                Dropout probability applied inside the GRU (between layers)
                and in the text and fusion FC blocks.
                Defaults to ``config.GRU_DROPOUT``.
            bidirectional : (bool)
                If ``True`` the GRU runs in both directions and its output
                dimension is ``hidden_dim * 2``.
                Defaults to ``config.BIDIRECTIONAL``.
            use_attention : (bool)
                If ``True`` an :class:`AttentionLayer` is added after the GRU
                to compute a context vector via a soft weighted sum over all
                frames.  If ``False`` the final hidden state is used instead.
                Defaults to ``config.USE_ATTENTION``.

        Layers created:
            audio_gru : nn.GRU
                Processes the padded audio feature sequence.
            attention : AttentionLayer  (only when ``use_attention=True``)
                Computes a soft weighted sum over GRU hidden states.
            text_fc : nn.Sequential
                Two-layer FC projection for DistilBERT embeddings:
                ``text_dim → hidden_dim → gru_output_dim``.
            fusion : nn.Sequential
                Two-layer FC that compresses the concatenated audio+text vector:
                ``fusion_dim → hidden_dim → hidden_dim // 2``.
            classifier : nn.Linear
                Maps the fused representation to ``num_classes`` logits.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # Audio pathway: GRU
        self.audio_gru = nn.GRU(
            input_size=audio_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Attention for audio
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        if use_attention:
            self.attention = AttentionLayer(gru_output_dim)

        # Text pathway: Simple FC to match audio representation size
        self.text_fc = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, gru_output_dim),
        )

        # Fusion: Combine audio + text
        fusion_dim = gru_output_dim * 2  # Concatenate audio and text
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classification head
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, audio_features, text_embeddings, lengths):
        """
        Run a forward pass through the full multi-modal network.
        Audio frames are packed before the GRU (so padded frames are skipped)
        and unpacked afterwards.  The audio representation is obtained via
        attention if enabled, otherwise from the final hidden state.  The text
        embedding is projected by ``text_fc``, then both representations are
        concatenated, fused, and classified.

        Args:
            audio_features : (torch.Tensor)
                Shape ``(batch, seq_len, audio_dim)``.  Zero-padded
                MFCC+delta+delta-delta feature sequences.
            text_embeddings : (torch.Tensor)
                Shape ``(batch, text_dim)``.  DistilBERT [CLS]-token
                embeddings for each utterance.
            lengths : (torch.Tensor)
                Shape ``(batch,)``.  True (unpadded) sequence length for each
                sample.  Used by ``pack_padded_sequence`` and by
                :class:`AttentionLayer`.

        Returns:
            logits : (torch.Tensor)
                Shape ``(batch, num_classes)``.  Unnormalized class scores
                (apply softmax / argmax for probabilities / predicted class).
        """
        batch_size = audio_features.size(0)

        # --- Audio pathway ----------------------------------
        # Pack padded sequences for efficient RNN processing
        packed_audio = nn.utils.rnn.pack_padded_sequence(
            audio_features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # GRU forward pass
        packed_output, hidden = self.audio_gru(packed_audio)

        # Unpack
        gru_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )  # (batch, seq_len, hidden_dim * 2 if bidirectional)

        # Get audio representation
        if self.use_attention:
            # Use attention to weight important frames
            audio_repr, attention_weights = self.attention(gru_output, lengths)
        else:
            # Use final hidden state
            if self.bidirectional:
                # Concatenate final forward and backward states
                hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
                forward_hidden = hidden[-1, 0, :, :]  # (batch, hidden_dim)
                backward_hidden = hidden[-1, 1, :, :]  # (batch, hidden_dim)
                audio_repr = torch.cat([forward_hidden, backward_hidden], dim=1)
            else:
                audio_repr = hidden[-1]  # (batch, hidden_dim)

        # --- Text pathway -------------------------------------------
        text_repr = self.text_fc(text_embeddings)  # (batch, gru_output_dim)

        # --- Fusion --------------------------------------------------
        # Concatenate audio and text representations
        fused = torch.cat([audio_repr, text_repr], dim=1)  # (batch, fusion_dim)

        # Fusion layers
        fused_repr = self.fusion(fused)  # (batch, hidden_dim // 2)

        # Classification
        logits = self.classifier(fused_repr)  # (batch, num_classes)

        return logits
