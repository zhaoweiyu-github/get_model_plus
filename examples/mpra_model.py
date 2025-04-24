#!/usr/bin/env python
# Example of extending the base model to include MPRA prediction

import os
import sys
import torch
import torch.nn as nn
import lightning as L
from omegaconf import OmegaConf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from get_model.run import LitModel
from get_model.model.model import Transformer, Model

class MPRAModel(Model):
    """
    Extension of the base GET model to include MPRA activity prediction
    alongside gene expression prediction.
    """
    def __init__(self, config):
        super().__init__(config)
        
        # Create an additional prediction head for MPRA activity
        self.mpra_head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim // 2, 1)  # MPRA activity is a single value
        )
    
    def forward(self, x, mask=None, y=None, tss_idx=None):
        """
        Forward pass that includes both gene expression and MPRA prediction.
        
        Args:
            x: Input tensor with region-motif data
            mask: Mask for training/prediction
            y: Target tensor (gene expression labels)
            tss_idx: TSS indices
            
        Returns:
            Dictionary containing predictions for gene expression and MPRA activity
        """
        # Get base model outputs (including gene expression predictions)
        base_outputs = super().forward(x, mask, y, tss_idx)
        
        # Get the transformer outputs for MPRA prediction
        if hasattr(base_outputs, 'hidden_states'):
            transformer_outputs = base_outputs.hidden_states[-1]
        else:
            # If using a different model architecture, we might need
            # to extract features differently
            transformer_outputs = self.transformer(x)
        
        # Generate MPRA activity predictions
        mpra_preds = self.mpra_head(transformer_outputs)
        
        # Add MPRA predictions to output dictionary
        if isinstance(base_outputs, dict):
            base_outputs['mpra'] = mpra_preds
        else:
            base_outputs = {
                'exp': base_outputs,
                'mpra': mpra_preds
            }
            
        return base_outputs


class MPRALitModel(LitModel):
    """
    Lightning module for the MPRA-extended model.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def get_model(self):
        """
        Create a model instance with the MPRA-extended model.
        """
        config = self.cfg.model
        
        # Create base transformer
        transformer = Transformer(
            config.num_motifs + 1,  # +1 for ATAC signal
            config.model_dim,
            config.num_heads,
            config.depth,
            config.dim_feedforward,
            config.dropout,
            config.num_region_per_sample
        )
        
        # Use our MPRA-extended model
        model = MPRAModel(config)
        model.transformer = transformer
        
        return model
    
    def _shared_step(self, batch, batch_idx, stage="train"):
        """
        Shared step that handles both gene expression and MPRA prediction.
        """
        region_motif = batch["region_motif"]
        mask = batch["mask"]
        exp_label = batch["exp_label"]
        
        # Get model predictions
        preds = self.model(region_motif, mask)
        
        # Calculate gene expression loss
        exp_loss = self.criterion(preds['exp'], exp_label)
        
        # Calculate MPRA loss if MPRA data is available
        mpra_loss = 0
        if "mpra_label" in batch and "mpra_mask" in batch:
            mpra_label = batch["mpra_label"]
            mpra_mask = batch["mpra_mask"]
            
            # Only consider regions with MPRA data
            if torch.any(mpra_mask > 0):
                # Mask predictions to only include regions with MPRA data
                masked_preds = preds['mpra'][mpra_mask > 0]
                masked_labels = mpra_label[mpra_mask > 0]
                
                # Calculate MPRA loss (mean squared error)
                mpra_loss = torch.nn.functional.mse_loss(masked_preds, masked_labels)
        
        # Combined loss (can adjust weights between tasks)
        mpra_weight = self.cfg.get('mpra_weight', 0.5)  # Default weight
        total_loss = exp_loss + mpra_weight * mpra_loss
        
        # Log metrics
        self.log(f"{stage}_exp_loss", exp_loss, prog_bar=True)
        self.log(f"{stage}_mpra_loss", mpra_loss, prog_bar=True)
        self.log(f"{stage}_total_loss", total_loss, prog_bar=True)
        
        return total_loss, preds, batch


def main():
    """
    Example showing how to create and use the MPRA-extended model.
    """
    # Basic configuration
    config = {
        "model": {
            "num_motifs": 1000,
            "model_dim": 512,
            "num_heads": 8,
            "depth": 6,
            "dim_feedforward": 2048,
            "dropout": 0.1,
            "num_region_per_sample": 1000
        },
        "mpra_weight": 0.5  # Weight for MPRA loss
    }
    
    # Convert to OmegaConf
    cfg = OmegaConf.create(config)
    
    # Create model
    model = MPRALitModel(cfg)
    
    # Print model summary
    logging.info("Created MPRA-extended model")
    logging.info(f"Model uses MPRA prediction head with weight {cfg.mpra_weight}")


if __name__ == "__main__":
    main() 