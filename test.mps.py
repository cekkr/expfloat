import torch
from exp_float_mps import ExpFloat, quantize_model

# Carica un modello esistente
model = torch.load("il_tuo_modello.pt")

# Verifica disponibilità MPS
if torch.backends.mps.is_available():
    device = torch.device('mps')
    model = model.to(device)
    
    # Quantizza il modello al formato ExpFloat
    quantized_model = quantize_model(model)
    
    # Esegui inferenza con il modello quantizzato
    input_tensor = torch.randn(1, 3, 224, 224, device=device)
    output = quantized_model(input_tensor)
    
    # Salva il modello quantizzato
    torch.save(quantized_model, "modello_quantizzato.pt")