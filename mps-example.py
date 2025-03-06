import torch
import torch.nn as nn
import numpy as np
import time
from expfloat import ExpFloat, ExpFloatLinear, quantize_model, check_mps_availability, benchmark_expfloat_mps

# Verifica disponibilità MPS
print("Verificando disponibilità MPS...")
has_mps = check_mps_availability()
default_device = torch.device('mps' if has_mps else 'cpu')

# Modello di esempio
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Test di quantizzazione base
def basic_quantization_test():
    print("\n=== Test di quantizzazione base ===")
    
    # Crea un tensore di esempio su MPS (se disponibile)
    x = torch.randn(10, 10, device=default_device)
    print(f"Tensore originale (shape: {x.shape}, device: {x.device})")
    
    # Converti a ExpFloat
    x_exp = ExpFloat(x)
    print(f"ExpFloat (shape degli esponenti: {x_exp.exponents.shape}, device: {x_exp.exponents.device})")
    
    # Riconverti a tensore normale
    x_back = x_exp.to_tensor()
    print(f"Riconvertito (shape: {x_back.shape}, device: {x_back.device})")
    
    # Calcola errore di quantizzazione
    error = torch.abs(x - x_back).mean()
    print(f"Errore medio di quantizzazione: {error:.6f}")

# Test di confronto memoria
def memory_comparison():
    print("\n=== Confronto utilizzo memoria ===")
    
    # Crea matrici di diversi formati per confrontare memoria
    size = 1000
    
    # Crea una matrice float32 standard
    matrix_f32 = torch.randn(size, size, device=default_device)
    
    # Crea una matrice float16
    matrix_f16 = matrix_f32.to(torch.float16)
    
    # Crea una matrice ExpFloat
    matrix_exp = ExpFloat(matrix_f32)
    
    # Calcola utilizzo memoria
    mem_f32 = matrix_f32.nelement() * matrix_f32.element_size()
    mem_f16 = matrix_f16.nelement() * matrix_f16.element_size()
    mem_exp = matrix_exp.exponents.nelement() * matrix_exp.exponents.element_size()
    
    print(f"Matrice {size}x{size}:")
    print(f"Float32: {mem_f32/1024/1024:.2f} MB")
    print(f"Float16: {mem_f16/1024/1024:.2f} MB")
    print(f"ExpFloat: {mem_exp/1024/1024:.2f} MB")
    print(f"Risparmio vs Float32: {(1 - mem_exp/mem_f32) * 100:.1f}%")
    print(f"Risparmio vs Float16: {(1 - mem_exp/mem_f16) * 100:.1f}%")

# Test di operazioni matriciali
def matrix_operations_test():
    print("\n=== Test operazioni matriciali ===")
    
    # Crea due matrici
    size = 512
    a = torch.randn(size, size, device=default_device)
    b = torch.randn(size, size, device=default_device)
    
    # Converti a ExpFloat
    a_exp = ExpFloat(a)
    b_exp = ExpFloat(b)
    
    # Esegui moltiplicazione standard
    start = time.time()
    c = torch.matmul(a, b)
    standard_time = time.time() - start
    
    # Esegui moltiplicazione ExpFloat
    start = time.time()
    c_exp = ExpFloat.matmul(a_exp, b_exp).to_tensor()
    exp_time = time.time() - start
    
    # Calcola errore
    error = torch.abs(c - c_exp).mean() / torch.abs(c).mean()
    
    print(f"Moltiplicazione matriciale {size}x{size}:")
    print(f"Tempo standard: {standard_time*1000:.2f} ms")
    print(f"Tempo ExpFloat: {exp_time*1000:.2f} ms")
    print(f"Ratio: {exp_time/standard_time:.2f}x")
    print(f"Errore relativo medio: {error:.6f}")

# Quantizzazione di modello
def model_quantization_test():
    print("\n=== Test quantizzazione modello ===")
    
    # Crea un modello e sposta su dispositivo appropriato
    model = SimpleModel().to(default_device)
    
    # Crea dati di input
    batch_size = 32
    x = torch.randn(batch_size, 1, 28, 28, device=default_device)
    
    # Forward pass con modello originale
    start = time.time()
    original_output = model(x)
    original_time = time.time() - start
    
    # Quantizza il modello
    quantized_model = quantize_model(model)
    
    # Forward pass con modello quantizzato
    start = time.time()
    quantized_output = quantized_model(x)
    quantized_time = time.time() - start
    
    # Calcola errore di quantizzazione
    error = torch.abs(original_output - quantized_output).mean()
    
    print(f"Inferenza con batch size {batch_size}:")
    print(f"Tempo modello originale: {original_time*1000:.2f} ms")
    print(f"Tempo modello quantizzato: {quantized_time*1000:.2f} ms")
    print(f"Ratio: {quantized_time/original_time:.2f}x")
    print(f"Errore medio: {error:.6f}")

# Benchmark completo
def run_benchmark():
    print("\n=== Benchmark ExpFloat su MPS ===")
    if has_mps:
        benchmark_expfloat_mps()
    else:
        print("MPS non disponibile, impossibile eseguire benchmark MPS")

# Esempio con CNN per visione
def cnn_example():
    print("\n=== Test con modello CNN ===")
    
    # Verifica che torchvision sia disponibile
    try:
        from torchvision.models import resnet18
        
        # Carica un modello ResNet18 pre-addestrato
        model = resnet18(pretrained=False).to(default_device)
        
        # Crea input di esempio
        x = torch.randn(1, 3, 224, 224, device=default_device)
        
        # Forward pass con modello originale
        with torch.no_grad():
            start = time.time()
            original_output = model(x)
            original_time = time.time() - start
        
        # Quantizza il modello
        print("Quantizzazione del modello ResNet18...")
        quantized_model = quantize_model(model)
        
        # Forward pass con modello quantizzato
        with torch.no_grad():
            start = time.time()
            quantized_output = quantized_model(x)
            quantized_time = time.time() - start
        
        # Calcola errore di quantizzazione
        error = torch.abs(original_output - quantized_output).mean()
        
        print(f"Inferenza ResNet18:")
        print(f"Tempo modello originale: {original_time*1000:.2f} ms")
        print(f"Tempo modello quantizzato: {quantized_time*1000:.2f} ms")
        print(f"Ratio: {quantized_time/original_time:.2f}x")
        print(f"Errore medio: {error:.6f}")
        
    except ImportError:
        print("torchvision non disponibile. Installa con: pip install torchvision")

if __name__ == "__main__":
    print(f"PyTorch versione: {torch.__version__}")
    print(f"Device predefinito: {default_device}")
    
    # Esegui i test
    basic_quantization_test()
    memory_comparison()
    matrix_operations_test()
    model_quantization_test()
    run_benchmark()
    cnn_example()