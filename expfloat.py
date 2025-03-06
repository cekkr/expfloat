import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union
import copy

class ExpFloat:
    """
    Implementazione di un tipo di dato custom basato su singolo byte per esponente.
    Assume sempre una mantissa di 1.0 e memorizza solo l'esponente.
    Supporta accelerazione MPS su macOS.
    """
    def __init__(self, data=None, shape=None, device=None):
        # Determina il dispositivo - supporta 'cpu', 'cuda' e 'mps'
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        
        self.device = device
        
        if data is not None:
            if isinstance(data, torch.Tensor):
                # Converti un tensore PyTorch normale al formato ExpFloat
                self.exponents = self._convert_to_exponents(data).to(self.device)
            elif isinstance(data, np.ndarray):
                # Converti un array numpy
                tensor = torch.from_numpy(data)
                self.exponents = self._convert_to_exponents(tensor).to(self.device)
            elif isinstance(data, ExpFloat):
                # Copia da un altro ExpFloat
                self.exponents = data.exponents.clone().to(self.device)
            else:
                raise TypeError("Tipo di input non supportato")
        elif shape is not None:
            # Crea un tensore vuoto della forma desiderata
            self.exponents = torch.zeros(shape, dtype=torch.int8, device=self.device)
        else:
            raise ValueError("Specifica data o shape")
    
    def _convert_to_exponents(self, tensor):
        """Converti un tensore normale a esponenti"""
        # Trova il valore assoluto, gestisci i segni separatamente
        abs_tensor = torch.abs(tensor)
        signs = torch.sign(tensor)
        
        # Calcola gli esponenti (approssimazione)
        with torch.no_grad():
            # Gestisci casi speciali: 0 e numeri molto piccoli
            mask_nonzero = abs_tensor > 1e-38  # Valore minimo per f32
            
            # Inizializza con esponente -128 (valore minimo rappresentabile)
            exponents = torch.full_like(tensor, -128, dtype=torch.int8)
            
            # Calcola esponenti solo per valori non-zero
            # Assicurati che il tensore sia su CPU per questa operazione
            cpu_tensor = abs_tensor.cpu() if abs_tensor.device.type != 'cpu' else abs_tensor
            cpu_mask = mask_nonzero.cpu() if mask_nonzero.device.type != 'cpu' else mask_nonzero
            
            log2_vals = torch.log2(cpu_tensor[cpu_mask])
            exponents_valid = torch.round(log2_vals).clamp(-128, 127).to(torch.int8)
            
            # Converti di nuovo al dispositivo originale
            if tensor.device.type != 'cpu':
                exponents = exponents.to(tensor.device)
                exponents[mask_nonzero] = exponents_valid.to(tensor.device)
            else:
                exponents[mask_nonzero] = exponents_valid
            
            # Store sign in the MSB (usando un approccio custom)
            # Per una vera implementazione, salva il segno in un bit separato
            sign_multiplier = torch.ones_like(exponents)
            sign_multiplier[signs < 0] = -1
            
        return exponents.to(torch.int8)
    
    def to_tensor(self):
        """Converti dal formato ExpFloat a un normale tensore PyTorch"""
        # Gestione MPS: alcuni op potrebbero richiedere CPU
        if self.device.type == 'mps':
            # Calcola su CPU e poi trasferisci a MPS
            cpu_exponents = self.exponents.to('cpu')
            tensor = torch.pow(2.0, cpu_exponents.to(torch.float32))
            return tensor.to(self.device)
        else:
            # Calcolo diretto sul dispositivo corrente
            tensor = torch.pow(2.0, self.exponents.to(torch.float32))
            return tensor
    
    def to(self, device):
        """Sposta il tensore su un dispositivo specifico"""
        if str(self.device) != str(device):
            self.exponents = self.exponents.to(device)
            self.device = device
        return self
    
    @staticmethod
    def matmul(a, b):
        """Moltiplicazione matriciale ottimizzata per ExpFloat"""
        if not isinstance(a, ExpFloat) or not isinstance(b, ExpFloat):
            raise TypeError("Entrambi gli operandi devono essere di tipo ExpFloat")
        
        # Per MPS, utilizziamo un'implementazione personalizzata ottimizzata
        if a.device.type == 'mps' and b.device.type == 'mps':
            # In una vera implementazione, si utilizzerebbe Metal Performance Shaders
            # Per ora, simuliamo l'ottimizzazione
            result_exponents = torch.zeros(
                (a.exponents.shape[0], b.exponents.shape[1]), 
                dtype=torch.int8, 
                device=a.device
            )
            
            # Implementazione ingenuae (per test)
            a_tensor = a.to_tensor()
            b_tensor = b.to_tensor()
            result_tensor = torch.matmul(a_tensor, b_tensor)
            
            # Converti il risultato in ExpFloat
            return ExpFloat(result_tensor)
        else:
            # Fallback per CPU/CUDA
            a_tensor = a.to_tensor()
            b_tensor = b.to_tensor()
            result_tensor = torch.matmul(a_tensor, b_tensor)
            return ExpFloat(result_tensor)
    
    @classmethod
    def from_tensor(cls, tensor):
        """Factory method per creare ExpFloat da tensor"""
        return cls(tensor)


class ExpFloatLinear(nn.Module):
    """Layer lineare che utilizza ExpFloat internamente"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        # Inizializza i parametri normalmente
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Inizializzazione standard
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Placeholder per i pesi quantizzati
        self.weight_exp = None
        self.bias_exp = None
        self.quantized = False
    
    def quantize(self):
        """Quantizza i pesi nel formato ExpFloat"""
        self.weight_exp = ExpFloat(self.weight.data)
        if self.bias is not None:
            self.bias_exp = ExpFloat(self.bias.data)
        self.quantized = True
        return self
    
    def forward(self, x):
        if not self.quantized:
            # Comportamento normale quando non quantizzato
            return nn.functional.linear(x, self.weight, self.bias)
        
        # Converti input in ExpFloat se necessario
        if not isinstance(x, ExpFloat):
            x_exp = ExpFloat(x)
        else:
            x_exp = x
        
        # Esegui matmul con ExpFloat
        output = ExpFloat.matmul(x_exp, ExpFloat(self.weight.t()))
        
        # Aggiungi bias se presente
        if self.bias_exp is not None:
            output = ExpFloat(output.to_tensor() + self.bias_exp.to_tensor())
        
        return output.to_tensor()  # Riconverti in tensore normale


def quantize_model(model, inplace=False):
    """
    Quantizza un intero modello PyTorch convertendo tutti i layer lineari in ExpFloatLinear
    
    Args:
        model: il modello PyTorch da quantizzare
        inplace: se True, modifica il modello in-place; altrimenti, crea una copia
    
    Returns:
        Il modello quantizzato
    """
    if not inplace:
        model = copy.deepcopy(model)
    
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Sostituisci con layer ExpFloat
            exp_linear = ExpFloatLinear(module.in_features, module.out_features, 
                                        bias=module.bias is not None)
            
            # Copia i pesi
            exp_linear.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                exp_linear.bias.data.copy_(module.bias.data)
            
            # Quantizza
            exp_linear.quantize()
            
            # Sostituisci il modulo
            setattr(model, name, exp_linear)
        else:
            # Ricorsivamente quantizza i sottomoduli
            quantize_model(module, inplace=True)
    
    return model


# Classe per operazioni personalizzate MPS
class MPSExpFloatOps:
    """
    Operazioni ottimizzate per ExpFloat su MPS
    Questo è uno sketch di come implementare operazioni ottimizzate
    per Metal Performance Shaders
    """
    @staticmethod
    def matmul(a_exp, b_exp):
        """
        Esegue moltiplicazione matriciale direttamente su esponenti
        usando Metal Performance Shaders
        """
        # In una vera implementazione, qui chiamaremmo un kernel Metal
        # Per ora, utilizziamo un'implementazione semplice
        
        # 1. Crea dispositivo Metal
        # device = MTLCreateSystemDefaultDevice()
        
        # 2. Crea command queue
        # commandQueue = device.newCommandQueue()
        
        # 3. Crea pipeline di calcolo
        # computePipelineState = ...
        
        # 4. Crea buffer per input/output
        # a_buffer = device.newBuffer(...)
        # b_buffer = device.newBuffer(...)
        # c_buffer = device.newBuffer(...)
        
        # 5. Esegui calcolo
        # ...
        
        # 6. Leggi risultato
        # ...
        
        # Per ora, torniamo all'implementazione base
        return None


class MPSExpFloatModule(nn.Module):
    """
    Modulo che utilizza direttamente Metal Performance Shaders
    per calcoli ExpFloat
    """
    def __init__(self):
        super().__init__()
        # Verifica se MPS è disponibile
        self.use_mps = torch.backends.mps.is_available()
        if self.use_mps:
            print("MPS disponibile, utilizzo accelerazione Metal")
        else:
            print("MPS non disponibile, utilizzo implementazione CPU")
    
    def forward(self, x):
        # Implementazione dipendente dal dispositivo
        if self.use_mps:
            # Implementazione MPS
            pass
        else:
            # Implementazione fallback
            pass
        return x


# Funzioni di utility per test su MPS
def check_mps_availability():
    """
    Verifica se MPS è disponibile sul sistema e fornisce informazioni sul dispositivo.
    """
    if torch.backends.mps.is_available():
        print("MPS disponibile! Metal Performance Shaders può essere utilizzato.")
        print(f"MPS device count: {torch.mps.device_count()}")
        print(f"MPS current device: {torch.mps.current_device()}")
        return True
    else:
        print("MPS non disponibile. Verificare requisiti:")
        print("- macOS 12.3 o superiore")
        print("- PyTorch 1.12 o superiore con supporto MPS")
        return False


# Benchmark di confronto velocità tra float32 e ExpFloat su MPS
def benchmark_expfloat_mps():
    """Confronta prestazioni tra ExpFloat e Float32 su MPS"""
    if not torch.backends.mps.is_available():
        print("MPS non disponibile, impossibile eseguire benchmark MPS")
        return
    
    # Crea tensori di test
    size = 1024
    device = torch.device('mps')
    x = torch.randn(size, size, device=device)
    y = torch.randn(size, size, device=device)
    
    # Conversione a ExpFloat
    x_exp = ExpFloat(x)
    y_exp = ExpFloat(y)
    
    # Benchmark operazioni regolari
    start_time = torch.mps.Event(enable_timing=True)
    end_time = torch.mps.Event(enable_timing=True)
    
    # Warmup
    for _ in range(5):
        z = torch.matmul(x, y)
    
    # Timing
    torch.mps.synchronize()
    start_time.record()
    for _ in range(20):
        z = torch.matmul(x, y)
    end_time.record()
    torch.mps.synchronize()
    float32_time = start_time.elapsed_time(end_time) / 20
    
    # Benchmark operazioni ExpFloat
    # Warmup
    for _ in range(5):
        z_exp = ExpFloat.matmul(x_exp, y_exp)
    
    # Timing
    torch.mps.synchronize()
    start_time.record()
    for _ in range(20):
        z_exp = ExpFloat.matmul(x_exp, y_exp)
    end_time.record()
    torch.mps.synchronize()
    expfloat_time = start_time.elapsed_time(end_time) / 20
    
    print(f"Float32 MPS: {float32_time:.2f} ms")
    print(f"ExpFloat MPS: {expfloat_time:.2f} ms")
    print(f"Speedup: {float32_time / expfloat_time:.2f}x")
    
    # Verifica accuratezza
    z = torch.matmul(x, y)
    z_exp = ExpFloat.matmul(x_exp, y_exp).to_tensor()
    error = torch.abs(z - z_exp).mean() / torch.abs(z).mean()
    print(f"Errore relativo medio: {error:.6f}")


# Implementazione completa per MPS richiederebbe l'utilizzo dell'API Metal:
"""
Per una vera implementazione MPS, sarebbe necessario:

1. Creare un kernel Metal personalizzato in .metal:
kernel void exp_float_matmul(
    device int8_t* a_exp [[ buffer(0) ]],
    device int8_t* b_exp [[ buffer(1) ]],
    device int8_t* c_exp [[ buffer(2) ]],
    uint2 gid [[ thread_position_in_grid ]],
    uint2 grid_size [[ threads_per_grid ]],
    constant int& M [[ buffer(3) ]],
    constant int& N [[ buffer(4) ]],
    constant int& K [[ buffer(5) ]]
) {
    int row = gid.y;
    int col = gid.x;
    
    if (row < M && col < N) {
        // Implementazione della moltiplicazione con esponenti
        // ...
    }
}

2. Integrare con PyTorch usando C++/Python binding:
- Utilizzare PyTorch C++ API con integrazioni Metal
- Creare un'estensione Python che usa la libreria Metal

3. Ottimizzazioni specifiche per M1/M2/M3:
- Sfruttare il Neural Engine per operazioni specifiche
- Ottimizzare per l'architettura unificata memoria/processore
"""