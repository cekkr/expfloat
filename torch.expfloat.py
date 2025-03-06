# File: metal_integration.py
# Integrazione di Metal con PyTorch per operazioni ExpFloat

import torch
import numpy as np
import ctypes
import os
import platform
from typing import List, Tuple, Optional

class MetalExpFloatBackend:
    """
    Backend per ExpFloat utilizzando Metal su macOS.
    Gestisce l'interazione tra PyTorch e Metal per operazioni ottimizzate.
    """
    def __init__(self):
        self.is_available = self._check_metal_availability()
        self.lib = None
        
        if self.is_available:
            self._init_metal_library()
    
    def _check_metal_availability(self) -> bool:
        """Verifica se Metal è disponibile sul sistema"""
        if platform.system() != "Darwin":
            return False
        
        # Verifica che sia macOS con chip Apple Silicon o Metal compatibile
        try:
            # La presenza di mps in torch indica supporto Metal
            return torch.backends.mps.is_available()
        except:
            return False
    
    def _init_metal_library(self) -> None:
        """Inizializza la libreria Metal"""
        try:
            # In un'implementazione reale, qui caricheremmo 
            # una libreria C/Objective-C che interfaccia con Metal
            # self.lib = ctypes.CDLL("./libexpfloat_metal.dylib")
            
            # Per ora, impostiamo solo un flag
            self.initialized = True
            print("Metal backend inizializzato per ExpFloat")
        except Exception as e:
            print(f"Errore nell'inizializzazione del backend Metal: {e}")
            self.initialized = False
    
    def matmul(self, a_exp: torch.Tensor, b_exp: torch.Tensor) -> torch.Tensor:
        """
        Esegue moltiplicazione matriciale su esponenti usando Metal
        
        Args:
            a_exp: Tensore di esponenti (int8)
            b_exp: Tensore di esponenti (int8)
            
        Returns:
            Tensore risultante di esponenti (int8)
        """
        if not self.is_available or not self.initialized:
            raise RuntimeError("Backend Metal non disponibile")
        
        # Verifica che i tensori siano sul dispositivo MPS
        if a_exp.device.type != 'mps' or b_exp.device.type != 'mps':
            raise ValueError("I tensori devono essere su dispositivo MPS")
        
        # Verifica che i tensori siano int8
        if a_exp.dtype != torch.int8 or b_exp.dtype != torch.int8:
            raise ValueError("I tensori devono essere di tipo int8")
        
        # Ottieni dimensioni
        M, K = a_exp.shape
        K2, N = b_exp.shape
        
        if K != K2:
            raise ValueError(f"Dimensioni incompatibili: {a_exp.shape} e {b_exp.shape}")
        
        # Prepara tensore di output
        c_exp = torch.zeros((M, N), dtype=torch.int8, device='mps')
        
        # In un'implementazione reale, qui chiameremmo la funzione C
        # che interfaccia con il kernel Metal
        # self.lib.exp_float_matmul(
        #     ctypes.c_void_p(a_exp.data_ptr()),
        #     ctypes.c_void_p(b_exp.data_ptr()),
        #     ctypes.c_void_p(c_exp.data_ptr()),
        #     ctypes.c_int(M),
        #     ctypes.c_int(N),
        #     ctypes.c_int(K)
        # )
        
        # Per ora, simuliamo l'operazione usando PyTorch
        # Questo è un fallback per dimostrare il concetto
        # Convertiamo al dominio lineare
        a_float = torch.pow(2.0, a_exp.to(torch.float32))
        b_float = torch.pow(2.0, b_exp.to(torch.float32))
        
        # Eseguiamo la moltiplicazione
        c_float = torch.matmul(a_float, b_float)
        
        # Convertiamo di nuovo al dominio esponente
        c_exp = torch.round(torch.log2(c_float)).clamp(-128, 127).to(torch.int8)
        
        return c_exp
    
    def float_to_exp(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Converte un tensore float in formato ExpFloat usando Metal
        
        Args:
            tensor: Tensore float da convertire
            
        Returns:
            Tensore di esponenti (int8)
        """
        if not self.is_available or not self.initialized:
            raise RuntimeError("Backend Metal non disponibile")
        
        # Verifica che il tensore sia sul dispositivo MPS
        if tensor.device.type != 'mps':
            raise ValueError("Il tensore deve essere su dispositivo MPS")
        
        # Prepara tensore di output
        output = torch.zeros(tensor.shape, dtype=torch.int8, device='mps')
        
        # In un'implementazione reale, chiameremmo la funzione C
        # self.lib.float_to_exp_float(
        #     ctypes.c_void_p(tensor.data_ptr()),
        #     ctypes.c_void_p(output.data_ptr()),
        #     ctypes.c_int(tensor.numel())
        # )
        
        # Per ora, fallback PyTorch
        abs_tensor = torch.abs(tensor)
        mask_nonzero = abs_tensor > 1e-38
        
        # Inizializza con -128 (valore minimo)
        exponents = torch.full_like(tensor, -128, dtype=torch.int8)
        
        # Calcola esponenti per valori non-zero
        log2_vals = torch.log2(abs_tensor[mask_nonzero])
        exponents_valid = torch.round(log2_vals).clamp(-128, 127).to(torch.int8)
        exponents[mask_nonzero] = exponents_valid
        
        return exponents
    
    def exp_to_float(self, exponents: torch.Tensor) -> torch.Tensor:
        """
        Converte un tensore di esponenti in float usando Metal
        
        Args:
            exponents: Tensore di esponenti (int8)
            
        Returns:
            Tensore float
        """
        if not self.is_available or not self.initialized:
            raise RuntimeError("Backend Metal non disponibile")
        
        # Verifica che il tensore sia sul dispositivo MPS
        if exponents.device.type != 'mps':
            raise ValueError("Il tensore deve essere su dispositivo MPS")
        
        # Verifica che il tensore sia int8
        if exponents.dtype != torch.int8:
            raise ValueError("Il tensore deve essere di tipo int8")
        
        # Prepara tensore di output
        output = torch.zeros(exponents.shape, dtype=torch.float32, device='mps')
        
        # In un'implementazione reale, chiameremmo la funzione C
        # self.lib.exp_float_to_float(
        #     ctypes.c_void_p(exponents.data_ptr()),
        #     ctypes.c_void_p(output.data_ptr()),
        #     ctypes.c_int(exponents.numel())
        # )
        
        # Per ora, fallback PyTorch
        output = torch.pow(2.0, exponents.to(torch.float32))
        
        return output

# Funzione per creare un'implementazione in C/Objective-C per l'integrazione Metal
def generate_metal_bridge_code() -> str:
    """
    Genera il codice C/Objective-C per interfacciare PyTorch con Metal.
    Questa funzione restituisce il codice che dovrebbe essere compilato
    in una libreria dinamica per l'uso effettivo.
    """
    code = """
    // metal_bridge.mm - Integrazione tra PyTorch e Metal per ExpFloat
    
    #import <Foundation/Foundation.h>
    #import <Metal/Metal.h>
    #include <torch/torch.h>
    
    // Dichiarazione delle funzioni esportate
    extern "C" {
        int init_metal_device();
        int exp_float_matmul(void* a_ptr, void* b_ptr, void* c_ptr, int M, int N, int K);
        int float_to_exp_float(void* input_ptr, void* output_ptr, int size);
        int exp_float_to_float(void* input_ptr, void* output_ptr, int size);
    }
    
    // Variabili globali per Metal
    static id<MTLDevice> device = nil;
    static id<MTLLibrary> library = nil;
    static id<MTLCommandQueue> commandQueue = nil;
    static id<MTLComputePipelineState> matmulPipeline = nil;
    static id<MTLComputePipelineState> floatToExpPipeline = nil;
    static id<MTLComputePipelineState> expToFloatPipeline = nil;
    
    // Inizializza il dispositivo Metal
    int init_metal_device() {
        @autoreleasepool {
            // Ottieni dispositivo Metal predefinito
            device = MTLCreateSystemDefaultDevice();
            if (!device) {
                NSLog(@"Metal non disponibile su questo dispositivo");
                return -1;
            }
            
            // Carica la libreria di shader
            NSError *error = nil;
            NSString *kernelSource = @"#include <metal_stdlib>\\n"
                                     "using namespace metal;\\n"
                                     // ... [inserire codice degli shader qui] ...
                                     ;
            
            library = [device newLibraryWithSource:kernelSource options:nil error:&error];
            if (!library) {
                NSLog(@"Errore nel caricamento della libreria Metal: %@", error);
                return -2;
            }
            
            // Crea command queue
            commandQueue = [device newCommandQueue];
            if (!commandQueue) {
                NSLog(@"Errore nella creazione della command queue");
                return -3;
            }
            
            // Crea pipeline per matmul
            id<MTLFunction> matmulFunction = [library newFunctionWithName:@"exp_float_matmul"];
            matmulPipeline = [device newComputePipelineStateWithFunction:matmulFunction error:&error];
            if (!matmulPipeline) {
                NSLog(@"Errore nella creazione della pipeline matmul: %@", error);
                return -4;
            }
            
            // Crea pipeline per float_to_exp_float
            id<MTLFunction> floatToExpFunction = [library newFunctionWithName:@"float_to_exp_float"];
            floatToExpPipeline = [device newComputePipelineStateWithFunction:floatToExpFunction error:&error];
            if (!floatToExpPipeline) {
                NSLog(@"Errore nella creazione della pipeline float_to_exp: %@", error);
                return -5;
            }
            
            // Crea pipeline per exp_float_to_float
            id<MTLFunction> expToFloatFunction = [library newFunctionWithName:@"exp_float_to_float"];
            expToFloatPipeline = [device newComputePipelineStateWithFunction:expToFloatFunction error:&error];
            if (!expToFloatPipeline) {
                NSLog(@"Errore nella creazione della pipeline exp_to_float: %@", error);
                return -6;
            }
            
            NSLog(@"Metal inizializzato correttamente");
            return 0;
        }
    }
    
    // Implementazione di exp_float_matmul
    int exp_float_matmul(void* a_ptr, void* b_ptr, void* c_ptr, int M, int N, int K) {
        @autoreleasepool {
            if (!device || !commandQueue || !matmulPipeline) {
                NSLog(@"Metal non inizializzato");
                return -1;
            }
            
            // Crea buffer Metal direttamente dai puntatori PyTorch
            id<MTLBuffer> aBuffer = [device newBufferWithBytesNoCopy:a_ptr 
                                                             length:M * K * sizeof(char) 
                                                            options:MTLResourceStorageModeShared 
                                                        deallocator:nil];
            
            id<MTLBuffer> bBuffer = [device newBufferWithBytesNoCopy:b_ptr 
                                                             length:K * N * sizeof(char) 
                                                            options:MTLResourceStorageModeShared 
                                                        deallocator:nil];
            
            id<MTLBuffer> cBuffer = [device newBufferWithBytesNoCopy:c_ptr 
                                                             length:M * N * sizeof(char) 
                                                            options:MTLResourceStorageModeShared 
                                                        deallocator:nil];
            
            // Crea buffer per dimensioni
            id<MTLBuffer> mBuffer = [device newBufferWithBytes:&M length:sizeof(int) options:MTLResourceStorageModeShared];
            id<MTLBuffer> nBuffer = [device newBufferWithBytes:&N length:sizeof(int) options:MTLResourceStorageModeShared];
            id<MTLBuffer> kBuffer = [device newBufferWithBytes:&K length:sizeof(int) options:MTLResourceStorageModeShared];
            
            // Crea command buffer
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            
            // Crea compute command encoder
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
            [computeEncoder setComputePipelineState:matmulPipeline];
            
            // Set buffers
            [computeEncoder setBuffer:aBuffer offset:0 atIndex:0];
            [computeEncoder setBuffer:bBuffer offset:0 atIndex:1];
            [computeEncoder setBuffer:cBuffer offset:0 atIndex:2];
            [computeEncoder setBuffer:mBuffer offset:0 atIndex:3];
            [computeEncoder setBuffer:nBuffer offset:0 atIndex:4];
            [computeEncoder setBuffer:kBuffer offset:0 atIndex:5];
            
            // Calcola dimensioni grid e thread group
            MTLSize gridSize = MTLSizeMake(N, M, 1);
            MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
            
            // Dispatch
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
            [computeEncoder endEncoding];
            
            // Esegui
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            
            return 0;
        }
    }
    
    // Implementazione di float_to_exp_float
    int float_to_exp_float(void* input_ptr, void* output_ptr, int size) {
        @autoreleasepool {
            if (!device || !commandQueue || !floatToExpPipeline) {
                NSLog(@"Metal non inizializzato");
                return -1;
            }
            
            // Crea buffer
            id<MTLBuffer> inputBuffer = [device newBufferWithBytesNoCopy:input_ptr 
                                                                length:size * sizeof(float) 
                                                               options:MTLResourceStorageModeShared 
                                                           deallocator:nil];
            
            id<MTLBuffer> outputBuffer = [device newBufferWithBytesNoCopy:output_ptr 
                                                                 length:size * sizeof(char) 
                                                                options:MTLResourceStorageModeShared 
                                                            deallocator:nil];
            
            id<MTLBuffer> sizeBuffer = [device newBufferWithBytes:&size length:sizeof(int) options:MTLResourceStorageModeShared];
            
            // Crea command buffer ed encoder
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
            
            [computeEncoder setComputePipelineState:floatToExpPipeline];
            [computeEncoder setBuffer:inputBuffer offset:0 atIndex:0];
            [computeEncoder setBuffer:outputBuffer offset:0 atIndex:1];
            [computeEncoder setBuffer:sizeBuffer offset:0 atIndex:2];
            
            // Calcola dimensioni grid e thread group
            MTLSize gridSize = MTLSizeMake(size, 1, 1);
            int threadGroupSize = MIN(floatToExpPipeline.maxTotalThreadsPerThreadgroup, 256);
            MTLSize threadGroupSizeObj = MTLSizeMake(threadGroupSize, 1, 1);
            
            // Dispatch
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSizeObj];
            [computeEncoder endEncoding];
            
            // Esegui
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            
            return 0;
        }
    }
    
    // Implementazione di exp_float_to_float
    int exp_float_to_float(void* input_ptr, void* output_ptr, int size) {
        @autoreleasepool {
            if (!device || !commandQueue || !expToFloatPipeline) {
                NSLog(@"Metal non inizializzato");
                return -1;
            }
            
            // Crea buffer
            id<MTLBuffer> inputBuffer = [device newBufferWithBytesNoCopy:input_ptr 
                                                                length:size * sizeof(char) 
                                                               options:MTLResourceStorageModeShared 
                                                           deallocator:nil];
            
            id<MTLBuffer> outputBuffer = [device newBufferWithBytesNoCopy:output_ptr 
                                                                 length:size * sizeof(float) 
                                                                options:MTLResourceStorageModeShared 
                                                            deallocator:nil];
            
            id<MTLBuffer> sizeBuffer = [device newBufferWithBytes:&size length:sizeof(int) options:MTLResourceStorageModeShared];
            
            // Crea command buffer ed encoder
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
            
            [computeEncoder setComputePipelineState:expToFloatPipeline];
            [computeEncoder setBuffer:inputBuffer offset:0 atIndex:0];
            [computeEncoder setBuffer:outputBuffer offset:0 atIndex:1];
            [computeEncoder setBuffer:sizeBuffer offset:0 atIndex:2];
            
            // Calcola dimensioni grid e thread group
            MTLSize gridSize = MTLSizeMake(size, 1, 1);
            int threadGroupSize = MIN(expToFloatPipeline.maxTotalThreadsPerThreadgroup, 256);
            MTLSize threadGroupSizeObj = MTLSizeMake(threadGroupSize, 1, 1);
            
            // Dispatch
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSizeObj];
            [computeEncoder endEncoding];
            
            // Esegui
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            
            return 0;
        }
    }
    """
    
    return code

# Funzione per generare un file Makefile per la compilazione
def generate_makefile() -> str:
    """
    Genera un Makefile per compilare il codice di integrazione Metal
    """
    makefile = """
    # Makefile per libreria ExpFloat Metal

    CXX = clang++
    CXXFLAGS = -std=c++17 -O3 -Wall
    
    # Flags per framework Metal e PyTorch
    METAL_FLAGS = -framework Metal -framework Foundation -framework CoreGraphics
    
    # Trova i path di PyTorch
    TORCH_DIR = $(shell python -c 'import torch; print(torch.utils.cmake_prefix_path)')
    TORCH_INCLUDE = $(TORCH_DIR)/../../../include
    TORCH_LIB = $(TORCH_DIR)/../../../lib
    
    INCLUDES = -I$(TORCH_INCLUDE)
    LIBS = -L$(TORCH_LIB) -ltorch -ltorch_cpu
    
    # Target principale
    all: libexpfloat_metal.dylib
    
    # Compila la libreria dinamica
    libexpfloat_metal.dylib: metal_bridge.mm
    	$(CXX) $(CXXFLAGS) $(METAL_FLAGS) $(INCLUDES) -shared -o $@ $< $(LIBS)
    
    # Installa nella directory corrente
    install: libexpfloat_metal.dylib
    	cp libexpfloat_metal.dylib .
    
    # Pulisci
    clean:
    	rm -f libexpfloat_metal.dylib
    """
    
    return makefile

# Funzione per generare istruzioni di installazione
def generate_installation_guide() -> str:
    """
    Genera una guida all'installazione per l'integrazione Metal
    """
    guide = """
    # Guida all'installazione per ExpFloat con supporto Metal
    
    ## Requisiti
    - macOS 12.0 o superiore
    - Apple Silicon (M1/M2/M3) o GPU Metal compatibile
    - Xcode Command Line Tools
    - PyTorch 1.12 o superiore con supporto MPS
    
    ## Installazione
    
    1. Assicurati di avere PyTorch installato con supporto MPS:
       ```bash
       pip install torch torchvision
       ```
    
    2. Crea i file necessari per l'integrazione Metal:
       ```bash
       # Crea il file metal_shaders.metal
       cat > metal_shaders.metal << 'EOL'
       [CONTENUTO DEL KERNEL METAL]
       EOL
       
       # Crea il file metal_bridge.mm
       cat > metal_bridge.mm << 'EOL'
       [CONTENUTO DEL BRIDGE C/OBJECTIVE-C]
       EOL
       
       # Crea il Makefile
       cat > Makefile << 'EOL'
       [CONTENUTO DEL MAKEFILE]
       EOL
       ```
    
    3. Compila la libreria:
       ```bash
       make
       ```
    
    4. Installa nella directory corrente:
       ```bash
       make install
       ```
    
    5. Testa l'installazione:
       ```python
       import torch
       from metal_integration import MetalExpFloatBackend
       
       # Verifica che MPS sia disponibile
       print(f"MPS disponibile: {torch.backends.mps.is_available()}")
       
       # Inizializza il backend Metal
       backend = MetalExpFloatBackend()
       print(f"Backend Metal disponibile: {backend.is_available}")
       
       # Test su un tensore semplice
       if backend.is_available:
           x = torch.randn(10, 10, device='mps')
           x_exp = backend.float_to_exp(x)
           x_back = backend.exp_to_float(x_exp)
           error = torch.abs(x - x_back).mean()
           print(f"Errore medio: {error}")
       ```
    
    ## Risoluzione problemi
    
    - Se ricevi errori di compilazione relativi a framework mancanti, assicurati di avere Xcode e Command Line Tools installati:
      ```bash
      xcode-select --install
      ```
    
    - Se ricevi errori relativi a PyTorch, verifica che il path sia corretto nel Makefile.
    
    - Se il backend non è disponibile, verifica che MPS sia supportato sul tuo dispositivo:
      ```python
      import torch
      print(torch.backends.mps.is_built())
      print(torch.backends.mps.is_available())
      ```
    """
    
    return guide