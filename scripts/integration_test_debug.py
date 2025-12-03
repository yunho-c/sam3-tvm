"""
Debug version of integration_test.py with detailed error tracing.
Helps pinpoint exactly where the typebool error occurs.
"""

import torch
import tvm
from tvm import relax
import numpy as np
import sys
import os
import traceback

# Add scripts directory to path to import export modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import export_vision_backbone
import export_geometry_encoder
import export_transformer_encoder
import export_decoder
import export_segmentation_head
import export_scoring_head

import patch_rope
import tvm_custom_ops

def safe_import_to_tvm(component_name, exported_program, import_func):
    """
    Safely import a component to TVM with detailed error reporting.
    
    Args:
        component_name: Name of the component for logging
        exported_program: The exported PyTorch program
        import_func: The import function to call
    
    Returns:
        The TVM module or None if import failed
    """
    print(f"\n{'='*60}")
    print(f"[DEBUG] Importing {component_name} to TVM Relax")
    print(f"{'='*60}")
    
    try:
        # Call the import function
        mod = import_func(exported_program)
        
        # Save IR to file for inspection
        ir_filename = f"{component_name.lower().replace(' ', '_')}_debug_ir.txt"
        with open(ir_filename, 'w') as f:
            f.write(str(mod))
        print(f"✓ Successfully imported {component_name}")
        print(f"  IR saved to: {ir_filename}")
        
        return mod
        
    except Exception as e:
        print(f"\n{'!'*60}")
        print(f"✗ FAILED to import {component_name}")
        print(f"{'!'*60}")
        print(f"\nError Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print(f"\nFull Traceback:")
        print("-" * 60)
        traceback.print_exc()
        print("-" * 60)
        
        # Try to extract more information from the error
        if "typebool" in str(e).lower():
            print(f"\n⚠️  TYPEBOOL ERROR DETECTED in {component_name}!")
            print("This component is the source of the typebool error.")
            
        return None

def compile_module(mod, target="llvm", component_name="Unknown"):
    """Compile a Relax module to a TVM VM with error handling."""
    if mod is None:
        print(f"[DEBUG] Skipping compilation for {component_name} (module is None)")
        return None
        
    print(f"[DEBUG] Compiling {component_name}...")
    try:
        ex = relax.build(mod, target=target)
        vm = relax.VirtualMachine(ex, tvm.cpu())
        print(f"✓ Successfully compiled {component_name}")
        return vm
    except Exception as e:
        print(f"✗ FAILED to compile {component_name}")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return None

def to_numpy(t):
    return t.detach().cpu().numpy()

def main():
    print("="*60)
    print("SAM3 TVM Integration Test - DEBUG MODE")
    print("="*60)
    print("\nThis script will:")
    print("1. Import each component separately with error catching")
    print("2. Save IR to files for inspection")
    print("3. Identify exactly which component causes the typebool error")
    print("="*60)
    
    # Track which components succeed
    results = {}
    
    # 1. Export and Import Vision Backbone
    print("\n\n[1/6] Processing Vision Backbone...")
    try:
        prog_vision = export_vision_backbone.export_with_dynamo()
        mod_vision = safe_import_to_tvm(
            "Vision Backbone",
            prog_vision,
            export_vision_backbone.import_to_tvm
        )
        vm_vision = compile_module(mod_vision, component_name="Vision Backbone")
        results["Vision Backbone"] = (mod_vision is not None, vm_vision is not None)
    except Exception as e:
        print(f"✗ Export failed for Vision Backbone: {e}")
        results["Vision Backbone"] = (False, False)
    
    # 2. Export and Import Geometry Encoder
    print("\n\n[2/6] Processing Geometry Encoder...")
    try:
        prog_geo = export_geometry_encoder.export_geometry_encoder()
        mod_geo = safe_import_to_tvm(
            "Geometry Encoder",
            prog_geo,
            export_geometry_encoder.import_to_tvm
        )
        vm_geo = compile_module(mod_geo, component_name="Geometry Encoder")
        results["Geometry Encoder"] = (mod_geo is not None, vm_geo is not None)
    except Exception as e:
        print(f"✗ Export failed for Geometry Encoder: {e}")
        results["Geometry Encoder"] = (False, False)
    
    # 3. Export and Import Transformer Encoder
    print("\n\n[3/6] Processing Transformer Encoder...")
    try:
        prog_enc = export_transformer_encoder.export_transformer_encoder()
        mod_enc = safe_import_to_tvm(
            "Transformer Encoder",
            prog_enc,
            export_transformer_encoder.import_to_tvm
        )
        vm_enc = compile_module(mod_enc, component_name="Transformer Encoder")
        results["Transformer Encoder"] = (mod_enc is not None, vm_enc is not None)
    except Exception as e:
        print(f"✗ Export failed for Transformer Encoder: {e}")
        results["Transformer Encoder"] = (False, False)
    
    # 4. Export and Import Transformer Decoder
    print("\n\n[4/6] Processing Transformer Decoder...")
    try:
        prog_dec = export_decoder.export_decoder()
        mod_dec = safe_import_to_tvm(
            "Transformer Decoder",
            prog_dec,
            export_decoder.import_to_tvm
        )
        vm_dec = compile_module(mod_dec, component_name="Transformer Decoder")
        results["Transformer Decoder"] = (mod_dec is not None, vm_dec is not None)
    except Exception as e:
        print(f"✗ Export failed for Transformer Decoder: {e}")
        results["Transformer Decoder"] = (False, False)
    
    # 5. Export and Import Segmentation Head
    print("\n\n[5/6] Processing Segmentation Head...")
    try:
        prog_seg = export_segmentation_head.export_segmentation_head()
        mod_seg = safe_import_to_tvm(
            "Segmentation Head",
            prog_seg,
            export_segmentation_head.import_to_tvm
        )
        vm_seg = compile_module(mod_seg, component_name="Segmentation Head")
        results["Segmentation Head"] = (mod_seg is not None, vm_seg is not None)
    except Exception as e:
        print(f"✗ Export failed for Segmentation Head: {e}")
        results["Segmentation Head"] = (False, False)
    
    # 6. Export and Import Scoring Head
    print("\n\n[6/6] Processing Scoring Head...")
    try:
        prog_score = export_scoring_head.export_scoring_head()
        mod_score = safe_import_to_tvm(
            "Scoring Head",
            prog_score,
            export_scoring_head.import_to_tvm
        )
        vm_score = compile_module(mod_score, component_name="Scoring Head")
        results["Scoring Head"] = (mod_score is not None, vm_score is not None)
    except Exception as e:
        print(f"✗ Export failed for Scoring Head: {e}")
        results["Scoring Head"] = (False, False)
    
    # Print summary
    print("\n\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    print(f"{'Component':<25} {'Import':<10} {'Compile':<10}")
    print("-"*60)
    
    for component, (imported, compiled) in results.items():
        import_status = "✓ PASS" if imported else "✗ FAIL"
        compile_status = "✓ PASS" if compiled else "✗ FAIL"
        print(f"{component:<25} {import_status:<10} {compile_status:<10}")
    
    print("="*60)
    
    # Identify the problematic component
    failed_components = [comp for comp, (imported, _) in results.items() if not imported]
    if failed_components:
        print(f"\n⚠️  Components that failed TVM import:")
        for comp in failed_components:
            print(f"   - {comp}")
        print("\nCheck the debug IR files and error messages above for details.")
    else:
        print("\n✓ All components imported successfully!")
        print("The typebool error might be occurring during compilation or runtime.")

if __name__ == "__main__":
    main()
