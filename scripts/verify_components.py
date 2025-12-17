"""
Component-level verification script for SAM3 to TVM port.

This script compares PyTorch and TVM outputs for each component to verify correctness.
"""

import torch
import tvm
from tvm import relax
import numpy as np
import json
import sys
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict

# Add scripts directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import export_vision_backbone
import export_geometry_encoder
import export_transformer_encoder
import export_decoder
import export_segmentation_head
import export_scoring_head


@dataclass
class VerificationResult:
    """Results from comparing PyTorch and TVM outputs."""
    component_name: str
    passed: bool
    max_abs_error: float
    mean_abs_error: float
    max_rel_error: float
    rtol: float
    atol: float
    num_outputs: int
    output_shapes: List[tuple]
    error_message: str = ""


class ComponentVerifier:
    """Base class for component verification."""
    
    def __init__(self, rtol: float = 1e-5, atol: float = 1e-5):
        self.rtol = rtol
        self.atol = atol
    
    def to_numpy(self, t):
        """Convert tensor to numpy array."""
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        elif isinstance(t, tvm.nd.NDArray):
            return t.numpy()
        elif isinstance(t, np.ndarray):
            return t
        else:
            raise TypeError(f"Unsupported type: {type(t)}")
    
    def compare_outputs(self, pytorch_out, tvm_out) -> Tuple[bool, Dict[str, float]]:
        """
        Compare PyTorch and TVM outputs.
        
        Returns:
            (passed, metrics) where metrics contains max_abs_error, mean_abs_error, max_rel_error
        """
        # Handle tuple/list outputs
        if isinstance(pytorch_out, (tuple, list)):
            if not isinstance(tvm_out, (tuple, list)):
                return False, {"error": "Output type mismatch"}
            
            if len(pytorch_out) != len(tvm_out):
                return False, {"error": f"Output count mismatch: {len(pytorch_out)} vs {len(tvm_out)}"}
            
            all_passed = True
            max_abs_err = 0.0
            mean_abs_err = 0.0
            max_rel_err = 0.0
            
            for i, (pt, tv) in enumerate(zip(pytorch_out, tvm_out)):
                passed, metrics = self.compare_outputs(pt, tv)
                if not passed:
                    all_passed = False
                max_abs_err = max(max_abs_err, metrics.get("max_abs_error", 0))
                mean_abs_err += metrics.get("mean_abs_error", 0)
                max_rel_err = max(max_rel_err, metrics.get("max_rel_error", 0))
            
            mean_abs_err /= len(pytorch_out)
            return all_passed, {
                "max_abs_error": max_abs_err,
                "mean_abs_error": mean_abs_err,
                "max_rel_error": max_rel_err
            }
        
        # Convert to numpy
        pt_np = self.to_numpy(pytorch_out)
        tv_np = self.to_numpy(tvm_out)
        
        # Check shapes match
        if pt_np.shape != tv_np.shape:
            return False, {"error": f"Shape mismatch: {pt_np.shape} vs {tv_np.shape}"}
        
        # Compute errors
        abs_diff = np.abs(pt_np - tv_np)
        max_abs_error = np.max(abs_diff)
        mean_abs_error = np.mean(abs_diff)
        
        # Relative error (avoid division by zero)
        rel_diff = abs_diff / (np.abs(pt_np) + 1e-10)
        max_rel_error = np.max(rel_diff)
        
        # Check if within tolerance
        passed = np.allclose(pt_np, tv_np, rtol=self.rtol, atol=self.atol)
        
        return passed, {
            "max_abs_error": float(max_abs_error),
            "mean_abs_error": float(mean_abs_error),
            "max_rel_error": float(max_rel_error)
        }
    
    def get_output_shapes(self, output) -> List[tuple]:
        """Extract shapes from output (handles tuples/lists)."""
        if isinstance(output, (tuple, list)):
            shapes = []
            for item in output:
                shapes.extend(self.get_output_shapes(item))
            return shapes
        else:
            np_out = self.to_numpy(output)
            return [np_out.shape]
    
    def verify(self, component_name: str) -> VerificationResult:
        """Override this method in subclasses."""
        raise NotImplementedError


class VisionBackboneVerifier(ComponentVerifier):
    """Verify Vision Backbone component."""
    
    def verify(self, component_name: str = "Vision Backbone") -> VerificationResult:
        try:
            print(f"\n=== Verifying {component_name} ===")
            
            # Export and import
            print("Exporting PyTorch model...")
            exported_program = export_vision_backbone.export_with_dynamo()
            
            print("Importing to TVM...")
            tvm_mod = export_vision_backbone.import_to_tvm(exported_program)
            
            # Compile TVM module
            print("Compiling TVM module...")
            ex = relax.build(tvm_mod, target="llvm")
            vm = relax.VirtualMachine(ex, tvm.cpu())
            
            # Prepare input
            dummy_input = torch.randn(1, 3, 1008, 1008)
            normalized_input = (dummy_input - 0.5) / 0.5
            
            # Get PyTorch output
            print("Running PyTorch inference...")
            from sam3.model_builder import build_sam3_image_model
            import patch_rope
            import patch_vitdet
            patch_rope.apply_patches()
            patch_vitdet.apply_patches()
            
            model = build_sam3_image_model(
                checkpoint_path=None,
                eval_mode=True,
                load_from_HF=False,
            )
            wrapper = export_vision_backbone.VisionBackboneWrapper(model.backbone.vision_backbone)
            wrapper.eval()
            
            with torch.no_grad():
                pytorch_out = wrapper(normalized_input)
            
            # Get TVM output
            print("Running TVM inference...")
            tvm_input = tvm.nd.array(self.to_numpy(normalized_input))
            tvm_out = vm["main"](tvm_input)
            
            # Convert TVM tuple output to list
            tvm_out_list = [tvm_out[i] for i in range(len(pytorch_out))]
            
            # Compare outputs
            print("Comparing outputs...")
            passed, metrics = self.compare_outputs(pytorch_out, tvm_out_list)
            
            return VerificationResult(
                component_name=component_name,
                passed=passed,
                max_abs_error=metrics.get("max_abs_error", 0),
                mean_abs_error=metrics.get("mean_abs_error", 0),
                max_rel_error=metrics.get("max_rel_error", 0),
                rtol=self.rtol,
                atol=self.atol,
                num_outputs=len(pytorch_out),
                output_shapes=self.get_output_shapes(pytorch_out)
            )
            
        except Exception as e:
            import traceback
            return VerificationResult(
                component_name=component_name,
                passed=False,
                max_abs_error=0,
                mean_abs_error=0,
                max_rel_error=0,
                rtol=self.rtol,
                atol=self.atol,
                num_outputs=0,
                output_shapes=[],
                error_message=f"{str(e)}\n{traceback.format_exc()}"
            )


class GeometryEncoderVerifier(ComponentVerifier):
    """Verify Geometry Encoder component."""
    
    def verify(self, component_name: str = "Geometry Encoder") -> VerificationResult:
        try:
            print(f"\n=== Verifying {component_name} ===")
            
            # Export and import
            print("Exporting PyTorch model...")
            exported_program = export_geometry_encoder.export_geometry_encoder()
            
            print("Importing to TVM...")
            tvm_mod = export_geometry_encoder.import_to_tvm(exported_program)
            
            # Compile TVM module
            print("Compiling TVM module...")
            ex = relax.build(tvm_mod, target="llvm")
            vm = relax.VirtualMachine(ex, tvm.cpu())
            
            # Prepare inputs (same as export script)
            bs = 1
            n_points = 5
            n_boxes = 2
            H, W = 32, 32
            d_model = 256
            
            point_embeddings = torch.rand(n_points, bs, 2)
            point_mask = torch.zeros(bs, n_points, dtype=torch.bool)
            point_labels = torch.ones(n_points, bs, dtype=torch.long)
            
            box_embeddings = torch.rand(n_boxes, bs, 4)
            box_mask = torch.zeros(bs, n_boxes, dtype=torch.bool)
            box_labels = torch.ones(n_boxes, bs, dtype=torch.long)
            
            img_feats_last = torch.randn(H * W, bs, d_model)
            img_size_h = H
            img_size_w = W
            
            # Get PyTorch output
            print("Running PyTorch inference...")
            from sam3.model_builder import build_sam3_image_model
            import patch_rope
            patch_rope.apply_patches()
            
            model = build_sam3_image_model(
                checkpoint_path=None,
                eval_mode=True,
                load_from_HF=False,
            )
            wrapper = export_geometry_encoder.GeometryEncoderWrapper(model.geometry_encoder)
            wrapper.eval()
            
            with torch.no_grad():
                pytorch_out = wrapper(
                    point_embeddings, point_mask, point_labels,
                    box_embeddings, box_mask, box_labels,
                    img_feats_last, img_size_h, img_size_w
                )
            
            # Get TVM output
            print("Running TVM inference...")
            tvm_inputs = [
                tvm.nd.array(self.to_numpy(point_embeddings)),
                tvm.nd.array(self.to_numpy(point_mask)),
                tvm.nd.array(self.to_numpy(point_labels)),
                tvm.nd.array(self.to_numpy(box_embeddings)),
                tvm.nd.array(self.to_numpy(box_mask)),
                tvm.nd.array(self.to_numpy(box_labels)),
                tvm.nd.array(self.to_numpy(img_feats_last)),
                tvm.nd.array(np.array(img_size_h, dtype=np.int64)),
                tvm.nd.array(np.array(img_size_w, dtype=np.int64))
            ]
            tvm_out = vm["main"](*tvm_inputs)
            
            # Convert TVM tuple output to list
            tvm_out_list = [tvm_out[i] for i in range(len(pytorch_out))]
            
            # Compare outputs
            print("Comparing outputs...")
            passed, metrics = self.compare_outputs(pytorch_out, tvm_out_list)
            
            return VerificationResult(
                component_name=component_name,
                passed=passed,
                max_abs_error=metrics.get("max_abs_error", 0),
                mean_abs_error=metrics.get("mean_abs_error", 0),
                max_rel_error=metrics.get("max_rel_error", 0),
                rtol=self.rtol,
                atol=self.atol,
                num_outputs=len(pytorch_out),
                output_shapes=self.get_output_shapes(pytorch_out)
            )
            
        except Exception as e:
            import traceback
            return VerificationResult(
                component_name=component_name,
                passed=False,
                max_abs_error=0,
                mean_abs_error=0,
                max_rel_error=0,
                rtol=self.rtol,
                atol=self.atol,
                num_outputs=0,
                output_shapes=[],
                error_message=f"{str(e)}\n{traceback.format_exc()}"
            )


class TransformerEncoderVerifier(ComponentVerifier):
    """Verify Transformer Encoder component."""
    
    def verify(self, component_name: str = "Transformer Encoder") -> VerificationResult:
        try:
            print(f"\n=== Verifying {component_name} ===")
            
            # Export and import
            print("Exporting PyTorch model...")
            exported_program = export_transformer_encoder.export_transformer_encoder()
            
            print("Importing to TVM...")
            tvm_mod = export_transformer_encoder.import_to_tvm(exported_program)
            
            # Compile TVM module
            print("Compiling TVM module...")
            ex = relax.build(tvm_mod, target="llvm")
            vm = relax.VirtualMachine(ex, tvm.cpu())
            
            # Prepare inputs (same as export script)
            inputs = export_transformer_encoder.prepare_dummy_inputs()
            
            # Get PyTorch output
            print("Running PyTorch inference...")
            encoder = export_transformer_encoder.build_transformer_encoder()
            wrapper = export_transformer_encoder.TransformerEncoderWrapper(encoder)
            wrapper.eval()
            
            with torch.no_grad():
                pytorch_out = wrapper(*inputs)
            
            # Get TVM output
            print("Running TVM inference...")
            tvm_inputs = [tvm.nd.array(self.to_numpy(inp)) for inp in inputs]
            tvm_out = vm["main"](*tvm_inputs)
            
            # Convert TVM tuple output to list
            tvm_out_list = [tvm_out[i] for i in range(len(pytorch_out))]
            
            # Compare outputs
            print("Comparing outputs...")
            passed, metrics = self.compare_outputs(pytorch_out, tvm_out_list)
            
            return VerificationResult(
                component_name=component_name,
                passed=passed,
                max_abs_error=metrics.get("max_abs_error", 0),
                mean_abs_error=metrics.get("mean_abs_error", 0),
                max_rel_error=metrics.get("max_rel_error", 0),
                rtol=self.rtol,
                atol=self.atol,
                num_outputs=len(pytorch_out),
                output_shapes=self.get_output_shapes(pytorch_out)
            )
            
        except Exception as e:
            import traceback
            return VerificationResult(
                component_name=component_name,
                passed=False,
                max_abs_error=0,
                mean_abs_error=0,
                max_rel_error=0,
                rtol=self.rtol,
                atol=self.atol,
                num_outputs=0,
                output_shapes=[],
                error_message=f"{str(e)}\n{traceback.format_exc()}"
            )


class TransformerDecoderVerifier(ComponentVerifier):
    """Verify Transformer Decoder component."""
    
    def verify(self, component_name: str = "Transformer Decoder") -> VerificationResult:
        try:
            print(f"\n=== Verifying {component_name} ===")
            
            # Export and import
            print("Exporting PyTorch model...")
            exported_program = export_decoder.export_decoder()
            
            print("Importing to TVM...")
            tvm_mod = export_decoder.import_to_tvm(exported_program)
            
            # Compile TVM module
            print("Compiling TVM module...")
            ex = relax.build(tvm_mod, target="llvm")
            vm = relax.VirtualMachine(ex, tvm.cpu())
            
            # Prepare inputs
            inputs = export_decoder.prepare_dummy_inputs()
            # Remove spatial_shapes (5th element) for wrapper
            wrapper_inputs = inputs[:4] + inputs[5:]
            
            # Get PyTorch output
            print("Running PyTorch inference...")
            decoder = export_decoder.build_decoder()
            wrapper = export_decoder.TransformerDecoderWrapper(decoder, height=72, width=72)
            wrapper.eval()
            
            with torch.no_grad():
                pytorch_out = wrapper(*wrapper_inputs)
            
            # Get TVM output
            print("Running TVM inference...")
            tvm_inputs = [tvm.nd.array(self.to_numpy(inp)) for inp in wrapper_inputs]
            tvm_out = vm["main"](*tvm_inputs)
            
            # Convert TVM tuple output to list
            tvm_out_list = [tvm_out[i] for i in range(len(pytorch_out))]
            
            # Compare outputs
            print("Comparing outputs...")
            passed, metrics = self.compare_outputs(pytorch_out, tvm_out_list)
            
            return VerificationResult(
                component_name=component_name,
                passed=passed,
                max_abs_error=metrics.get("max_abs_error", 0),
                mean_abs_error=metrics.get("mean_abs_error", 0),
                max_rel_error=metrics.get("max_rel_error", 0),
                rtol=self.rtol,
                atol=self.atol,
                num_outputs=len(pytorch_out),
                output_shapes=self.get_output_shapes(pytorch_out)
            )
            
        except Exception as e:
            import traceback
            return VerificationResult(
                component_name=component_name,
                passed=False,
                max_abs_error=0,
                mean_abs_error=0,
                max_rel_error=0,
                rtol=self.rtol,
                atol=self.atol,
                num_outputs=0,
                output_shapes=[],
                error_message=f"{str(e)}\n{traceback.format_exc()}"
            )


class SegmentationHeadVerifier(ComponentVerifier):
    """Verify Segmentation Head component."""
    
    def verify(self, component_name: str = "Segmentation Head") -> VerificationResult:
        try:
            print(f"\n=== Verifying {component_name} ===")
            
            # Export and import
            print("Exporting PyTorch model...")
            exported_program = export_segmentation_head.export_segmentation_head()
            
            print("Importing to TVM...")
            tvm_mod = export_segmentation_head.import_to_tvm(exported_program)
            
            # Compile TVM module
            print("Compiling TVM module...")
            ex = relax.build(tvm_mod, target="llvm")
            vm = relax.VirtualMachine(ex, tvm.cpu())
            
            # Prepare inputs (same as export script)
            inputs = export_segmentation_head.prepare_dummy_inputs()
            
            # Get PyTorch output
            print("Running PyTorch inference...")
            from sam3.model_builder import build_sam3_image_model
            model = build_sam3_image_model(
                checkpoint_path=None,
                eval_mode=True,
                load_from_HF=False,
            )
            wrapper = export_segmentation_head.SegmentationHeadWrapper(model.segmentation_head)
            wrapper.eval()
            
            with torch.no_grad():
                pytorch_out = wrapper(*inputs)
            
            # Get TVM output
            print("Running TVM inference...")
            tvm_inputs = [tvm.nd.array(self.to_numpy(inp)) for inp in inputs]
            tvm_out = vm["main"](*tvm_inputs)
            
            # Convert TVM tuple output to list
            tvm_out_list = [tvm_out[i] for i in range(len(pytorch_out))]
            
            # Compare outputs
            print("Comparing outputs...")
            passed, metrics = self.compare_outputs(pytorch_out, tvm_out_list)
            
            return VerificationResult(
                component_name=component_name,
                passed=passed,
                max_abs_error=metrics.get("max_abs_error", 0),
                mean_abs_error=metrics.get("mean_abs_error", 0),
                max_rel_error=metrics.get("max_rel_error", 0),
                rtol=self.rtol,
                atol=self.atol,
                num_outputs=len(pytorch_out),
                output_shapes=self.get_output_shapes(pytorch_out)
            )
            
        except Exception as e:
            import traceback
            return VerificationResult(
                component_name=component_name,
                passed=False,
                max_abs_error=0,
                mean_abs_error=0,
                max_rel_error=0,
                rtol=self.rtol,
                atol=self.atol,
                num_outputs=0,
                output_shapes=[],
                error_message=f"{str(e)}\n{traceback.format_exc()}"
            )


class ScoringHeadVerifier(ComponentVerifier):
    """Verify Scoring Head component."""
    
    def verify(self, component_name: str = "Scoring Head") -> VerificationResult:
        try:
            print(f"\n=== Verifying {component_name} ===")
            
            # Export and import
            print("Exporting PyTorch model...")
            exported_program = export_scoring_head.export_scoring_head()
            
            print("Importing to TVM...")
            tvm_mod = export_scoring_head.import_to_tvm(exported_program)
            
            # Compile TVM module
            print("Compiling TVM module...")
            ex = relax.build(tvm_mod, target="llvm")
            vm = relax.VirtualMachine(ex, tvm.cpu())
            
            # Prepare inputs (same as export script)
            inputs = export_scoring_head.prepare_dummy_inputs()
            
            # Get PyTorch output
            print("Running PyTorch inference...")
            from sam3.model_builder import build_sam3_image_model
            model = build_sam3_image_model(
                checkpoint_path=None,
                eval_mode=True,
                load_from_HF=False,
            )
            wrapper = export_scoring_head.ScoringHeadWrapper(model.scoring_head)
            wrapper.eval()
            
            with torch.no_grad():
                pytorch_out = wrapper(*inputs)
            
            # Get TVM output
            print("Running TVM inference...")
            tvm_inputs = [tvm.nd.array(self.to_numpy(inp)) for inp in inputs]
            tvm_out = vm["main"](*tvm_inputs)
            
            # Compare outputs (scoring head returns single tensor)
            print("Comparing outputs...")
            passed, metrics = self.compare_outputs(pytorch_out, tvm_out)
            
            return VerificationResult(
                component_name=component_name,
                passed=passed,
                max_abs_error=metrics.get("max_abs_error", 0),
                mean_abs_error=metrics.get("mean_abs_error", 0),
                max_rel_error=metrics.get("max_rel_error", 0),
                rtol=self.rtol,
                atol=self.atol,
                num_outputs=1,
                output_shapes=self.get_output_shapes(pytorch_out)
            )
            
        except Exception as e:
            import traceback
            return VerificationResult(
                component_name=component_name,
                passed=False,
                max_abs_error=0,
                mean_abs_error=0,
                max_rel_error=0,
                rtol=self.rtol,
                atol=self.atol,
                num_outputs=0,
                output_shapes=[],
                error_message=f"{str(e)}\n{traceback.format_exc()}"
            )


def main():
    """Run verification for all components."""
    print("=== SAM3 Component Verification ===")
    print(f"Tolerance: rtol=1e-5, atol=1e-5\n")
    
    results = []
    
    # Verify Vision Backbone
    print("\n" + "="*60)
    verifier = VisionBackboneVerifier()
    result = verifier.verify()
    results.append(result)
    print_result(result)
    
    # Verify Geometry Encoder
    print("\n" + "="*60)
    verifier = GeometryEncoderVerifier()
    result = verifier.verify()
    results.append(result)
    print_result(result)
    
    # Verify Transformer Encoder
    print("\n" + "="*60)
    verifier = TransformerEncoderVerifier()
    result = verifier.verify()
    results.append(result)
    print_result(result)
    
    # Verify Transformer Decoder
    print("\n" + "="*60)
    verifier = TransformerDecoderVerifier()
    result = verifier.verify()
    results.append(result)
    print_result(result)
    
    # Verify Segmentation Head
    print("\n" + "="*60)
    verifier = SegmentationHeadVerifier()
    result = verifier.verify()
    results.append(result)
    print_result(result)
    
    # Verify Scoring Head
    print("\n" + "="*60)
    verifier = ScoringHeadVerifier()
    result = verifier.verify()
    results.append(result)
    print_result(result)
    
    # Save results to JSON
    save_results(results)
    
    # Print summary
    print_summary(results)


def print_result(result: VerificationResult):
    """Print verification result."""
    status = "✓ PASSED" if result.passed else "✗ FAILED"
    print(f"\n{status}: {result.component_name}")
    
    if result.error_message:
        print(f"  Error: {result.error_message}")
    else:
        print(f"  Outputs: {result.num_outputs}")
        print(f"  Max Abs Error: {result.max_abs_error:.2e}")
        print(f"  Mean Abs Error: {result.mean_abs_error:.2e}")
        print(f"  Max Rel Error: {result.max_rel_error:.2e}")


def save_results(results: List[VerificationResult]):
    """Save results to JSON file."""
    output_file = "verification_report.json"
    
    results_dict = {
        "results": [asdict(r) for r in results],
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed)
        }
    }
    
    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")


def print_summary(results: List[VerificationResult]):
    """Print summary of all results."""
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    
    print(f"Total Components: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed Components:")
        for r in results:
            if not r.passed:
                print(f"  - {r.component_name}")
                if r.error_message:
                    print(f"    Error: {r.error_message[:100]}...")


if __name__ == "__main__":
    main()
