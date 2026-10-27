// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-virtual-vector-lowering))" --split-input-file %s | FileCheck %s

// For an n-D row-major contiguous memref, the gather lowering should add the
// gather index directly into the innermost offset rather than emit a per-lane
// affine.linearize_index/affine.delinearize_index pair.

func.func @gather_contiguous_3d(%buffer: memref<50x40x40xi8>,
                                %idx: vector<8xindex>,
                                %mask: vector<8xi1>,
                                %pass: vector<8xi8>,
                                %i: index, %j: index, %k: index)
    -> vector<8xi8> {
  %r = vector.gather %buffer[%i, %j, %k] [%idx], %mask, %pass
     : memref<50x40x40xi8>, vector<8xindex>, vector<8xi1>, vector<8xi8>
       into vector<8xi8>
  return %r : vector<8xi8>
}

// CHECK-LABEL:   func.func @gather_contiguous_3d(
// CHECK-SAME:        %[[BUF:[^:]+]]: memref<50x40x40xi8>
// CHECK-SAME:        %[[I:[^:]+]]: index, %[[J:[^:]+]]: index, %[[K:[^:]+]]: index
// CHECK-NOT:   vector.gather
// CHECK-NOT:   affine.linearize_index
// CHECK-NOT:   affine.delinearize_index
// CHECK:       arith.addi %[[K]],
// CHECK:       vector.load %[[BUF]][%[[I]], %[[J]], %{{.+}}] : memref<50x40x40xi8>, vector<1xi8>

// -----

// For a non-contiguous (strided, non-row-major) memref, the gather lowering
// must keep the linearize/delinearize path because the per-dimension and
// linearized addresses no longer agree.

func.func @gather_non_contiguous_2d(
    %buffer: memref<10x20xf32, strided<[40, 1]>>,
    %idx: vector<4xindex>, %mask: vector<4xi1>, %pass: vector<4xf32>,
    %i: index, %j: index) -> vector<4xf32> {
  %r = vector.gather %buffer[%i, %j] [%idx], %mask, %pass
     : memref<10x20xf32, strided<[40, 1]>>, vector<4xindex>,
       vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %r : vector<4xf32>
}

// CHECK-LABEL:   func.func @gather_non_contiguous_2d(
// CHECK-NOT:   vector.gather
// CHECK:       affine.linearize_index
// CHECK:       affine.delinearize_index

// -----

// Dynamic-shape identity-layout memref also qualifies as row-major contiguous,
// so the gather lowering should still take the contiguous path even though
// `memref::isStaticShapeAndContiguousRowMajor` returns false here.

func.func @gather_dynamic_identity_2d(%buffer: memref<?x?xf32>,
                                      %idx: vector<4xindex>,
                                      %mask: vector<4xi1>,
                                      %pass: vector<4xf32>,
                                      %i: index, %j: index) -> vector<4xf32> {
  %r = vector.gather %buffer[%i, %j] [%idx], %mask, %pass
     : memref<?x?xf32>, vector<4xindex>, vector<4xi1>, vector<4xf32>
       into vector<4xf32>
  return %r : vector<4xf32>
}

// CHECK-LABEL:   func.func @gather_dynamic_identity_2d(
// CHECK-SAME:        %[[BUF:[^:]+]]: memref<?x?xf32>
// CHECK-SAME:        %[[I:[^:]+]]: index, %[[J:[^:]+]]: index
// CHECK-NOT:   vector.gather
// CHECK-NOT:   affine.linearize_index
// CHECK-NOT:   affine.delinearize_index
// CHECK:       arith.addi %[[J]],
// CHECK:       vector.load %[[BUF]][%[[I]], %{{.+}}] : memref<?x?xf32>, vector<1xf32>

// -----

// A static-shape memref with an explicit strided<[N, 1]> layout whose strides
// match the row-major contiguous sequence should also take the contiguous
// path via `isStaticShapeAndContiguousRowMajor`.

func.func @gather_strided_contiguous_2d(
    %buffer: memref<10x20xf32, strided<[20, 1]>>,
    %idx: vector<4xindex>, %mask: vector<4xi1>, %pass: vector<4xf32>,
    %i: index, %j: index) -> vector<4xf32> {
  %r = vector.gather %buffer[%i, %j] [%idx], %mask, %pass
     : memref<10x20xf32, strided<[20, 1]>>, vector<4xindex>,
       vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %r : vector<4xf32>
}

// CHECK-LABEL:   func.func @gather_strided_contiguous_2d(
// CHECK-SAME:        %[[BUF:[^:]+]]: memref<10x20xf32, strided<[20, 1]>>
// CHECK-SAME:        %[[I:[^:]+]]: index, %[[J:[^:]+]]: index
// CHECK-NOT:   vector.gather
// CHECK-NOT:   affine.linearize_index
// CHECK-NOT:   affine.delinearize_index
// CHECK:       arith.addi %[[J]],
// CHECK:       vector.load %[[BUF]][%[[I]], %{{.+}}] : memref<10x20xf32, strided<[20, 1]>>, vector<1xf32>

// -----

// Tensor base falls through to the upstream pattern, which lowers each lane
// to a `tensor.extract`. The IREE-local pattern does not match (memref-only)
// and must not interfere.

func.func @gather_tensor_base_2d(%buffer: tensor<10x20xf32>,
                                 %idx: vector<4xindex>,
                                 %mask: vector<4xi1>,
                                 %pass: vector<4xf32>,
                                 %i: index, %j: index) -> vector<4xf32> {
  %r = vector.gather %buffer[%i, %j] [%idx], %mask, %pass
     : tensor<10x20xf32>, vector<4xindex>, vector<4xi1>, vector<4xf32>
       into vector<4xf32>
  return %r : vector<4xf32>
}

// CHECK-LABEL:   func.func @gather_tensor_base_2d(
// CHECK-SAME:        %[[BUF:[^:]+]]: tensor<10x20xf32>
// CHECK-NOT:   vector.gather
// CHECK:       tensor.extract %[[BUF]]

// -----

// A matmat-like contraction whose lhs is indexed by the second parallel
// dimension and whose rhs is indexed by the first one (the form a batched 1x1
// convolution takes after unit dims are dropped). It should lower to the same
// broadcast + FMA sequence as a canonical matmul: rows of the lhs tile are used
// directly, and the elements of the rhs become scalar broadcasts, without any
// transposition of either operand.

func.func @contract_lhs_kn_rhs_mk(%x: memref<100x128x32xf32>,
                                  %w: memref<128x128xf32>,
                                  %acc: vector<8x8xf32>, %b: index, %i: index,
                                  %o: index, %wi: index) -> vector<8x8xf32> {
  %cst = arith.constant 0.0 : f32
  %lhs = vector.transfer_read %x[%b, %i, %wi], %cst {in_bounds = [true, true]}
    : memref<100x128x32xf32>, vector<8x8xf32>
  %rhs = vector.transfer_read %w[%o, %i], %cst {in_bounds = [true, true]}
    : memref<128x128xf32>, vector<8x8xf32>
  %0 = vector.contract {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// CHECK-LABEL:   func.func @contract_lhs_kn_rhs_mk(
// CHECK-SAME:        %[[X:[a-zA-Z0-9]+]]: memref<100x128x32xf32>
// CHECK-SAME:        %[[W:[a-zA-Z0-9]+]]: memref<128x128xf32>
// CHECK-NOT:     vector.transpose
// CHECK:         %[[XT:.+]] = vector.transfer_read %[[X]]
// CHECK:         %[[XROW0:.+]] = vector.extract %[[XT]][0] : vector<8xf32> from vector<8x8xf32>
// CHECK:         %[[W0:.+]] = memref.load %[[W]]
// CHECK:         %[[B0:.+]] = vector.broadcast %[[W0]] : f32 to vector<8xf32>
// CHECK:         vector.fma %[[B0]], %[[XROW0]], %{{.+}} : vector<8xf32>
// CHECK-COUNT-63: vector.fma %{{.+}}, %{{.+}}, %{{.+}} : vector<8xf32>
// CHECK-NOT:     vector.fma
// CHECK-NOT:     vector.transpose
// CHECK-NOT:     vector.from_elements
// CHECK:         return

// -----

// Same as above, but with the reduction dimension innermost in both operands.
// After the operands are swapped this is the (m, k) x (n, k) layout, which
// needs one transposition of the original lhs tile.

func.func @contract_lhs_nk_rhs_mk(%x: memref<100x32x128xf32>,
                                  %w: memref<128x128xf32>,
                                  %acc: vector<8x8xf32>, %b: index, %i: index,
                                  %o: index, %wi: index) -> vector<8x8xf32> {
  %cst = arith.constant 0.0 : f32
  %lhs = vector.transfer_read %x[%b, %wi, %i], %cst {in_bounds = [true, true]}
    : memref<100x32x128xf32>, vector<8x8xf32>
  %rhs = vector.transfer_read %w[%o, %i], %cst {in_bounds = [true, true]}
    : memref<128x128xf32>, vector<8x8xf32>
  %0 = vector.contract {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc
    : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// CHECK-LABEL:   func.func @contract_lhs_nk_rhs_mk(
// CHECK-SAME:        %[[X:[a-zA-Z0-9]+]]: memref<100x32x128xf32>
// CHECK-SAME:        %[[W:[a-zA-Z0-9]+]]: memref<128x128xf32>
// CHECK:         %[[XT:.+]] = vector.transfer_read %[[X]]
// CHECK:         %[[XTT:.+]] = vector.transpose %[[XT]], [1, 0] : vector<8x8xf32> to vector<8x8xf32>
// CHECK:         %[[XROW0:.+]] = vector.extract %[[XTT]][0] : vector<8xf32> from vector<8x8xf32>
// CHECK:         %[[W0:.+]] = memref.load %[[W]]
// CHECK:         %[[B0:.+]] = vector.broadcast %[[W0]] : f32 to vector<8xf32>
// CHECK:         vector.fma %[[B0]], %[[XROW0]], %{{.+}} : vector<8xf32>
// CHECK-COUNT-63: vector.fma %{{.+}}, %{{.+}}, %{{.+}} : vector<8xf32>
// CHECK-NOT:     vector.fma
// CHECK-NOT:     vector.transpose
// CHECK-NOT:     vector.from_elements
// CHECK:         return
