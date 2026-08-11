// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-stablehlo-legalize-shape-computations))" \
// RUN:          --split-input-file %s | FileCheck %s

// CHECK-LABEL: func @get_dimension_size
func.func @get_dimension_size(%arg0: tensor<?x?xf32>) -> (tensor<i32>) {
  %1 = "stablehlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1
// CHECK-DAG: %[[DIM:.+]] = tensor.dim %arg0, %[[C1]]
// CHECK-DAG: %[[IDX:.+]] = arith.index_cast %[[DIM]]
// CHECK-DAG: %[[FROM:.+]] = tensor.from_elements %[[IDX]]
// CHECK: return %[[FROM]] : tensor<i32>

// -----

// CHECK-LABEL: func @reshape_dimension_size
func.func @reshape_dimension_size(%arg0: tensor<?x?xf32>) -> (tensor<1xi32>) {
  %0 = "stablehlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
  %1 = "stablehlo.reshape"(%0) : (tensor<i32>) -> tensor<1xi32>
  func.return %1 : tensor<1xi32>
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1
// CHECK-DAG: %[[DIM:.+]] = tensor.dim %arg0, %[[C1]]
// CHECK-DAG: %[[IDX:.+]] = arith.index_cast %[[DIM]]
// CHECK-DAG: %[[FROM:.+]] = tensor.from_elements %[[IDX]]
// CHECK: return %[[FROM]] : tensor<1xi32>

// -----

// CHECK-LABEL: func @multiply_dimension_size
func.func @multiply_dimension_size(%arg0: tensor<?x?xf32>) -> (tensor<i32>) {
  %0 = stablehlo.constant dense<2> : tensor<i32>
  %1 = "stablehlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
  %2 = "stablehlo.multiply"(%0, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %2 : tensor<i32>
}


// CHECK-DAG: %[[C1:.+]] = arith.constant 1
// CHECK-DAG: %[[C2:.+]] = arith.constant 2
// CHECK-DAG: %[[DIM:.+]] = tensor.dim %arg0, %[[C1]]
// CHECK-DAG: %[[IDX:.+]] = arith.index_cast %[[DIM]]
// CHECK-DAG: %[[MUL:.+]] = arith.muli %[[IDX]], %[[C2]]
// CHECK-DAG: %[[RES:.+]] = tensor.from_elements %[[MUL]]
// CHECK: return %[[RES]]

// -----

// CHECK-LABEL: func @concat_dimension_size
func.func @concat_dimension_size(%arg0: tensor<?x?xf32>) -> (tensor<2xi32>) {
  %0 = "stablehlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
  %1 = "stablehlo.reshape"(%0) : (tensor<i32>) -> tensor<1xi32>
  %2 = stablehlo.constant dense<2> : tensor<1xi32>
  %3 = "stablehlo.concatenate"(%1, %2) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  func.return %3 : tensor<2xi32>
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1
// CHECK-DAG: %[[C2:.+]] = arith.constant 2
// CHECK-DAG: %[[DIM:.+]] = tensor.dim %arg0, %[[C1]]
// CHECK-DAG: %[[IDX:.+]] = arith.index_cast %[[DIM]]
// CHECK-DAG: %[[RES:.+]] = tensor.from_elements %[[IDX]], %[[C2]]
// CHECK: return %[[RES]]

// -----

// CHECK-LABEL: func @select_dimension_size
func.func @select_dimension_size(%arg0: tensor<?x?xf32>) -> (tensor<i32>) {
  %0 = "stablehlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
  %1 = stablehlo.constant dense<0> : tensor<i32>
  %2 = stablehlo.constant dense<7> : tensor<i32>
  %3 = stablehlo.compare LT, %0, %1, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %4 = stablehlo.select %3, %2, %0 : tensor<i1>, tensor<i32>
  func.return %4 : tensor<i32>
}

// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
// CHECK-DAG: %[[C7:.+]] = arith.constant 7 : i32
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[DIM:.+]] = tensor.dim %arg0, %[[C1]]
// CHECK-DAG: %[[IDX:.+]] = arith.index_cast %[[DIM]]
// CHECK-DAG: %[[CMP:.+]] = arith.cmpi slt, %[[IDX]], %[[C0]]
// CHECK-DAG: %[[SEL:.+]] = arith.select %[[CMP]], %[[C7]], %[[IDX]]
// CHECK-DAG: %[[RES:.+]] = tensor.from_elements %[[SEL]]
// CHECK: return %[[RES]]

// -----

// CHECK-LABEL: func @clamp_dimension_size
func.func @clamp_dimension_size(%arg0: tensor<?x?xf32>) -> (tensor<i32>) {
  %0 = "stablehlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<?x?xf32>) -> tensor<i32>
  %1 = stablehlo.constant dense<0> : tensor<i32>
  %2 = stablehlo.constant dense<4> : tensor<i32>
  %3 = stablehlo.clamp %1, %0, %2 : tensor<i32>
  func.return %3 : tensor<i32>
}

// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i32
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[DIM:.+]] = tensor.dim %arg0, %[[C1]]
// CHECK-DAG: %[[IDX:.+]] = arith.index_cast %[[DIM]]
// CHECK-DAG: %[[MAX:.+]] = arith.maxsi %[[IDX]], %[[C0]]
// CHECK-DAG: %[[MIN:.+]] = arith.minsi %[[MAX]], %[[C4]]
// CHECK-DAG: %[[RES:.+]] = tensor.from_elements %[[MIN]]
// CHECK: return %[[RES]]

// -----

// Slicing an iota is the index itself, clamped to the valid start range.
// CHECK-LABEL: func @dynamic_slice_of_iota
func.func @dynamic_slice_of_iota(%arg0: i32) -> (tensor<1xi32>) {
  %1 = tensor.from_elements %arg0 : tensor<i32>
  %2 = stablehlo.iota dim = 0 : tensor<4xi32>
  %3 = stablehlo.dynamic_slice %2, %1, sizes = [1] : (tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
  func.return %3 : tensor<1xi32>
}

// CHECK-NOT: stablehlo.iota
// CHECK-NOT: stablehlo.dynamic_slice
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG: %[[IDX:.+]] = arith.index_cast %arg0 : i32 to index
// CHECK-DAG: %[[MAX:.+]] = arith.maxsi %[[IDX]], %[[C0]]
// CHECK-DAG: %[[MIN:.+]] = arith.minsi %[[MAX]], %[[C3]]
// CHECK-DAG: %[[CAST:.+]] = arith.index_cast %[[MIN]] : index to i32
// CHECK-DAG: %[[RES:.+]] = tensor.from_elements %[[CAST]]
// CHECK: return %[[RES]]

// -----

// A multi-element slice of an iota yields consecutive indices.
// CHECK-LABEL: func @dynamic_slice_of_iota_multi
func.func @dynamic_slice_of_iota_multi(%arg0: i32) -> (tensor<2xi32>) {
  %1 = tensor.from_elements %arg0 : tensor<i32>
  %2 = stablehlo.iota dim = 0 : tensor<5xi32>
  %3 = stablehlo.dynamic_slice %2, %1, sizes = [2] : (tensor<5xi32>, tensor<i32>) -> tensor<2xi32>
  func.return %3 : tensor<2xi32>
}

// CHECK-NOT: stablehlo.dynamic_slice
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG: %[[IDX:.+]] = arith.index_cast %arg0 : i32 to index
// CHECK-DAG: %[[MAX:.+]] = arith.maxsi %[[IDX]], %[[C0]]
// CHECK-DAG: %[[MIN:.+]] = arith.minsi %[[MAX]], %[[C3]]
// CHECK-DAG: %[[ADD:.+]] = arith.addi %[[MIN]], %[[C1]]
// CHECK-DAG: %[[E0:.+]] = arith.index_cast %[[MIN]] : index to i32
// CHECK-DAG: %[[E1:.+]] = arith.index_cast %[[ADD]] : index to i32
// CHECK-DAG: %[[RES:.+]] = tensor.from_elements %[[E0]], %[[E1]]
// CHECK: return %[[RES]]

// -----

// A dynamic index that is not already a scalar must stay on the device;
// extracting it would introduce a device-to-host readback.
// CHECK-LABEL: func @dynamic_slice_of_iota_tensor_index
func.func @dynamic_slice_of_iota_tensor_index(%arg0: tensor<i32>) -> (tensor<1xi32>) {
  %0 = stablehlo.iota dim = 0 : tensor<4xi32>
  %1 = stablehlo.dynamic_slice %0, %arg0, sizes = [1] : (tensor<4xi32>, tensor<i32>) -> tensor<1xi32>
  func.return %1 : tensor<1xi32>
}

// CHECK: stablehlo.dynamic_slice
