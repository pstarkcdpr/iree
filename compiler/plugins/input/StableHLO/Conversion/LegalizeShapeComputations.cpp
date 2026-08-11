// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering StableHLO dialect to scalar shape operations.

#include "compiler/plugins/input/StableHLO/Conversion/MapStableHLOToScalarOp.h"
#include "compiler/plugins/input/StableHLO/Conversion/Passes.h"
#include "compiler/plugins/input/StableHLO/Conversion/Rewriters.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_LEGALIZESHAPECOMPUTATIONS
#include "compiler/plugins/input/StableHLO/Conversion/Passes.h.inc"

namespace {

// We assume that if one of the operands is a FromElements operation that means
// it is a shape computation.
bool opIsShapeComputation(Operation *op) {
  bool foundFromElements = false;
  for (Value operand : op->getOperands()) {
    auto shapedTy = cast<ShapedType>(operand.getType());
    if (!shapedTy.hasRank() || shapedTy.getRank() > 1) {
      return false;
    }
    if (auto fromElements = operand.getDefiningOp<tensor::FromElementsOp>()) {
      foundFromElements = true;
      continue;
    }
  }
  return foundFromElements;
}

template <typename OpTy>
struct HloElementwiseConverter : OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const final {
    if (!opIsShapeComputation(op)) {
      return failure();
    }

    auto resultTy = cast<ShapedType>(op.getType());

    Location loc = op.getLoc();
    SmallVector<Value> operands;
    for (int i = 0, s = resultTy.getNumElements(); i < s; ++i) {
      SmallVector<Value> extracts;
      for (Value operand : op->getOperands()) {
        ShapedType operandTy = cast<ShapedType>(operand.getType());
        if (operandTy.getRank() == 0) {
          Value extract =
              tensor::ExtractOp::create(rewriter, loc, operand, ValueRange({}));
          extracts.push_back(extract);
        } else {
          Value idx = arith::ConstantIndexOp::create(rewriter, loc, i);
          Value extract =
              tensor::ExtractOp::create(rewriter, loc, operand, idx);
          extracts.push_back(extract);
        }
      }

      Value scalarOp = mlir::stablehlo::StableHloOpToStdScalarOp::mapOp(
          op, resultTy.getElementType(), extracts, &rewriter);
      operands.push_back(scalarOp);
    }

    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, resultTy, operands);
    return success();
  }
};

struct ConcatenateConverter final
    : OpRewritePattern<mlir::stablehlo::ConcatenateOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    if (!opIsShapeComputation(op)) {
      return failure();
    }

    Location loc = op.getLoc();
    auto resultTy = cast<ShapedType>(op.getType());
    llvm::SmallVector<Value> elements;
    elements.reserve(resultTy.getNumElements());

    for (Value operand : op->getOperands()) {
      ShapedType operandTy = cast<ShapedType>(operand.getType());
      if (operandTy.getRank() == 0) {
        Value extract =
            tensor::ExtractOp::create(rewriter, loc, operand, ValueRange({}));
        elements.push_back(extract);
      } else {
        for (int i = 0, s = operandTy.getNumElements(); i < s; ++i) {
          Value idx = arith::ConstantIndexOp::create(rewriter, loc, i);
          Value extract =
              tensor::ExtractOp::create(rewriter, loc, operand, idx);
          elements.push_back(extract);
        }
      }
    }

    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, resultTy, elements);
    return success();
  }
};

// Rewrites a `dynamic_slice` of a 1-D `iota` into scalar index arithmetic.
// Because `iota[i] == i`, taking `k` elements starting at `s` produces the
// values `[s, s + 1, ..., s + k - 1]`, where `s` is the start index clamped to
// the valid range `[0, dim - k]`. This keeps an index that is already available
// as a scalar out of tensor land, where it would otherwise have to be computed
// on the device and read back.
struct IotaDynamicSliceConverter final
    : OpRewritePattern<mlir::stablehlo::DynamicSliceOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(mlir::stablehlo::DynamicSliceOp op,
                                PatternRewriter &rewriter) const override {
    if (!opIsShapeComputation(op)) {
      return failure();
    }

    auto iota = op.getOperand().getDefiningOp<mlir::stablehlo::IotaOp>();
    if (!iota) {
      return failure();
    }

    auto operandTy = dyn_cast<RankedTensorType>(iota.getType());
    auto resultTy = dyn_cast<RankedTensorType>(op.getType());
    if (!operandTy || !resultTy || operandTy.getRank() != 1 ||
        iota.getIotaDimension() != 0 || !operandTy.hasStaticShape() ||
        !resultTy.hasStaticShape()) {
      return failure();
    }

    // Only signless integers can be fed to the arith ops emitted below.
    Type elementTy = resultTy.getElementType();
    Value startIndex = op.getStartIndices().front();
    if (!elementTy.isSignlessInteger() ||
        !getElementTypeOrSelf(startIndex.getType()).isSignlessInteger()) {
      return failure();
    }

    Location loc = op.getLoc();
    int64_t sliceSize = resultTy.getDimSize(0);
    int64_t maxStart = operandTy.getDimSize(0) - sliceSize;

    // Clamp in `index` so that narrow start index types cannot overflow.
    Value start =
        tensor::ExtractOp::create(rewriter, loc, startIndex, ValueRange({}));
    Value startIdx = arith::IndexCastOp::create(rewriter, loc,
                                                rewriter.getIndexType(), start);
    Value lb = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value ub = arith::ConstantIndexOp::create(rewriter, loc, maxStart);
    Value clamped = arith::MaxSIOp::create(rewriter, loc, startIdx, lb);
    clamped = arith::MinSIOp::create(rewriter, loc, clamped, ub);

    SmallVector<Value> elements;
    elements.reserve(sliceSize);
    for (int64_t i = 0; i < sliceSize; ++i) {
      Value element = clamped;
      if (i != 0) {
        Value offset = arith::ConstantIndexOp::create(rewriter, loc, i);
        element = arith::AddIOp::create(rewriter, loc, element, offset);
      }
      elements.push_back(
          arith::IndexCastOp::create(rewriter, loc, elementTy, element));
    }

    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, resultTy, elements);
    return success();
  }
};

struct GetDimSizeConverter final
    : OpRewritePattern<mlir::stablehlo::GetDimensionSizeOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(mlir::stablehlo::GetDimensionSizeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resultTy = op.getType();
    Type elementTy = getElementTypeOrSelf(resultTy);
    IntegerAttr dimAttr = rewriter.getIndexAttr(op.getDimension());
    auto dimConst = arith::ConstantOp::create(rewriter, loc, dimAttr);

    Value dimOp = tensor::DimOp::create(rewriter, loc, rewriter.getIndexType(),
                                        op.getOperand(), dimConst);

    // Cast to the correct element type and convert to a tensor.
    Value cast = arith::IndexCastOp::create(rewriter, loc, elementTy, dimOp);
    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, resultTy, cast);
    return success();
  }
};

struct ReshapeConverter : OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    Value operand = op.getOperand();
    auto shapedTy = cast<ShapedType>(operand.getType());
    if (!shapedTy.hasRank() || shapedTy.getRank() > 1) {
      return failure();
    }

    auto resultTy = cast<ShapedType>(op.getType());

    auto fromElements = op.getOperand().getDefiningOp<tensor::FromElementsOp>();
    if (!fromElements) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(
        op, resultTy, fromElements.getOperands());
    return success();
  }
};

struct LegalizeShapeComputations final
    : impl::LegalizeShapeComputationsBase<LegalizeShapeComputations> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, math::MathDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext &ctx = this->getContext();
    RewritePatternSet patterns(&ctx);

    auto func = this->getOperation();
    populateLegalizeShapeComputationPatterns(&ctx, &patterns);
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      this->signalPassFailure();
    }
  }
};
} // namespace

void populateLegalizeShapeComputationPatterns(MLIRContext *context,
                                              RewritePatternSet *patterns) {
  patterns->add<HloElementwiseConverter<mlir::stablehlo::AbsOp>,
                HloElementwiseConverter<mlir::stablehlo::AddOp>,
                HloElementwiseConverter<mlir::stablehlo::AndOp>,
                HloElementwiseConverter<mlir::stablehlo::CeilOp>,
                HloElementwiseConverter<mlir::stablehlo::ClampOp>,
                HloElementwiseConverter<mlir::stablehlo::CompareOp>,
                HloElementwiseConverter<mlir::stablehlo::ConvertOp>,
                HloElementwiseConverter<mlir::stablehlo::DivOp>,
                HloElementwiseConverter<mlir::stablehlo::FloorOp>,
                HloElementwiseConverter<mlir::stablehlo::MaxOp>,
                HloElementwiseConverter<mlir::stablehlo::MinOp>,
                HloElementwiseConverter<mlir::stablehlo::MulOp>,
                HloElementwiseConverter<mlir::stablehlo::NegOp>,
                HloElementwiseConverter<mlir::stablehlo::RemOp>,
                HloElementwiseConverter<mlir::stablehlo::RoundOp>,
                HloElementwiseConverter<mlir::stablehlo::RsqrtOp>,
                HloElementwiseConverter<mlir::stablehlo::SelectOp>,
                HloElementwiseConverter<mlir::stablehlo::SqrtOp>,
                HloElementwiseConverter<mlir::stablehlo::SubtractOp>,
                ConcatenateConverter, GetDimSizeConverter,
                IotaDynamicSliceConverter, ReshapeConverter>(context);
}

} // namespace mlir::iree_compiler::stablehlo
