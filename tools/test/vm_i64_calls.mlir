// Tests that 64-bit values survive calls to non-inlined internal functions.
//
// Unlike imports, internal calls marshal registers directly between the caller
// and callee frames (iree_vm_bytecode_internal_enter / _leave in
// runtime/src/iree/vm/bytecode/dispatch.c). A 64-bit value is named by a single
// register ordinal but occupies that slot and the next, and the callee's ABI
// registers align 64-bit values to an even ordinal. Marshalling one 32-bit word
// per register-list entry drops each high word and shifts the following
// operand's low word into its place.

// RUN: (iree-compile --compile-mode=vm %s | \
// RUN:  iree-run-module --device=local-sync --module=- --function=call_i64 \
// RUN:                  --input=11 --input=22 --input=33) | \
// RUN: FileCheck %s --check-prefix=I64

// RUN: (iree-compile --compile-mode=vm %s | \
// RUN:  iree-run-module --device=local-sync --module=- --function=call_mixed \
// RUN:                  --input=1 --input=8589934592 --input=3 --input=-4) | \
// RUN: FileCheck %s --check-prefix=MIXED

vm.module @vm_i64_calls {

  vm.func private @pass_i64(%a : i64, %b : i64, %c : i64) -> (i64, i64, i64)
      attributes {inlining_policy = #util.inline.never} {
    vm.return %a, %b, %c : i64, i64, i64
  }

  // I64-LABEL: EXEC @call_i64
  //  I64-NEXT: result[0]: i64=11
  //  I64-NEXT: result[1]: i64=22
  //  I64-NEXT: result[2]: i64=33
  vm.export @call_i64
  vm.func @call_i64(%a : i64, %b : i64, %c : i64) -> (i64, i64, i64) {
    %r:3 = vm.call @pass_i64(%a, %b, %c) : (i64, i64, i64) -> (i64, i64, i64)
    vm.return %r#0, %r#1, %r#2 : i64, i64, i64
  }

  // Interleaving 32-bit and 64-bit values forces the callee's 64-bit ABI
  // registers to be padded up to an even ordinal.
  vm.func private @pass_mixed(%a : i32, %b : i64, %c : i32, %d : i64)
      -> (i32, i64, i32, i64)
      attributes {inlining_policy = #util.inline.never} {
    vm.return %a, %b, %c, %d : i32, i64, i32, i64
  }

  // 8589934592 is 2^33: its low word is zero, so a dropped high word reads
  // back as 0. -4 covers the all-ones high word.
  // MIXED-LABEL: EXEC @call_mixed
  //  MIXED-NEXT: result[0]: i32=1
  //  MIXED-NEXT: result[1]: i64=8589934592
  //  MIXED-NEXT: result[2]: i32=3
  //  MIXED-NEXT: result[3]: i64=-4
  vm.export @call_mixed
  vm.func @call_mixed(%a : i32, %b : i64, %c : i32, %d : i64)
      -> (i32, i64, i32, i64) {
    %r:4 = vm.call @pass_mixed(%a, %b, %c, %d)
        : (i32, i64, i32, i64) -> (i32, i64, i32, i64)
    vm.return %r#0, %r#1, %r#2, %r#3 : i32, i64, i32, i64
  }

}
