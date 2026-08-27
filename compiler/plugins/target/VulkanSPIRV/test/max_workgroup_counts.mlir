// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(iree-hal-assign-target-devices{targetDevices=vulkan},iree-hal-materialize-target-devices,iree-hal-resolve-device-aliases)" \
// RUN:   --iree-vulkan-target=sm_75 %s | FileCheck %s --check-prefix=DEFAULT
// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(iree-hal-assign-target-devices{targetDevices=vulkan},iree-hal-materialize-target-devices,iree-hal-resolve-device-aliases)" \
// RUN:   --iree-vulkan-target=sm_75 --iree-vulkan-max-workgroup-counts=1024 %s \
// RUN:   | FileCheck %s --check-prefix=OVERRIDE-X
// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(iree-hal-assign-target-devices{targetDevices=vulkan},iree-hal-materialize-target-devices,iree-hal-resolve-device-aliases)" \
// RUN:   --iree-vulkan-target=sm_75 --iree-vulkan-max-workgroup-counts=1024,512,256 %s \
// RUN:   | FileCheck %s --check-prefix=OVERRIDE-XYZ

module {
  util.func public @empty() {
    util.return
  }
}

//    DEFAULT: max_workgroup_counts = [2147483647, 65535, 65535]
// OVERRIDE-X: max_workgroup_counts = [1024, 65535, 65535]
// OVERRIDE-XYZ: max_workgroup_counts = [1024, 512, 256]
