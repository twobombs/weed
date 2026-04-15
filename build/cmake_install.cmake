# Install script for directory: /home/aryan/Documents/vscode/weed

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/weed_cl_precompile" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/weed_cl_precompile")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/weed_cl_precompile"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/aryan/Documents/vscode/weed/build/weed_cl_precompile")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/weed_cl_precompile" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/weed_cl_precompile")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/weed_cl_precompile")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/weed/common" TYPE FILE FILES
    "/home/aryan/Documents/vscode/weed/build/include/common/config.h"
    "/home/aryan/Documents/vscode/weed/include/common/half.hpp"
    "/home/aryan/Documents/vscode/weed/include/common/oclapi.hpp"
    "/home/aryan/Documents/vscode/weed/include/common/oclengine.hpp"
    "/home/aryan/Documents/vscode/weed/include/common/parallel_for.hpp"
    "/home/aryan/Documents/vscode/weed/include/common/rapidcsv.h"
    "/home/aryan/Documents/vscode/weed/include/common/serializer.hpp"
    "/home/aryan/Documents/vscode/weed/include/common/weed_functions.hpp"
    "/home/aryan/Documents/vscode/weed/include/common/weed_types.hpp"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/weed" TYPE FILE FILES
    "/home/aryan/Documents/vscode/weed/include/shared_api.hpp"
    "/home/aryan/Documents/vscode/weed/include/autograd/adam.hpp"
    "/home/aryan/Documents/vscode/weed/include/autograd/bci_loss.hpp"
    "/home/aryan/Documents/vscode/weed/include/autograd/cross_entropy_loss.hpp"
    "/home/aryan/Documents/vscode/weed/include/autograd/mse_loss.hpp"
    "/home/aryan/Documents/vscode/weed/include/autograd/node.hpp"
    "/home/aryan/Documents/vscode/weed/include/autograd/sgd.hpp"
    "/home/aryan/Documents/vscode/weed/include/autograd/zero_grad.hpp"
    "/home/aryan/Documents/vscode/weed/include/devices/gpu_device.hpp"
    "/home/aryan/Documents/vscode/weed/include/devices/pool_item.hpp"
    "/home/aryan/Documents/vscode/weed/include/devices/queue_item.hpp"
    "/home/aryan/Documents/vscode/weed/include/enums/device_tag.hpp"
    "/home/aryan/Documents/vscode/weed/include/enums/dtype.hpp"
    "/home/aryan/Documents/vscode/weed/include/enums/module_type.hpp"
    "/home/aryan/Documents/vscode/weed/include/enums/storage_type.hpp"
    "/home/aryan/Documents/vscode/weed/include/enums/quantum_function_type.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/dropout.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/embedding.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/flatten.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/gru.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/learned_positional_encoding.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/linear.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/logsoftmax.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/lstm.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/max.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/mean.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/mean_center.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/migrate_cpu.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/migrate_gpu.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/min.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/module.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/multihead_attention.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/positional_encoding.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/relu.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/rms_norm.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/rope.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/qrack_neuron.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/qrack_neuron_layer.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/sequential.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/sigmoid.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/swiglu.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/softmax.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/tanh.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/transformer_encoder_layer.hpp"
    "/home/aryan/Documents/vscode/weed/include/modules/qwen_decoder_layer.hpp"
    "/home/aryan/Documents/vscode/weed/include/ops/abs.hpp"
    "/home/aryan/Documents/vscode/weed/include/ops/clamp.hpp"
    "/home/aryan/Documents/vscode/weed/include/ops/commuting.hpp"
    "/home/aryan/Documents/vscode/weed/include/ops/copy_broadcast.hpp"
    "/home/aryan/Documents/vscode/weed/include/ops/div.hpp"
    "/home/aryan/Documents/vscode/weed/include/ops/embedding.hpp"
    "/home/aryan/Documents/vscode/weed/include/ops/in_place.hpp"
    "/home/aryan/Documents/vscode/weed/include/ops/logsoftmax.hpp"
    "/home/aryan/Documents/vscode/weed/include/ops/matmul.hpp"
    "/home/aryan/Documents/vscode/weed/include/ops/pow.hpp"
    "/home/aryan/Documents/vscode/weed/include/ops/reduce.hpp"
    "/home/aryan/Documents/vscode/weed/include/ops/real_extremum.hpp"
    "/home/aryan/Documents/vscode/weed/include/ops/real_unary.hpp"
    "/home/aryan/Documents/vscode/weed/include/ops/softmax.hpp"
    "/home/aryan/Documents/vscode/weed/include/ops/sub.hpp"
    "/home/aryan/Documents/vscode/weed/include/ops/sum.hpp"
    "/home/aryan/Documents/vscode/weed/include/ops/triu_fill.hpp"
    "/home/aryan/Documents/vscode/weed/include/ops/util.hpp"
    "/home/aryan/Documents/vscode/weed/include/storage/all_storage.hpp"
    "/home/aryan/Documents/vscode/weed/include/storage/cpu_complex_storage.hpp"
    "/home/aryan/Documents/vscode/weed/include/storage/cpu_real_storage.hpp"
    "/home/aryan/Documents/vscode/weed/include/storage/cpu_storage.hpp"
    "/home/aryan/Documents/vscode/weed/include/storage/gpu_complex_storage.hpp"
    "/home/aryan/Documents/vscode/weed/include/storage/gpu_real_storage.hpp"
    "/home/aryan/Documents/vscode/weed/include/storage/gpu_storage.hpp"
    "/home/aryan/Documents/vscode/weed/include/storage/sparse_cpu_complex_storage.hpp"
    "/home/aryan/Documents/vscode/weed/include/storage/sparse_cpu_real_storage.hpp"
    "/home/aryan/Documents/vscode/weed/include/storage/sparse_cpu_storage.hpp"
    "/home/aryan/Documents/vscode/weed/include/storage/storage.hpp"
    "/home/aryan/Documents/vscode/weed/include/storage/typed_storage.hpp"
    "/home/aryan/Documents/vscode/weed/include/tensors/base_tensor.hpp"
    "/home/aryan/Documents/vscode/weed/include/tensors/complex_scalar.hpp"
    "/home/aryan/Documents/vscode/weed/include/tensors/complex_tensor.hpp"
    "/home/aryan/Documents/vscode/weed/include/tensors/parameter.hpp"
    "/home/aryan/Documents/vscode/weed/include/tensors/real_scalar.hpp"
    "/home/aryan/Documents/vscode/weed/include/tensors/real_tensor.hpp"
    "/home/aryan/Documents/vscode/weed/include/tensors/scalar.hpp"
    "/home/aryan/Documents/vscode/weed/include/tensors/symbol_tensor.hpp"
    "/home/aryan/Documents/vscode/weed/include/tensors/tensor.hpp"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/weed" TYPE STATIC_LIBRARY FILES "/home/aryan/Documents/vscode/weed/build/libweed.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/weed/libweed_shared.so.0.7.3" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/weed/libweed_shared.so.0.7.3")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/weed/libweed_shared.so.0.7.3"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/weed" TYPE SHARED_LIBRARY FILES "/home/aryan/Documents/vscode/weed/build/libweed_shared.so.0.7.3")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/weed/libweed_shared.so.0.7.3" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/weed/libweed_shared.so.0.7.3")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/weed/libweed_shared.so.0.7.3")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/weed" TYPE SHARED_LIBRARY FILES "/home/aryan/Documents/vscode/weed/build/libweed_shared.so")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pkgconfig" TYPE FILE FILES "/home/aryan/Documents/vscode/weed/build/libweed.pc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/doc/weed" TYPE FILE FILES "/home/aryan/Documents/vscode/weed/debian/copyright")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/doc/weed" TYPE FILE FILES "/home/aryan/Documents/vscode/weed/build/changelog.gz")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/share/man/man1/weed_cl_precompile.1.gz")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/usr/local/share/man/man1" TYPE FILE FILES "/home/aryan/Documents/vscode/weed/build/weed_cl_precompile.1.gz")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/aryan/Documents/vscode/weed/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
