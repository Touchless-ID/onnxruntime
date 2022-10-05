# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
add_library(wil INTERFACE)


    FetchContent_Declare(
      microsoft_wil
      URL https://github.com/microsoft/wil/archive/e8c599bca6c56c44b6730ad93f6abbc9ecd60fc1.zip
    )
    FetchContent_Populate(microsoft_wil)
    target_include_directories(wil INTERFACE ${microsoft_wil_SOURCE_DIR}/include)
