REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

REM This file is used by "DML Nuget Pipeline", "Nuget WindowsAI Pipeline", "Zip-Nuget-Java-Nodejs Packaging Pipeline", "Windows CPU CI Pipeline".
set PATH=C:\azcopy;C:\Program Files (x86)\dotnet;%PATH%
set GRADLE_OPTS=-Dorg.gradle.daemon=false
