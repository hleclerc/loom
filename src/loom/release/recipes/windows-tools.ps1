# loom's C++ toolchain for the `windows-x86_64` wheels ( `loom.release.envs` ), unpacked into C:\tools ( or the directory given ) when it is not there yet:
#   llvm-mingw   clang + lld + libc++ for MinGW-w64 ( UCRT ): builds the kernels as DLLs, with the C++ runtime linked in statically.
#                clang rather than GCC: GCC does not realign the stack of Windows x86-64 for AVX, so a kernel built for a recent
#                processor ( vmovdqa64 on the stack ) crashes there.
#   winlibs      GNU make ( its GCC is not used ), as mingw64\bin\make.exe -- to be put on the PATH after llvm-mingw's
#
#     powershell -NoProfile -File scripts\win_tools.ps1 [directory]
param( $root = "C:\tools" )
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
New-Item -ItemType Directory -Force $root | Out-Null

function Unpack( $url, $check ) {
    if ( Test-Path "$root\$check" ) { return }
    $zip = "$root\download.zip"
    Invoke-WebRequest $url -OutFile $zip
    tar -xf $zip -C $root
    if ( $LASTEXITCODE ) { throw "tar: exit $LASTEXITCODE" }
    Remove-Item $zip
}

Unpack "https://github.com/mstorsjo/llvm-mingw/releases/download/20261006/llvm-mingw-20261006-ucrt-x86_64.zip" "llvm-mingw-20261006-ucrt-x86_64\bin\clang++.exe"
Unpack "https://github.com/brechtsanders/winlibs_mingw/releases/download/16.2.0posix-14.0.0-ucrt-r2/winlibs-x86_64-posix-seh-gcc-16.2.0-mingw-w64ucrt-14.0.0-r2.zip" "mingw64\bin\mingw32-make.exe"
Copy-Item -Force "$root\mingw64\bin\mingw32-make.exe" "$root\mingw64\bin\make.exe"

& "$root\llvm-mingw-20261006-ucrt-x86_64\bin\clang++.exe" --version
& "$root\mingw64\bin\make.exe" --version
