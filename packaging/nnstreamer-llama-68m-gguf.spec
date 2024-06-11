Name:       nnstreamer-llama-68m-gguf
Summary:    nnstreamer-llama-68m-gguf shared library
Version:    1.0.0
Release:    0
Group:      Machine Learning/ML Framework
Packager:   Suyeon Kim <suyeon5.kim@samsung.com>
License:    MIT License
Source0:    %{name}-%{version}.tar.gz
Source1001: %{name}.manifest
Source2001: llama-68m-chat-v1.fp16.gguf
# Source2002: gb0.wav

Requires:	nnstreamer

%ifarch armv7l
BuildRequires: clang-accel-armv7l-cross-arm
%endif

%ifarch armv7hl
BuildRequires: clang-accel-armv7hl-cross-arm
%endif

%ifarch aarch64
BuildRequires: clang-accel-aarch64-cross-aarch64
%endif

BuildRequires:  clang
BuildRequires:  pkg-config
BuildRequires:  nnstreamer-devel

%define     nnstexampledir	/usr/lib/nnstreamer/bin

%description
nnstreamer-llama-68m-gguf shared library

%prep
%setup -q
mkdir -p models
cp %{SOURCE1001} .
cp %{SOURCE2001} models/
# cp %{SOURCE2002} samples/

%build

# CLANG_ASMFLAGS=" --target=%{_host} "
CLANG_CFLAGS=" --target=%{_host} "
CLANG_CXXFLAGS=" --target=%{_host} "

# export ASMFLAGS="${CLANG_ASMFLAGS}"
export CFLAGS="${CLANG_CFLAGS}"
export CXXFLAGS="${CLANG_CXXFLAGS}"

export CC=clang
export CXX=clang++

make nnstreamer

%install
mkdir -p %{buildroot}%{nnstexampledir}/models
# cp main %{buildroot}%{nnstexampledir}
cp libnnstreamer-llama.so %{buildroot}%{nnstexampledir}
cp models/llama-68m-chat-v1.fp16.gguf %{buildroot}%{nnstexampledir}/models

%files
%manifest nnstreamer-llama-68m-gguf.manifest
%defattr(-,root,root,-)
%{nnstexampledir}/*

%changelog
* Fri May 10 2024 Suyeon Kim <suyeon5.kim@samsung.com>
- create the nnstreamer-llama-68m-gguf shared library

