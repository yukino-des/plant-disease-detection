%define name plant-disease-detection-3181137349
%define version 0.0.1
%define unmangled_version 0.0.1
%define release 1

Summary: Plant disease detection, using MobileNetV2 and YOLOv4, Pytorch implementation.
Name: %{name}
Version: %{version}
Release: %{release}
Source0: %{name}-%{unmangled_version}.tar.gz
License: UNKNOWN
Group: Development/Libraries
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-buildroot
Prefix: %{_prefix}
BuildArch: noarch
Vendor: YukinoShita Yukino <3181137349go@gmail.com>
Url: https://github.com/yukin-des/plant-disease-detection.git

%description
# backend

### Plant disease detection, using MobileNetV2 and YOLOv4, Pytorch implementation.

install requirements (CPU/GPU version)

```shell
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install python-multipart

pip freeze > requirements.txt
pip uninstall -r requirements.txt -y
```

%prep
%setup -n %{name}-%{unmangled_version} -n %{name}-%{unmangled_version}

%build
python3 setup.py build

%install
python3 setup.py install --single-version-externally-managed -O1 --root=$RPM_BUILD_ROOT --record=INSTALLED_FILES

%clean
rm -rf $RPM_BUILD_ROOT

%files -f INSTALLED_FILES
%defattr(-,root,root)
