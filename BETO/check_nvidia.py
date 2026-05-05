#!/usr/bin/env python3
import subprocess
import platform
import shutil


def run_cmd(cmd):
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return None, "", str(e)


def check_nvidia_smi():
    print("\n=== nvidia-smi ===")
    if not shutil.which("nvidia-smi"):
        print("❌ nvidia-smi no está en el PATH")
        return

    code, out, err = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )

    if code == 0 and out:
        print("✅ NVIDIA detectada por nvidia-smi:")
        print(out)
    else:
        print("❌ nvidia-smi falló")
        if err:
            print(err)


def check_torch():
    print("\n=== PyTorch CUDA ===")
    try:
        import torch
    except ImportError:
        print("⚠️ PyTorch no está instalado")
        return

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"✅ GPUs CUDA detectadas: {count}")
        for i in range(count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("❌ PyTorch no ve CUDA")


def check_system():
    print("\n=== Sistema ===")
    print(f"SO: {platform.system()} {platform.release()}")

    system = platform.system().lower()

    if system == "linux":
        code, out, err = run_cmd(["lspci"])
        if code == 0:
            lines = [line for line in out.splitlines() if "nvidia" in line.lower()]
            if lines:
                print("✅ NVIDIA detectada por lspci:")
                for line in lines:
                    print(line)
            else:
                print("❌ lspci no encontró NVIDIA")
        else:
            print("⚠️ No se pudo ejecutar lspci")

    elif system == "windows":
        code, out, err = run_cmd(
            ["wmic", "path", "win32_VideoController", "get", "name"]
        )
        if code == 0:
            lines = [line for line in out.splitlines() if "nvidia" in line.lower()]
            if lines:
                print("✅ NVIDIA detectada por Windows:")
                for line in lines:
                    print(line)
            else:
                print("❌ Windows no encontró NVIDIA en VideoController")
        else:
            print("⚠️ No se pudo ejecutar wmic")

    elif system == "darwin":
        code, out, err = run_cmd(["system_profiler", "SPDisplaysDataType"])
        if code == 0 and "nvidia" in out.lower():
            print("✅ NVIDIA detectada en macOS")
        else:
            print("❌ macOS no encontró NVIDIA")


def main():
    print("Comprobando si hay GPU NVIDIA...\n")
    check_nvidia_smi()
    check_torch()
    check_system()


if __name__ == "__main__":
    main()
