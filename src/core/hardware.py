import platform
import os

class HardwareMonitor:
    """Automatically records comprehensive hardware metrics."""
    @staticmethod
    def get_hardware_info() -> dict:
        info = {
            "CPU_Name": platform.processor(),
            "CPU_Threads": os.cpu_count() or 1,
            "OperatingSystem": platform.system() + " " + platform.release(),
            "Architecture": platform.machine(),
            "PythonVersion": platform.python_version()
        }
        try:
            import psutil
            info["CPU_Usage"] = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            info["RAM_Total_GB"] = f"{mem.total / (1024**3):.2f}"
            info["RAM_Available_GB"] = f"{mem.available / (1024**3):.2f}"
            disk = psutil.disk_usage('/')
            info["Disk_Total_GB"] = f"{disk.total / (1024**3):.2f}"
            info["Disk_Free_GB"] = f"{disk.free / (1024**3):.2f}"
        except ImportError:
            info["RAM"] = "Unknown"
        
        try:
            import torch
            info["PyTorch_Version"] = torch.__version__
            if torch.cuda.is_available():
                info["GPU_Name"] = torch.cuda.get_device_name(0)
                info["VRAM_GB"] = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f}"
                info["CUDA_Version"] = torch.version.cuda
        except ImportError:
            info["PyTorch_Version"] = "Not Installed"
            
        return info
