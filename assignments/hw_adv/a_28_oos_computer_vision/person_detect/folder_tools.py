from pathlib import Path
from enum import IntFlag, auto

SAFE_VIDEO_EXTENSIONS = {
    ".mp4",   # The global standard (usually H.264 or H.265)
    ".mkv",   # Very popular in research datasets; supports almost any codec
    ".mov",   # Apple standard, but well-supported by OpenCV/FFmpeg
    ".avi",   # Older format, but very common in legacy CCTV systems
    ".webm",  # Google's web-optimized format (VP8/VP9 codecs)
}
FFMPEG_VIDEO_EXTENSIONS = {
    '.ts',
    '.m4v',
    '.flv',
    '.h265',
}
PROPRIETARY_VIDEO_EXTENSIONS = {
    '.dav',
    '.dv4',
    '.hk',
    '.nvr',
    '.264',
}
SAFE_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg",  # Most common for surveillance (lossy compression)
    ".png",           # Best for annotations/masks (lossless, supports transparency)
    ".bmp",           # Raw, uncompressed (large files, but easy to process)
    ".tiff", ".tif",  # High-quality/scientific imaging
    ".webp",          # Modern web format (good compression, newer libraries support it)
}
WINDOWS_SENSITIVE_PATHS = {
    # System Roots & Core
    "c:\\windows",
    "c:\\program files",
    "c:\\program files (x86)",
    "c:\\programdata",
    
    # Restricted / Hidden System Folders
    "system volume information",  # Guaranteed PermissionError
    "$recycle.bin",               # Deleted files
    "recovery",
    "boot",
    "hiberfil.sys",               # Huge memory dump file
    "pagefile.sys",               # Huge swap file
    "swapfile.sys",

    # Performance Killers (User Data)
    # Note: You might want to scan specific parts of Users, but avoid AppData
    "appdata",                    # Contains browser caches (millions of small files)
    "application data",           # Legacy junction point
    "local settings",             # Legacy junction point
    "cookies",
    "temporary internet files"
}
LINUX_SENSITIVE_PATHS = {
    # Virtual File Systems (CRITICAL TO BLOCK)
    "/proc",  # Kernel process info. Infinite depth.
    "/sys",   # System hardware info.
    "/dev",   # Device nodes (reading /dev/zero will never stop)
    "/run",   # Runtime variable data
    "/snap",  # Snap package mounts (read-only loops)

    # System Directories (Unlikely to contain your footage)
    "/boot",
    "/etc",
    "/usr",   # Binaries and libraries
    "/lib",
    "/lib64",
    "/bin",
    "/sbin",
    "/var/lib", # Database files often live here
    "/var/run",

    # Mount points (Avoid scanning network drives/backups accidentally)
    "/mnt",
    "/media",
    
    # Trash
    ".trash",
    ".trash-1000"
}
MAC_SENSITIVE_PATHS = {
    # Core System (SIP Protected)
    "/system",
    "/library",             # System-wide library (not User library)
    "/private/var",         # System logs and temp files
    "/private/etc",
    
    # External / Virtual
    "/volumes",             # Mounts for USBs, Network Drives, DMG files
    "/network",
    "/dev",                 # Device nodes
    "/cores",               # Kernel panic dumps
    
    # User Specific Performance Killers
    # Note: ~ is /Users/username
    "library/caches",       # User cache (massive)
    "library/logs",
    "library/containers",   # Sandboxed app data
    ".trash"
}
UNIVERSAL_NOISE_PATHS = {
    # Version Control
    ".git",
    ".svn",
    ".hg",
    
    # Dependency Managers (The #1 cause of slow scripts)
    "node_modules",         # JavaScript/Node
    "venv",                 # Python Virtual Env
    ".venv",
    "__pycache__",          # Python compiled bytecode
    ".idea",                # JetBrains IDE settings
    ".vscode",              # VS Code settings
    "build",                # Compiled artifacts
    "dist",
    "target"                # Java/Rust build folders
}
BLOCK_LIST = UNIVERSAL_NOISE_PATHS | LINUX_SENSITIVE_PATHS \
         | WINDOWS_SENSITIVE_PATHS | MAC_SENSITIVE_PATHS

def is_safe_path(path_to_check, block_list: set[str] = BLOCK_LIST) -> bool:
    """
    Checks if path_to_check is inside any blocked folder.
    """
    try:
        p = Path(path_to_check).resolve() # resolve() converts relative paths to absolute
        p_str = str(p).lower()
        
        # 1. Block Empty/Current directory
        if path_to_check in ["", ".", ".."]:
            print("Error: Relative navigation paths are blocked.")
            return False

        # 2. Block File System Roots
        if p.parent == p:
            print(f"Error: {p} is a system root. Iteration blocked.")
            return False
        
        # 3. Check for exact path prefixes (e.g., "/proc" or "C:\Windows")
        for blocked in block_list:
            if p_str.startswith(blocked.lower()):
                return False

        # 4. Check individual components (e.g., any folder named ".git" or "node_modules")
        parts = set(part.lower() for part in p.parts)

        if not parts.isdisjoint(block_list):
            return False

        return True
    except (PermissionError, OSError):
        print("Error: Access Denied")
        return False

class ft(IntFlag):
    VID = auto()  # Value: 1
    IMG = auto()  # Value: 2

def file_generator(folder_path: Path, types: ft):
    # Ensure the path exists first
    if not folder_path.exists():
        print(f"Error: The folder '{folder_path}' does not exist.")
        return iter(())

    if not is_safe_path(folder_path):
        print("Error: Unsafe path.")
        return iter(())
    
    selected_exts = set()

    if ft.VID in types:
        selected_exts.update(SAFE_VIDEO_EXTENSIONS)
    
    if ft.IMG in types:
        selected_exts.update(SAFE_IMAGE_EXTENSIONS)

    files = (f for f in folder_path.rglob('*') 
         if f.is_file() and f.suffix.lower() in selected_exts)

    return files

if __name__ == "__main__":
    # example usage
    files = file_generator(Path("folder/to/test"), ft.VID | ft.IMG)

    # show one
    print(next(files, None))