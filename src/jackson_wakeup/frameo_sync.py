from __future__ import annotations

import ctypes
import logging
import os
import shutil
import string
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)


_IMAGE_EXTS: set[str] = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def _com_initialize() -> None:
    """Best-effort COM initialization for the current thread (Windows only)."""

    if not _is_windows():
        return
    try:
        import pythoncom  # type: ignore

        pythoncom.CoInitialize()
    except Exception:
        # Not fatal; CopyHere may still work, but this improves reliability.
        logger.debug("COM initialization failed (pythoncom.CoInitialize)", exc_info=True)


def _com_pump() -> None:
    """Best-effort Windows message pump to help Shell CopyHere progress."""

    if not _is_windows():
        return
    try:
        import pythoncom  # type: ignore

        pythoncom.PumpWaitingMessages()
    except Exception:
        pass


def _is_windows() -> bool:
    return os.name == "nt"


def _get_volume_label_windows(root: str) -> str | None:
    # root must look like "E:\\"
    try:
        vol_name_buf = ctypes.create_unicode_buffer(261)
        fs_name_buf = ctypes.create_unicode_buffer(261)
        serial = ctypes.c_uint(0)
        max_comp_len = ctypes.c_uint(0)
        fs_flags = ctypes.c_uint(0)
        ok = ctypes.windll.kernel32.GetVolumeInformationW(
            ctypes.c_wchar_p(root),
            vol_name_buf,
            ctypes.sizeof(vol_name_buf),
            ctypes.byref(serial),
            ctypes.byref(max_comp_len),
            ctypes.byref(fs_flags),
            fs_name_buf,
            ctypes.sizeof(fs_name_buf),
        )
        if not ok:
            return None
        return vol_name_buf.value or None
    except Exception:
        return None


def _get_drive_type_windows(root: str) -> int | None:
    try:
        return int(ctypes.windll.kernel32.GetDriveTypeW(ctypes.c_wchar_p(root)))
    except Exception:
        return None


def _candidate_frameo_dirs(drive_root: Path) -> list[Path]:
    # Some devices expose DCIM at root; others under "Internal storage".
    return [
        drive_root / "DCIM",
        drive_root / "Internal storage" / "DCIM",
    ]


def resolve_frameo_destination_dir(*, explicit_dir: str | None, device_label: str) -> Path | None:
    """Resolve a filesystem directory for copying images to Frameo.

    Notes:
    - Windows "This PC\\Frame\\Internal storage\\DCIM" is a shell/MTP namespace path and is
      often NOT directly accessible as a normal filesystem path.
    - This function supports either an explicit filesystem path (preferred) or auto-detection
      of a removable drive by volume label.
    """

    if explicit_dir:
        # If this is a Windows Shell / MTP path like "This PC\\Frame\\Internal storage\\DCIM",
        # it won't exist as a normal filesystem path. We'll handle that separately.
        if explicit_dir.strip().lower().startswith("this pc\\"):
            return None

        p = Path(explicit_dir)
        if p.exists() and p.is_dir():
            return p
        logger.warning("Frameo destination dir does not exist or is not a directory: %s", p)
        return None

    if not _is_windows():
        return None

    wanted = (device_label or "").strip().lower()
    if not wanted:
        return None

    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        if not os.path.exists(root):
            continue

        # Filter to removable drives when possible (2 = DRIVE_REMOVABLE).
        dtype = _get_drive_type_windows(root)
        if dtype is not None and dtype not in (2,):
            continue

        label = _get_volume_label_windows(root)
        if not label:
            continue
        if label.strip().lower() != wanted:
            continue

        drive_root = Path(root)
        for cand in _candidate_frameo_dirs(drive_root):
            if cand.exists() and cand.is_dir():
                return cand

        # If the label matches but we didn't find DCIM where expected, at least return the root.
        logger.warning("Found drive %s with label %r but no DCIM dir found; using drive root", root, label)
        return drive_root

    return None


def copy_image_to_frameo(
    *,
    src_image: Path,
    dest_dir: Path,
    dest_filename: str,
) -> Path:
    """Copy image to Frameo directory, overwriting previous image.

    Uses a temp file + replace for best-effort atomicity.
    """

    if not src_image.exists():
        raise FileNotFoundError(f"Source image does not exist: {src_image}")
    if not dest_dir.exists() or not dest_dir.is_dir():
        raise FileNotFoundError(f"Destination directory does not exist: {dest_dir}")

    dest_path = dest_dir / dest_filename
    tmp_path = dest_dir / (dest_filename + ".tmp")

    # Copy then replace.
    shutil.copyfile(src_image, tmp_path)
    tmp_path.replace(dest_path)
    return dest_path


def purge_images_in_dir(dest_dir: Path) -> int:
    """Delete all image files in dest_dir.

    This is intentionally limited to common image extensions.
    Returns number of deleted files (best-effort).
    """

    deleted = 0
    for p in dest_dir.iterdir():
        try:
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                p.unlink(missing_ok=True)
                deleted += 1
        except Exception:
            logger.debug("Failed deleting %s", p, exc_info=True)
    return deleted


def _shell_find_child(folder, name: str):
    wanted = name.strip().lower()
    for item in folder.Items():
        try:
            if str(item.Name).strip().lower() == wanted:
                return item
        except Exception:
            continue
    return None


def _resolve_shell_folder(shell_path: str):
    """Resolve a Windows Shell folder object for a path like 'This PC\\Frame\\Internal storage\\DCIM'.

    Returns (folder, win32com.client) or (None, None) if not available.
    """

    if not _is_windows():
        return None, None

    try:
        import win32com.client  # type: ignore
    except Exception:
        return None, None

    parts = [p for p in shell_path.split("\\") if p.strip()]
    if len(parts) < 2 or parts[0].strip().lower() != "this pc":
        return None, None

    shell = win32com.client.Dispatch("Shell.Application")
    folder = shell.NameSpace("shell:MyComputerFolder")
    if folder is None:
        return None, win32com.client

    for seg in parts[1:]:
        item = _shell_find_child(folder, seg)
        if item is None:
            return None, win32com.client
        folder = shell.NameSpace(item)
        if folder is None:
            return None, win32com.client

    return folder, win32com.client


def purge_images_in_shell_path(*, shell_path: str) -> int:
    """Delete all image items in an MTP/Windows Shell folder.

    Requires pywin32. Returns number of deleted items (best-effort).
    """

    if not _is_windows():
        return 0

    # Deletion is also async in many shell namespaces.
    _com_initialize()

    folder, _ = _resolve_shell_folder(shell_path)
    if folder is None:
        return 0

    def _is_folder_item(it) -> bool:
        try:
            return bool(getattr(it, "IsFolder", False))
        except Exception:
            return False

    def _list_payload_names() -> list[str]:
        """List non-folder items in the MTP folder.

        Note: On some MTP devices, FolderItem.Name may omit file extensions, so we
        cannot reliably detect images by suffix alone.
        """

        names: list[str] = []
        try:
            for it in folder.Items():
                try:
                    if _is_folder_item(it):
                        continue
                    nm = str(getattr(it, "Name", "") or "")
                    if nm:
                        names.append(nm)
                except Exception:
                    continue
        except Exception:
            return []
        return names

    def _do_delete(item) -> bool:
        """Try hard to execute the delete verb for a Shell item."""

        # First: direct invoke.
        try:
            item.InvokeVerb("delete")
            return True
        except Exception:
            pass

        # Second: look for an explicit "Delete" verb (localized name).
        try:
            verbs = item.Verbs()
            for v in verbs:
                try:
                    vn = str(getattr(v, "Name", "") or "")
                    vn_clean = vn.replace("&", "").strip().lower()
                    # English Windows uses "Delete"; include a couple of common alternatives.
                    if vn_clean in {"delete", "remove", "erase"} or "delete" in vn_clean:
                        v.DoIt()
                        return True
                except Exception:
                    continue
        except Exception:
            pass

        return False

    before = {n.strip().lower() for n in _list_payload_names()}
    if not before:
        return 0

    # Iterate a snapshot to avoid weirdness while deleting.
    items = list(folder.Items())
    attempted = 0
    for item in items:
        try:
            if _is_folder_item(item):
                continue
            if _do_delete(item):
                attempted += 1
        except Exception:
            continue

    # Wait for deletion to actually apply.
    deadline = time.time() + 45.0
    last_seen: list[str] = []
    while time.time() < deadline:
        _com_pump()
        cur = _list_payload_names()
        last_seen = cur
        if not cur:
            break
        time.sleep(0.25)
        _com_pump()

    after = {n.strip().lower() for n in _list_payload_names()}
    deleted = max(0, len(before) - len(after))
    if after:
        logger.warning(
            "Frameo MTP delete may be incomplete: before=%d attempted=%d after=%d (showing up to 10 remaining payload items): %s",
            len(before),
            attempted,
            len(after),
            ", ".join(sorted(after)[:10]),
        )
    return deleted


def copy_image_to_frameo_shell_path(
    *,
    src_image: Path,
    shell_path: str,
    dest_filename: str,
    timeout_seconds: float = 90.0,
) -> bool:
    """Copy image to a Windows Shell path like 'This PC\\Frame\\Internal storage\\DCIM'.

    This supports MTP devices that show up under "This PC" but do not mount as drive letters.

    Requires pywin32 (win32com). Returns True on best-effort success.
    """

    if not _is_windows():
        return False

    # Shell.CopyHere is asynchronous and often needs COM/message pumping to complete
    # reliably from a script.
    _com_initialize()

    if not src_image.exists():
        raise FileNotFoundError(f"Source image does not exist: {src_image}")

    folder, win32com_client = _resolve_shell_folder(shell_path)
    if folder is None:
        if shell_path.strip().lower().startswith("this pc\\"):
            logger.warning(
                "Frameo appears to be an MTP device (This PC\\...). Install pywin32 to enable copying: py -m pip install pywin32"
            )
        else:
            logger.warning("Unsupported shell path (expected to start with 'This PC\\'): %r", shell_path)
        return False

    try:
        # Helps diagnose if we resolved the folder you expect.
        logger.debug(
            "Resolved Frameo MTP folder: Name=%r Title=%r",
            getattr(getattr(folder, "Self", None), "Name", None),
            getattr(folder, "Title", None),
        )
    except Exception:
        pass

    def _list_item_names() -> list[str]:
        names: list[str] = []
        try:
            for it in folder.Items():
                try:
                    nm = str(getattr(it, "Name", "") or "")
                    if nm:
                        names.append(nm)
                except Exception:
                    continue
        except Exception:
            return []
        return names

    def _is_folder_item(it) -> bool:
        try:
            return bool(getattr(it, "IsFolder", False))
        except Exception:
            return False

    def _list_payload_names() -> list[str]:
        """List non-folder items.

        Note: On some MTP devices, FolderItem.Name may omit file extensions.
        """

        names: list[str] = []
        try:
            for it in folder.Items():
                try:
                    if _is_folder_item(it):
                        continue
                    nm = str(getattr(it, "Name", "") or "")
                    if nm:
                        names.append(nm)
                except Exception:
                    continue
        except Exception:
            return []
        return names

    try:
        total_items = int(folder.Items().Count)
    except Exception:
        total_items = -1

    all_pre = _list_item_names()
    before_names = {n.strip().lower() for n in _list_payload_names()}
    logger.debug(
        "Frameo MTP folder pre-copy: total_items=%s, payload_items=%d (showing up to 10): %s",
        total_items,
        len(before_names),
        ", ".join(list(sorted(before_names))[:10]),
    )
    if all_pre and not before_names:
        logger.debug(
            "Frameo MTP folder pre-copy: non-image items (up to 10): %s",
            ", ".join([n.strip() for n in all_pre[:10]]),
        )

    # CopyHere is async; use flags to suppress UI.
    FOF_SILENT = 4
    FOF_NOCONFIRMATION = 16
    FOF_NOCONFIRMMKDIR = 512
    FOF_NOERRORUI = 1024
    flags = FOF_SILENT | FOF_NOCONFIRMATION | FOF_NOCONFIRMMKDIR | FOF_NOERRORUI

    # Important: CopyHere preserves the SOURCE filename. To force a specific destination name
    # (Frameo caches by name), stage a temporary local file with the requested dest_filename.
    staged_path: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="jackson_frameo_") as td:
            staged_path = Path(td) / dest_filename
            shutil.copyfile(src_image, staged_path)

            try:
                # Some MTP devices ignore CopyHere when given a raw filesystem path.
                # Passing a Shell item is often more reliable.
                shell = win32com_client.Dispatch("Shell.Application")
                src_ns = shell.NameSpace(str(staged_path.parent))
                src_item = None
                if src_ns is not None:
                    try:
                        src_item = src_ns.ParseName(staged_path.name)
                    except Exception:
                        src_item = None

                if src_item is not None:
                    folder.CopyHere(src_item, flags)
                else:
                    folder.CopyHere(str(staged_path), flags)
            except Exception:
                logger.exception("Shell CopyHere failed")
                return False

            # Poll for the file to appear.
            # NOTE: MTP folders can be slow to refresh; ParseName is also unreliable.
            wanted = dest_filename.strip().lower()
            deadline = time.time() + max(1.0, timeout_seconds)
            last_seen: list[str] = []
            while time.time() < deadline:
                _com_pump()
                # First try direct lookup.
                try:
                    if folder.ParseName(dest_filename) is not None:
                        return True
                except Exception:
                    pass

                # Then enumerate and look for new items.
                cur = _list_payload_names()
                last_seen = cur
                cur_lower = {n.strip().lower() for n in cur}
                if wanted in cur_lower:
                    return True

                # If any new payload item appears (even with a different name), treat as success but log it.
                new_items = [n for n in cur_lower if n not in before_names]
                if new_items:
                    logger.warning(
                        "Frameo MTP copy appears to have completed but filename differs (expected %r, saw new: %s)",
                        dest_filename,
                        ", ".join(sorted(new_items)[:10]),
                    )
                    return True

                time.sleep(0.25)
                _com_pump()

            logger.warning(
                "Timed out waiting for Frameo shell copy to complete. Last visible payload items (up to 10): %s",
                ", ".join([n.strip() for n in last_seen][:10]),
            )
            return False
    finally:
        # tempfile.TemporaryDirectory cleans up; nothing to do.
        staged_path = None
