"""Utility functions for interacting with Google Drive."""

import os
import tempfile
from typing import Optional

import chex
import pydrive
from absl import flags
from oauth2client.client import GoogleCredentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

_USE_PYDRIVE = flags.DEFINE_bool(
    'use_pydrive', False, 'Whether or not to use PyDrive to save files.')

_GOOGLE_DRIVE: Optional[GoogleDrive] = None


@chex.dataclass(frozen=True)
class DriveDirectory:
    """A directory that may or may not exist on Google Drive."""
    google_drive: Optional[GoogleDrive]
    directory: str


def initialize_drive():
    """Initializes the Google Drive API."""
    global _GOOGLE_DRIVE  # pylint: disable=global-statement
    if _USE_PYDRIVE.value:
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        _GOOGLE_DRIVE = GoogleDrive(gauth)


def _get_drive_dir(directory_path: str) -> pydrive.files.GoogleDriveFile:
    """Gets the Google Drive directory."""
    file_id = 'root'
    for subdir in directory_path.split('/'):
        if not subdir:
            continue
        file_list = _GOOGLE_DRIVE.ListFile({
            'q':
            f"'{file_id}' in parents "
            f"and mimeType='application/vnd.google-apps.folder' "
            f"and title='{subdir}'"
        }).GetList()
        if len(file_list) < 1:
            raise LookupError(f"Failed to find sub directory '{subdir}'")
        if len(file_list) > 1:
            raise LookupError(f"Found multiple sub directories for '{subdir}'")
        file_id = file_list[0]['id']
    return file_list[0]


def _get_drive_file(filepath: str):
    """Gets the Google Drive file."""
    head, tail = os.path.split(filepath)
    directory = _get_drive_dir(head)
    file_list = _GOOGLE_DRIVE.ListFile({
        'q':
        f"title='{tail}' and '{directory['id']}' in parents "
        f"and mimeType!='application/vnd.google-apps.folder'"
    }).GetList()
    if len(file_list) < 1:
        raise LookupError(
            f"Failed to find file '{tail}' in directory '{head}'")
    if len(file_list) > 1:
        raise LookupError(
            f"Found multiple files '{tail}' in directory '{head}'")
    return file_list[0]


def open_file(filepath: str,
              mode: str | None = None,
              encoding: str | None = None):
    """Opens a file."""
    drive_file = _get_drive_file(filepath)
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpfilepath = os.path.join(tmpdirname, drive_file['id'])
        drive_file.GetContentFile(tmpfilepath)
        return open(tmpfilepath, mode, encoding=encoding)
