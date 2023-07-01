"""Utility functions for interacting with Google Drive."""

from typing import Optional

from absl import flags
from oauth2client.client import GoogleCredentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

_USE_PYDRIVE = flags.DEFINE_bool(
    'use_pydrive', False, 'Whether or not to use PyDrive to save files.')


def initialize_drive() -> Optional[GoogleDrive]:
    """Initializes the Google Drive API."""
    if _USE_PYDRIVE.value:
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        return GoogleDrive(gauth)
    return None
