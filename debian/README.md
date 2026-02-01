# Debian Packaging

This directory contains the necessary files to build a Debian package (`.deb`) for the Weed library.

## Files

*   **control.in**: Template for the Debian `control` file, defining package metadata, dependencies, and descriptions for the library and its development headers (`libqrack-dev` equivalent).
*   **rules**: The `make` rules for building the package.
*   **copyright**: Copyright information for the package.
*   **README.Debian**: specific notes for Debian users.
*   **README.source**: Information about the source package.
*   **changelog**: The changelog for the Debian package.
*   **files**, **libqrack*.dirs.in**, **libqrack*.install.in**: Helper files for installation directories and file mappings.
*   **triggers**: Package triggers.

These files are typically used by packaging tools (like `dpkg-buildpackage` or CPack's Debian generator) to create installable packages.
