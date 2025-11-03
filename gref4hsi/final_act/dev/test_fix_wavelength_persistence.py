# Test fix for wavelength interpolation persistence issue
# This file demonstrates a potential fix for the problem where
# interpolated/cropped data reverts to original after training

import numpy as np


def create_preprocessing_checkpoint(cube):
    """
    Create a checkpoint of preprocessed data to detect if it gets modified.

    Returns a dict with:
    - data_hash: Hash of data_corrected array
    - wavelengths_copy: Copy of wavelengths
    - data_shape: Shape of data_corrected
    """
    checkpoint = {
        "data_hash": hash(cube.data_corrected.tobytes()),
        "wavelengths": cube.wavelengths.copy(),
        "data_shape": cube.data_corrected.shape,
        "wavelengths_id": id(cube.wavelengths),
        "data_id": id(cube.data_corrected),
    }
    return checkpoint


def verify_preprocessing_checkpoint(cube, checkpoint):
    """
    Verify that preprocessing hasn't been undone.

    Returns (is_valid, message)
    """
    current_hash = hash(cube.data_corrected.tobytes())

    if current_hash != checkpoint["data_hash"]:
        return (False, "❌ data_corrected has been modified!")

    if cube.data_corrected.shape != checkpoint["data_shape"]:
        return (
            False,
            f"❌ data_corrected shape changed: {checkpoint['data_shape']} → {cube.data_corrected.shape}",
        )

    if len(cube.wavelengths) != len(checkpoint["wavelengths"]):
        return (
            False,
            f"❌ wavelengths length changed: {len(checkpoint['wavelengths'])} → {len(cube.wavelengths)}",
        )

    if not np.allclose(cube.wavelengths, checkpoint["wavelengths"]):
        return (False, "❌ wavelengths array has different values!")

    if id(cube.wavelengths) != checkpoint["wavelengths_id"]:
        return (False, f"⚠️  wavelengths is a different object (but values match)")

    if id(cube.data_corrected) != checkpoint["data_id"]:
        return (False, f"⚠️  data_corrected is a different object (but values match)")

    return (True, "✅ Preprocessing intact!")


# Potential fix: Make wavelengths and data_corrected immutable
def protect_preprocessed_data(cube):
    """
    Make data_corrected and wavelengths read-only to prevent accidental modification.
    """
    # Make arrays read-only
    cube.data_corrected.flags.writeable = False
    cube.wavelengths.flags.writeable = False

    print("🔒 Preprocessed data is now protected (read-only)")
    print(f"   data_corrected.flags.writeable = {cube.data_corrected.flags.writeable}")
    print(f"   wavelengths.flags.writeable = {cube.wavelengths.flags.writeable}")


def unprotect_preprocessed_data(cube):
    """
    Make data_corrected and wavelengths writeable again.
    """
    cube.data_corrected.flags.writeable = True
    cube.wavelengths.flags.writeable = True

    print("🔓 Preprocessed data is now writeable")
