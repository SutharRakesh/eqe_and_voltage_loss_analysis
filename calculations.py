from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


HC_EV_NM = 1239.841984


def energy_ev(wavelength_nm: np.ndarray) -> np.ndarray:
    """Convert wavelength in nm to photon energy in eV."""
    return HC_EV_NM / wavelength_nm


def cumtrapz_np(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integration using the actual x spacing."""
    if len(x) < 2:
        return np.zeros_like(x)

    increments = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    return np.concatenate([[0.0], np.cumsum(increments)])


def trapz_np(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration using the actual x spacing."""
    if len(x) < 2:
        return 0.0

    return float(
        np.sum(
            0.5 * (y[1:] + y[:-1]) * np.diff(x)
        )
    )


def calculate_jsc(
    spectrum: pd.DataFrame,
    eqe: pd.DataFrame,
) -> Dict:
    """
    Calculate EQE-integrated short-circuit current density.

    The calculation is restricted to the actual overlap between the
    measured EQE wavelength range and the illumination spectrum.
    No EQE extrapolation is performed outside the measured range.
    """
    spectrum_wl = spectrum["Wavelength_nm"].to_numpy(float)
    spectrum_irr = spectrum["Irradiance_W_m2_nm"].to_numpy(float)

    eqe_wl = eqe["Wavelength_nm"].to_numpy(float)
    eqe_fraction = eqe["EQE_fraction"].to_numpy(float)

    wl_min = max(eqe_wl.min(), spectrum_wl.min())
    wl_max = min(eqe_wl.max(), spectrum_wl.max())

    if wl_max <= wl_min:
        raise ValueError(
            "The EQE wavelength range does not overlap "
            "the illumination spectrum."
        )

    spectrum_points = spectrum_wl[
        (spectrum_wl >= wl_min)
        & (spectrum_wl <= wl_max)
    ]

    eqe_points = eqe_wl[
        (eqe_wl >= wl_min)
        & (eqe_wl <= wl_max)
    ]

    wl = np.unique(
        np.concatenate(
            [
                [wl_min],
                spectrum_points,
                eqe_points,
                [wl_max],
            ]
        )
    )

    # Interpolation occurs only inside the validated overlap.
    irr = np.interp(
        wl,
        spectrum_wl,
        spectrum_irr,
    )

    eqe_grid = np.interp(
        wl,
        eqe_wl,
        eqe_fraction,
    )

    photon_energy = energy_ev(wl)

    # AM1.5G spectral irradiance:
    # W m^-2 nm^-1
    #
    # The 0.1 factor converts the integrated result
    # from A/m^2 to mA/cm^2.
    integrand = (
        eqe_grid
        * irr
        / photon_energy
    )

    cumulative = (
        0.1
        * cumtrapz_np(
            integrand,
            wl,
        )
    )

    jsc = (
        float(cumulative[-1])
        if len(cumulative)
        else 0.0
    )

    processed = pd.DataFrame(
        {
            "Wavelength_nm": wl,
            "Photon_energy_eV": photon_energy,
            "Irradiance_W_m2_nm": irr,
            "EQE_fraction": eqe_grid,
            "EQE_percent": 100 * eqe_grid,
            "Jsc_integrand": integrand,
            "Cumulative_Jsc_mA_cm2": cumulative,
        }
    )

    return {
        "jsc": jsc,
        "processed": processed,
        "integration_min_nm": float(wl_min),
        "integration_max_nm": float(wl_max),
    }