"""Command-line интерфейс для генерации crossfeed матриц."""

import argparse

from .pipeline import parse_strengths, run_pipeline


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Build a complex HRTF-derived, mid-safe virtual-speaker "
            "crossfeed matrix from a SOFA HRTF/DTF file."
        )
    )

    ap.add_argument(
        "--sofa",
        required=True,
        help="Input ARI HRTF/DTF SOFA file",
    )

    ap.add_argument(
        "--out-prefix",
        required=True,
        help="Output prefix, e.g. out/ari1059",
    )

    ap.add_argument(
        "--az",
        type=float,
        default=30.0,
        help="Speaker azimuth magnitude in degrees",
    )

    ap.add_argument(
        "--el",
        type=float,
        default=0.0,
        help="Speaker elevation in degrees",
    )

    ap.add_argument(
        "--analysis-ir-len",
        type=int,
        default=512,
        help="HRIR length used for complex HRTF analysis",
    )

    ap.add_argument(
        "--ir-len",
        type=int,
        default=512,
        help="Output matrix FIR length",
    )

    ap.add_argument(
        "--pre",
        type=int,
        default=12,
        help="Samples retained before detected HRIR onset",
    )

    ap.add_argument(
        "--n-fft",
        type=int,
        default=8192,
        help="FFT size used for matrix construction",
    )

    ap.add_argument(
        "--latency-samples",
        type=int,
        default=128,
        help=(
            "Common latency of direct and cross paths. "
            "Increase if omitted energy is too high."
        ),
    )

    ap.add_argument(
        "--onset-threshold-db",
        type=float,
        default=-24.0,
        help="Onset envelope threshold relative to maximum",
    )

    ap.add_argument(
        "--regularization-db",
        type=float,
        default=-40.0,
        help=(
            "Regularization power relative to maximum |Hmid|^2. "
            "More negative means weaker regularization."
        ),
    )

    ap.add_argument(
        "--smooth-octave",
        type=float,
        default=12.0,
        help=(
            "Magnitude-only smoothing fraction. "
            "12 = 1/12 octave, 0 = disabled."
        ),
    )

    ap.add_argument(
        "--max-side-gain-db",
        type=float,
        default=6.0,
        help="Maximum allowed magnitude of target Side response",
    )

    ap.add_argument(
        "--tail-fraction",
        type=float,
        default=0.15,
        help="Fraction of output FIR used for cosine tail fade",
    )

    ap.add_argument(
        "--strengths",
        type=str,
        default="0.70,1.00,1.20",
        help=(
            "0 = identity, 1 = measured speaker target, "
            ">1 = extrapolation"
        ),
    )

    ap.add_argument(
        "--ear-order",
        choices=["auto", "normal", "swapped"],
        default="auto",
        help="Receiver/ear ordering in Data.IR",
    )

    ap.add_argument(
        "--itd-mode",
        choices=["auto", "measured", "woodworth"],
        default="auto",
        help=(
            "ITD selection: measured, Woodworth model, "
            "or automatic validation/fallback"
        ),
    )

    ap.add_argument(
        "--hf-blend-start",
        type=float,
        default=3500.0,
        help=(
            "Frequency where complex Side response begins "
            "blending toward identity"
        ),
    )

    ap.add_argument(
        "--hf-blend-end",
        type=float,
        default=9000.0,
        help=(
            "Frequency where crossfeed becomes fully disabled "
            "and Side response reaches R=1"
        ),
    )

    ap.add_argument(
        "--remap",
        choices=[
            "normal",
            "swap_inputs",
            "swap_outputs",
            "swap_both",
        ],
        default="normal",
        help="Channel permutation for convolution plugin",
    )

    ap.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not save frequency-response CSV files",
    )

    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = ap.parse_args()

    strengths = parse_strengths(args.strengths)

    run_pipeline(
        sofa_path=args.sofa,
        out_prefix=args.out_prefix,
        az=args.az,
        el=args.el,
        analysis_ir_len=args.analysis_ir_len,
        ir_len=args.ir_len,
        pre=args.pre,
        n_fft=args.n_fft,
        latency_samples=args.latency_samples,
        onset_threshold_db=args.onset_threshold_db,
        regularization_db=args.regularization_db,
        smooth_octave=args.smooth_octave,
        max_side_gain_db=args.max_side_gain_db,
        tail_fraction=args.tail_fraction,
        strengths=strengths,
        ear_order=args.ear_order,
        itd_mode=args.itd_mode,
        hf_blend_start_hz=args.hf_blend_start,
        hf_blend_end_hz=args.hf_blend_end,
        remap=args.remap,
        no_csv=args.no_csv,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
