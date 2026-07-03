from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from .constants import repo_root
except ImportError:
    from constants import repo_root  # type: ignore[no-redef]

def page_upload_ddrs() -> None:
    st.header("Upload DDRs")
    st.caption(
        "Upload your DDR PDFs to add them to the pipeline. "
        "Supports **individual PDFs** (one per report day) or a **combined PDF** "
        "(all DDRs in a single file — auto-split on Report No. boundaries)."
    )

    st.subheader("1 · Well Configuration")
    existing_wells = sorted([
        p.name for p in (repo_root / "data" / "fields" / "UtahForge" / "wells").iterdir()
        if p.is_dir()
    ]) if (repo_root / "data" / "fields" / "UtahForge" / "wells").exists() else []

    col_a, col_b = st.columns(2)
    with col_a:
        well_mode = st.radio("Well", ["Existing well", "New well"], horizontal=True)
    with col_b:
        if well_mode == "Existing well" and existing_wells:
            well_id = st.selectbox("Select well", existing_wells)
        else:
            well_id = st.text_input("Well ID (e.g. FORGE-16A-78-32)", value="FORGE-16A-78-32")

    rig_name = st.text_input("Rig name (used in filename)", value="UtahForge")

    st.subheader("2 · Upload PDFs")
    st.info(
        "**Individual PDFs:** Upload one file per report day using the Utah FORGE filename pattern "
        "`Utah_Forge_FORGE_16A_(78)-32_<Phase>-C_<Date>_<Date>_<ReportNo>_reporttmp.pdf`.  \n"
        "**Combined PDF:** Upload a single file containing all DDRs — the pipeline will "
        "auto-detect and split on Report No. headers.",
        icon="ℹ️",
    )
    uploaded = st.file_uploader(
        "Drop DDR PDF(s) here",
        type=["pdf"],
        accept_multiple_files=True,
        help="Individual files or a single combined PDF containing all DDRs.",
    )

    if not uploaded:
        st.stop()

    st.subheader("3 · Preview")

    raw_dir = repo_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        staged: list[Path] = []

        for upload in uploaded:
            dst = tmp_path / upload.name
            dst.write_bytes(upload.read())
            staged.append(dst)

        is_combined = (
            len(staged) == 1 and
            _is_combined_pdf(staged[0])
        )

        if is_combined:
            st.info(f"**Combined PDF detected:** `{staged[0].name}`. Auto-splitting by Report No...")
            try:
                from scripts.split_combined_ddr import detect_ddr_starts
                starts = detect_ddr_starts(staged[0])
                st.success(f"Found **{len(starts)} DDRs** in combined file.")
                preview_df = pd.DataFrame([
                    {"Report No": rno, "Date": date, "First page": pg + 1}
                    for pg, rno, date in starts
                ])
                st.dataframe(preview_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Error scanning combined PDF: {e}")
                return
        else:
            st.success(f"**{len(staged)} individual PDF(s)** ready for processing.")
            for f in staged[:10]:
                st.write(f"  • `{f.name}`")
            if len(staged) > 10:
                st.write(f"  … and {len(staged) - 10} more")

        st.subheader("4 · Run Pipeline")

        if st.button("▶ Process DDRs", type="primary"):
            progress = st.progress(0, text="Saving files…")
            log_area = st.empty()
            logs: list[str] = []

            def _log(msg: str) -> None:
                logs.append(msg)
                log_area.code("\n".join(logs[-30:]))

            if is_combined:
                _log(f"Splitting combined PDF into data/raw/ …")
                try:
                    from scripts.split_combined_ddr import split_combined_ddr
                    results = split_combined_ddr(staged[0], raw_dir, well_id, rig_name)
                    for rno, date, out_path in results:
                        _log(f"  ✓ DDR-{rno:03d} {date} → {out_path.name}")
                    _log(f"Split {len(results)} DDRs.")
                except Exception as e:
                    st.error(f"Split failed: {e}")
                    return
            else:
                for f in staged:
                    dst = raw_dir / f.name
                    shutil.copy2(f, dst)
                    _log(f"  ✓ Saved {f.name}")

            progress.progress(20, text="Preprocessing PDFs…")

            well_dir = repo_root / "data" / "fields" / "UtahForge" / "wells" / well_id
            well_dir.mkdir(parents=True, exist_ok=True)
            safe_rig  = rig_name.replace(" ", "")
            safe_well = well_id.replace("/", "-").replace(" ", "_")
            prefix = f"{safe_rig}-DDR-"
            ids_file = well_dir / "ddr_ids.txt"
            existing_prefixes = set(ids_file.read_text().splitlines()) if ids_file.exists() else set()
            existing_prefixes.add(prefix)
            ids_file.write_text("\n".join(sorted(existing_prefixes)))
            _log(f"Registered well manifest: {well_id} → prefix '{prefix}'")

            _log("Running batch_preprocess_raw_ddrs.py …")
            result = subprocess.run(
                ["python3", "scripts/batch_preprocess_raw_ddrs.py", "--no-resume"],
                capture_output=True, text=True, cwd=repo_root,
            )
            for line in (result.stdout + result.stderr).splitlines()[-30:]:
                _log(line)
            if result.returncode != 0:
                st.error("Preprocessing failed — see log above.")
                return

            progress.progress(55, text="Extracting wellbore events…")

            for script, label, pct in [
                ("scripts/extract_wellbore_events.py",     "wellbore events",   65),
                ("scripts/extract_completion_string.py",   "completion string", 75),
                ("scripts/extract_frac_sleeve_status.py",  "frac sleeve status",82),
                ("scripts/rebuild_field_analysis.py",      "field analysis",    92),
            ]:
                _log(f"Running {script} …")
                r = subprocess.run(
                    ["python3", script],
                    capture_output=True, text=True, cwd=repo_root,
                )
                for line in (r.stdout + r.stderr).splitlines()[-10:]:
                    _log(line)
                progress.progress(pct, text=f"Running {label}…")

            progress.progress(100, text="Done.")
            st.success(
                "Pipeline complete. **Refresh the browser** to see your data in the dashboard.",
                icon="✅",
            )
            st.cache_data.clear()


def _is_combined_pdf(path: Path) -> bool:
    try:
        import pdfplumber
        _RPT = re.compile(r"Report\s+No\.?\s*:?\s*(\d+)", re.I)
        seen = set()
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages[:20]:
                t = page.extract_text() or ""
                m = _RPT.search(t)
                if m:
                    seen.add(int(m.group(1)))
                if len(seen) >= 2:
                    return True
        return len(seen) >= 2
    except Exception:
        return False
