import streamlit as st
import subprocess
from pathlib import Path
from datetime import datetime
import shutil

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input"
FRAMES_DIR = BASE_DIR / "frames"
Sparse_DIR= BASE_DIR / "sparse"
LOG_FILE_FrameExtraction = BASE_DIR / "run_frame_extraction.log"
LOG_FILE_SparseReconstruction=BASE_DIR / "run_sparse_reconstruct.log"
DATABASE_PATH = BASE_DIR / "database.db"

INPUT_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)
Sparse_DIR.mkdir(exist_ok=True)

if "step" not in st.session_state:
    st.session_state.step = 1

if "extraction_done" not in st.session_state:
    st.session_state.extraction_done = False

if "sparse_reconstruct" not in st.session_state:
    st.session_state.sparse_reconstruct=False

if "logs" not in st.session_state:
    st.session_state.logs = ""

if "logs_reconstruct" not in st.session_state:
    st.session_state.logs_reconstruct=""

if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0



if st.session_state.step == 1:

    uploaded_video = st.file_uploader(
        "Drop MP4 Video",
        type=["mp4"]
    )


    fps = st.slider(
        "Frame Extraction FPS",
        min_value=1,
        max_value=30,
        value=5
    )


    extract_button = st.button(
        "Extract Frames",
        disabled=uploaded_video is None
    )

    if st.button("JUMP TO 3 Skip sparse Dev Option :) →"):
        st.session_state.step = 3
        st.rerun()

    if extract_button:

        # Save uploaded video
        video_path = INPUT_DIR / uploaded_video.name

        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())


        # Clear old frames
        for frame in FRAMES_DIR.glob("*.jpg"):
            frame.unlink()


        # Clear old logs
        st.session_state.logs = ""

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),

            "-vf",
            f"fps={fps}",

            str(FRAMES_DIR / "frame_%05d.jpg")
        ]

        with st.status(
            "Extracting frames...",
            expanded=True
        ) as status:

            st.write(f"Video: {uploaded_video.name}")
            st.write(f"Extraction rate: {fps} FPS")

            process = subprocess.Popen(

                command,

                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,

                text=True,
                bufsize=1
            )


            # Read FFmpeg output
            for line in process.stdout:

                st.session_state.logs += line

                # Also save full output to txt file
                with open(LOG_FILE_FrameExtraction, "a") as log_file:
                    log_file.write(
                        f"\n\n{'=' * 50}\n"
                        f"NEW RUN: {datetime.now()}\n"
                        f"{'=' * 50}\n"
                        )
                    log_file.write(line)


            return_code = process.wait()


            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            if return_code == 0:

                st.session_state.extraction_done = True

                # Count extracted frames
                st.session_state.frame_count = len(
                    list(FRAMES_DIR.glob("*.jpg"))
                )

                status.update(
                    label="Frame extraction complete",
                    state="complete",
                    expanded=False
                )


            # ------------------------------------------------
            # FAILURE
            # ------------------------------------------------

            else:

                st.session_state.extraction_done = False

                status.update(
                    label="Frame extraction failed",
                    state="error",
                    expanded=True
                )


    if st.session_state.extraction_done:

        st.success(
            f"✓ {st.session_state.frame_count} frames extracted"
        )

        st.caption(
            f"Saved to: {FRAMES_DIR}"
        )


        # ----------------------------------------------------
        # HIDDEN DEVELOPER LOG
        # ----------------------------------------------------

        with st.expander("Show processing log"):

            st.code(
                st.session_state.logs,
                language="text"
            )

            st.caption(
                f"Full log saved at: {LOG_FILE_FrameExtraction}"
            )


        # ----------------------------------------------------
        # NEXT
        # ----------------------------------------------------

        if st.button("Next →"):

            st.session_state.step = 2

            st.rerun()

        if st.button("JUMP TO 3 Skip sparse Dev Option :) →"):

            st.session_state.step = 3

            st.rerun()



# ============================================================
# STEP 2
# ============================================================

elif st.session_state.step == 2:

    st.write("COLMAP Sparse Reconstruction")

    st.write(
        f"Frames available: {st.session_state.frame_count}"
    )


    # ========================================================
    # START SPARSE RECONSTRUCTION
    # ========================================================

    if st.button("Start Sparse Reconstruction"):

        # Clear previous sparse model
        for item in Sparse_DIR.iterdir():

            if item.is_file() or item.is_symlink():
                item.unlink()

            elif item.is_dir():
                shutil.rmtree(item)


        # Clear old database
        if DATABASE_PATH.exists():
            DATABASE_PATH.unlink()


        # Clear current UI log
        st.session_state.logs_reconstruct = ""


        # ====================================================
        # COLMAP COMMAND
        # ====================================================

        Sparse_Cmd = f"""
        colmap feature_extractor \
        --database_path "{DATABASE_PATH}" \
        --image_path "{FRAMES_DIR}" && \
        colmap sequential_matcher \
        --database_path "{DATABASE_PATH}" && \
        colmap mapper \
        --database_path "{DATABASE_PATH}" \
        --image_path "{FRAMES_DIR}" \
        --output_path "{Sparse_DIR}"
        """


        # ====================================================
        # WRITE NEW RUN HEADER ONCE
        # ====================================================

        with open(
            LOG_FILE_SparseReconstruction,
            "a"
        ) as log_file:

            log_file.write(
                f"\n\n{'=' * 50}\n"
                f"NEW RUN: {datetime.now()}\n"
                f"{'=' * 50}\n"
            )


        # ====================================================
        # RUN COLMAP
        # ====================================================

        process = subprocess.Popen(

            ["bash", "-c", Sparse_Cmd],

            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,

            text=True,
            bufsize=1
        )


        # ====================================================
        # READ OUTPUT
        # ====================================================

        for line in process.stdout:

            st.session_state.logs_reconstruct += line

            with open(
                LOG_FILE_SparseReconstruction,
                "a"
            ) as log_file:

                log_file.write(line)


        # ====================================================
        # WAIT FOR COMPLETION
        # ====================================================

        return_code = process.wait()


        if return_code == 0:

            st.session_state.sparse_reconstruct = True

            st.success(
                "✓ Sparse Reconstruction completed"
            )

        else:

            st.session_state.sparse_reconstruct = False

            st.error(
                "Sparse Reconstruction failed"
            )


    # ========================================================
    # SHOW RESULT AFTER RECONSTRUCTION
    # ========================================================

    if st.session_state.sparse_reconstruct:

        st.success(
            "✓ Sparse Reconstruction completed"
        )

        st.caption(
            f"Saved to: {Sparse_DIR}"
        )


        with st.expander("Show processing log"):

            st.code(
                st.session_state.logs_reconstruct,
                language="text"
            )

            st.caption(
                f"Full log saved at: "
                f"{LOG_FILE_SparseReconstruction}"
            )


        # NEXT SHOULD ONLY EXIST AFTER SUCCESS
        if st.button("Next →"):

            st.session_state.step = 3

            st.rerun()


    if st.button("← Back"):

        st.session_state.step = 1

        st.rerun()

elif st.session_state.step==3:
        st.write("COLMAP Phase 3")
        if st.button("← Back to 1"):
            st.session_state.step = 1
            st.rerun()
