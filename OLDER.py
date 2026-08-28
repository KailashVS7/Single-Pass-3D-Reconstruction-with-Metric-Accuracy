import streamlit as st
import subprocess
from pathlib import Path


# ============================================================
# FOLDERS
# ============================================================

BASE_DIR = Path(__file__).parent

INPUT_DIR = BASE_DIR / "input"
FRAMES_DIR = BASE_DIR / "frames"
LOG_FILE = BASE_DIR / "run.log"

INPUT_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)


# ============================================================
# SESSION STATE
# ============================================================

if "step" not in st.session_state:
    st.session_state.step = 1

if "extraction_done" not in st.session_state:
    st.session_state.extraction_done = False

if "logs" not in st.session_state:
    st.session_state.logs = ""

if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0


# ============================================================
# STEP 1
# ============================================================

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


    # ========================================================
    # FRAME EXTRACTION
    # ========================================================

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

        LOG_FILE.write_text("")


        # FFmpeg command
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),

            "-vf",
            f"fps={fps}",

            str(FRAMES_DIR / "frame_%05d.jpg")
        ]


        # ----------------------------------------------------
        # CLEAN APP STATUS
        # ----------------------------------------------------

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
                with open(LOG_FILE, "a") as log_file:
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


    # ========================================================
    # RESULT
    # ========================================================

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
                f"Full log saved at: {LOG_FILE}"
            )


        # ----------------------------------------------------
        # NEXT
        # ----------------------------------------------------

        if st.button("Next →"):

            st.session_state.step = 2

            st.rerun()



# ============================================================
# STEP 2
# ============================================================

elif st.session_state.step == 2:

    st.write("COLMAP Sparse Reconstruction")

    st.write(
        f"Frames available: {st.session_state.frame_count}"
    )


    if st.button("← Back"):

        st.session_state.step = 1

        st.rerun()