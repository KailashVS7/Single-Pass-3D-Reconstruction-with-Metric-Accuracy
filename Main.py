import streamlit as st
import subprocess
from pathlib import Path
from datetime import datetime
import shutil
import cv2 
import numpy as np
import os

ANALYSIS_RATE = 10
NORMALIZE_SELECTED_FRAMES = False

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

def resize_for_analysis(frame, width=640):

    height, original_width = frame.shape[:2]

    if original_width <= width:
        return frame

    scale = width / original_width
    new_height = int(height * scale)

    return cv2.resize(
        frame,
        (width, new_height)
    )


def sharp_enough(frame, threshold=80):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    return blur_score >= threshold, blur_score


def exposure_ok(
    frame,
    min_brightness=35,
    max_brightness=220
):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    brightness = gray.mean()

    good = (
        min_brightness
        <= brightness
        <= max_brightness
    )

    return good, brightness


def normalize_illumination(frame):

    lab = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    corrected = cv2.merge(
        (l, a, b)
    )

    return cv2.cvtColor(
        corrected,
        cv2.COLOR_LAB2BGR
    )


def motion_overlap_ok(
    previous_gray,
    current_gray,
    min_motion=6,
    max_motion=100,
    min_tracks=30,
    min_track_ratio=0.35
):

    points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=300,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7
    )

    if points is None or len(points) < min_tracks:
        return False, 0.0, 0, 0.0

    new_points, status, error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None
    )

    if new_points is None or status is None:
        return False, 0.0, 0, 0.0

    status = status.reshape(-1).astype(bool)

    old_good = points.reshape(-1, 2)[status]
    new_good = new_points.reshape(-1, 2)[status]

    valid_tracks = len(new_good)

    track_ratio = valid_tracks / len(points)

    if valid_tracks < min_tracks:
        return False, 0.0, valid_tracks, track_ratio

    displacement = np.linalg.norm(
        new_good - old_good,
        axis=1
    )

    median_motion = float(
        np.median(displacement)
    )

    keep = (
        median_motion >= min_motion
        and median_motion <= max_motion
        and track_ratio >= min_track_ratio
    )

    return (
        keep,
        median_motion,
        valid_tracks,
        track_ratio
    )

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

    st.write("Adaptive Frame Selection")

    uploaded_video = st.file_uploader(
        "Drop MP4 Video",
        type=["mp4"]
    )

    extract_button = st.button(
        "Analyse & Select Frames",
        disabled=uploaded_video is None
    )

    if st.button(
        "JUMP TO 3 Skip sparse Dev Option :) →"
    ):

        st.session_state.step = 3
        st.rerun()


    if extract_button:

        video_path = (
            INPUT_DIR / uploaded_video.name
        )

        with open(video_path, "wb") as f:

            f.write(
                uploaded_video.getbuffer()
            )


        for frame in FRAMES_DIR.glob("*.jpg"):

            frame.unlink()


        st.session_state.logs = ""
        st.session_state.extraction_done = False


        video = cv2.VideoCapture(
            str(video_path)
        )


        if not video.isOpened():

            st.error(
                "Could not open video."
            )

            st.stop()


        source_fps = video.get(
            cv2.CAP_PROP_FPS
        )

        total_frames = int(
            video.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )


        if source_fps > 0:

            candidate_step = max(
                1,
                int(
                    round(
                        source_fps /
                        ANALYSIS_RATE
                    )
                )
            )

        else:

            candidate_step = 1


        run_header = (
            f"\n\n{'=' * 60}\n"
            f"NEW ADAPTIVE FRAME RUN: {datetime.now()}\n"
            f"Video: {uploaded_video.name}\n"
            f"Source FPS: {source_fps:.2f}\n"
            f"Candidate step: {candidate_step}\n"
            f"{'=' * 60}\n"
        )


        st.session_state.logs += run_header


        with open(
            LOG_FILE_FrameExtraction,
            "a"
        ) as log_file:

            log_file.write(run_header)


        frame_number = 0
        saved_number = 0
        rejected_blur = 0
        rejected_exposure = 0
        rejected_motion = 0

        last_keyframe_gray = None


        with st.status(
            "Analysing video...",
            expanded=True
        ) as status:

            st.write(
                f"Video: {uploaded_video.name}"
            )

            st.write(
                f"Source FPS: {source_fps:.2f}"
            )


            progress = st.progress(0)

            progress_text = st.empty()


            while True:

                success, frame = video.read()


                if not success:

                    break


                if total_frames > 0:

                    percent = min(
                        frame_number /
                        total_frames,
                        1.0
                    )

                    progress.progress(
                        percent
                    )


                if frame_number % candidate_step != 0:

                    frame_number += 1
                    continue


                small = resize_for_analysis(
                    frame
                )


                sharp, blur_score = sharp_enough(
                    small
                )


                if not sharp:

                    rejected_blur += 1
                    frame_number += 1
                    continue


                good_exposure, brightness = exposure_ok(
                    small
                )


                if not good_exposure:

                    rejected_exposure += 1
                    frame_number += 1
                    continue


                current_gray = cv2.cvtColor(
                    small,
                    cv2.COLOR_BGR2GRAY
                )


                if last_keyframe_gray is None:

                    keep_frame = True
                    median_motion = 0
                    valid_tracks = 0
                    track_ratio = 1.0


                else:

                    (
                        keep_frame,
                        median_motion,
                        valid_tracks,
                        track_ratio

                    ) = motion_overlap_ok(

                        last_keyframe_gray,
                        current_gray
                    )


                if not keep_frame:

                    rejected_motion += 1
                    frame_number += 1
                    continue


                saved_number += 1

                frame_to_save = frame


                if NORMALIZE_SELECTED_FRAMES:

                    frame_to_save = (
                        normalize_illumination(
                            frame_to_save
                        )
                    )


                output_path = (
                    FRAMES_DIR /
                    f"frame_{saved_number:05d}.jpg"
                )


                cv2.imwrite(
                    str(output_path),
                    frame_to_save
                )


                last_keyframe_gray = (
                    current_gray.copy()
                )


                log_line = (
                    f"KEEP source_frame={frame_number} "
                    f"-> frame_{saved_number:05d}.jpg | "
                    f"blur={blur_score:.1f} | "
                    f"brightness={brightness:.1f} | "
                    f"motion={median_motion:.1f}px | "
                    f"tracks={valid_tracks} | "
                    f"overlap={track_ratio:.2f}\n"
                )


                st.session_state.logs += (
                    log_line
                )


                with open(
                    LOG_FILE_FrameExtraction,
                    "a"
                ) as log_file:

                    log_file.write(
                        log_line
                    )


                progress_text.write(
                    f"Selected: {saved_number} frames"
                )


                frame_number += 1


            video.release()

            progress.progress(1.0)

            st.session_state.frame_count = (
                saved_number
            )


            if saved_number >= 2:

                st.session_state.extraction_done = True


                summary = (
                    f"\nSelection complete\n"
                    f"Selected frames: {saved_number}\n"
                    f"Rejected blur: {rejected_blur}\n"
                    f"Rejected exposure: {rejected_exposure}\n"
                    f"Rejected motion/overlap: {rejected_motion}\n"
                )


                st.session_state.logs += summary


                with open(
                    LOG_FILE_FrameExtraction,
                    "a"
                ) as log_file:

                    log_file.write(
                        summary
                    )


                status.update(
                    label="Adaptive frame selection complete",
                    state="complete",
                    expanded=False
                )


            else:

                st.session_state.extraction_done = False


                status.update(
                    label="Too few usable frames selected",
                    state="error",
                    expanded=True
                )


    if st.session_state.extraction_done:

        st.success(
            f"✓ {st.session_state.frame_count} "
            f"keyframes selected"
        )

        st.caption(
            f"Saved to: {FRAMES_DIR}"
        )


        with st.expander(
            "Show processing log"
        ):

            st.code(
                st.session_state.logs,
                language="text"
            )

            st.caption(
                f"Full log saved at: "
                f"{LOG_FILE_FrameExtraction}"
            )


        if st.button("Next →"):

            st.session_state.step = 2
            st.rerun()


        #if st.button(
        #    "JUMP TO 3 Skip sparse Dev Option :) →"
        #):

         #   st.session_state.step = 3
          #  st.rerun()

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

        colmap_env = os.environ.copy()
        colmap_env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
        colmap_env.pop("QT_QPA_FONTDIR", None)
        colmap_env.pop("QT_PLUGIN_PATH", None)

        process = subprocess.Popen(

            ["bash", "-c", Sparse_Cmd],

            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,

            text=True,
            bufsize=1,
            env=colmap_env
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
